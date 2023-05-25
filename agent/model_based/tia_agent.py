import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import copy
import math
from typing import Any, Dict, Optional, Type, Union, List

import utils.util as util
from model.encoder import make_rssm_encoder
from model.decoder import make_decoder, MaskDecoder, EnsembleMaskDecoder, DenseDecoder
from model.recurrent_state_space_model import *
from model.actor import Actor
from model.critic import Critic
from agent.model_free.base_agent import AgentSACBase
from utils.pytorch_util import weight_init

LOG_FREQ = 10000


class TIA(nn.Module):
    """TIA"""
    def __init__(
        self, obs_shape, z_dim,
        encoder, output_type="continuous"
    ):
        super(TIA, self).__init__()
        self.encoder = encoder
        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

    def encode(
        self, x, actions, device='cuda'
    ):
        """
        Encoder: z_t = e(o_t, s_{t-1}, a_{t-1})
        :param x: obs at t
        :return: z_t, value in r2
        """
        batch_t, batch_b, ch, h, w = x.size()

        # Obtain prev actions
        prev_actions = actions[:-1]
        prev_act = torch.zeros(
            batch_b, self.encoder.action_shape,
            device=device, dtype=prev_actions.dtype
        ).unsqueeze(0)
        prev_actions = torch.cat([prev_act, prev_actions], dim=0)
        # Embed the pixel observation
        prev_state = self.encoder.representation.initial_state(
            batch_b, device=device
        )
        # Rollout model by taking the same series of actions as the real model
        embeds = self.encoder.observation_encoder(x)
        prior, post = self.encoder.rollout.rollout_representation(
            batch_t, embeds, prev_actions, prev_state
        )

        return prior, post


class AgentTIA(AgentSACBase):
    def __init__(
        self, 
        obs_shape: int,
        action_shape: int,
        device: Union[torch.device, str],
        hidden_dim: int = 256,
        discount: float = 0.99,
        init_temperature: float = 0.01,
        alpha_lr: float = 1e-3,
        alpha_beta: float = 0.9,
        actor_lr: float = 1e-3,
        actor_beta: float = 0.9,
        actor_log_std_min: float = -10,
        actor_log_std_max: float = 2,
        actor_update_freq: int = 2,
        critic_lr: float = 1e-3,
        critic_beta: float = 0.9,
        critic_tau: float = 0.005,
        critic_target_update_freq: int = 2,
        encoder_type: str = 'rssm',
        decoder_type: str = 'pixel',
        encoder_feature_dim: int = 50,
        encoder_tau: float = 0.005,
        encoder_lr: float = 6e-4,
        num_layers: int = 4,
        num_filters: int = 32,
        num_units: int = 400,
        stochastic_size: int = 30,
        deterministic_size: int = 200,
        grad_clip: float = 100.0,
        disen_reward_lr: float = 6e-4,
        reward_scale: float = 1.0,
        reward_opt_num: int = 20,
        free_nats: float = 3.0,
        kl_scale: float = 1.0,
        disen_kl_scale: float = 1.0,
        disen_neg_rew_scale: float = 20000.0,
        disen_rec_scale: float = 1.5,
        disclam: float = 0.95,
        batch_size: int = 50,
        seq_len: int = 50,
        builtin_encoder: bool = False,
    ):
        super(AgentSACBase, self).__init__(obs_shape, action_shape, device, hidden_dim, discount, init_temperature, alpha_lr, alpha_beta, actor_lr, actor_beta, actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr, critic_beta, critic_tau, critic_target_update_freq, encoder_type, encoder_feature_dim, encoder_tau, num_layers, num_filters, builtin_encoder)
        self.grad_clip = grad_clip
        self.reward_opt_num = reward_opt_num
        self.reward_scale = reward_scale
        self.free_nats = free_nats
        self.kl_scale = kl_scale
        self.disen_kl_scale = disen_kl_scale
        self.disen_neg_rew_scale = disen_neg_rew_scale
        self.disen_rec_scale = disen_rec_scale
        self.stoch_size = stochastic_size
        self.deter_size = deterministic_size
        self.disclam = disclam
        self.batch_size = batch_size
        self.seq_len = seq_len
        feature_dim = stochastic_size + deterministic_size

        # distractor dynamic model
        self.disen_encoder = make_rssm_encoder(
            encoder_type, obs_shape, action_shape, encoder_feature_dim,
            stochastic_size, deterministic_size, num_layers,
            num_filters, hidden_dim, output_logits=True
        ).to(self.device)
        self.disen_model = TIA(obs_shape, feature_dim, self.disen_encoder, output_type='continuous')
        self.disen_reward = DenseDecoder((), 2, feature_dim, num_units).to(self.device)
        self.disen_decoder = self._build_decoder(decoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters)

        # task dynamic model
        self.task_encoder = make_rssm_encoder(
            encoder_type, obs_shape, action_shape, encoder_feature_dim,
            stochastic_size, deterministic_size, num_layers,
            num_filters, hidden_dim, output_logits=True
        ).to(self.device)
        self.task_model = TIA(obs_shape, feature_dim, self.task_encoder, output_type='continuous')
        self.task_reward = DenseDecoder((), 2, feature_dim, num_units).to(self.device)

        self.actor = self._build_actor(obs_shape, action_shape, hidden_dim, encoder_type, encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, builtin_encoder)
        self.critic = DenseDecoder((feature_dim), 3, num_units).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.disen_optimizer = torch.optim.Adam(list(self.disen_model.parameters()) + list(self.disen_decoder.parameters()), lr=encoder_lr)
        self.disen_reward_optimizer = torch.optim.Adam(self.disen_reward.parameters(), lr=disen_reward_lr)
        self.task_optimizer = torch.optim.Adam(list(self.task_model.parameters()) + list(self.task_reward.parameters()), lr=encoder_lr)


        # joint decode
        self.joint_mask_decoder = self._build_main_decoder(obs_shape, feature_dim, num_layers, num_filters)
        self.joint_mask_decoder_optimizer = torch.optim.Adam(self.joint_mask_decoder.parameters(), lr=encoder_lr)


        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)

    def _build_main_decoder(self, obs_shape, feature_dim, num_layers, num_filters, dtype=torch.float32):
        main_mask_decoder = MaskDecoder(obs_shape, feature_dim, num_layers, num_filters).to(self.device)
        dis_mask_decoder = MaskDecoder(obs_shape, feature_dim, num_layers, num_filters).to(self.device)
        joint_mask_decoder = EnsembleMaskDecoder(main_mask_decoder, dis_mask_decoder, dtype).to(self.device)
        return joint_mask_decoder
        
    def _build_decoder(self, decoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters):
        decoder = None
        if decoder_type != 'identity':
            # create decoder
            decoder = make_decoder(
                decoder_type, obs_shape, encoder_feature_dim, num_layers,
                num_filters
            ).to(self.device)
            decoder.apply(weight_init)
        return decoder

    def imagine_ahead(self, post, planning_horizon=15):
        '''
        imagine_ahead is the function to draw the imaginary tracjectory using the dynamics model, actor, critic.
        Input: current state (posterior), current belief (hidden), policy, transition_model  # torch.Size([50, 30]) torch.Size([50, 200])
        Output: generated trajectory of features includes beliefs, prior_states, prior_means, prior_std_devs
                torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
        '''
        prev_state = flatten_states(post)

        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        T = planning_horizon
        prior_states = RSSMState(
            torch.zeros(T, self._stoch_size),
            torch.zeros(T, self._stoch_size),
            torch.zeros(T, self._stoch_size),
            torch.zeros(T, self._deter_size),
        )
        prior_states[0] = prev_state
        # Loop over time sequence
        for t in range(T - 1):
            prev_state = prior_states[t]
            prev_action = self.sample_action(get_feat(prev_state).detach())
            prior_states[t + 1] = self.task_model.encoder.transition(prev_action, prev_state)
        
        # Return new hidden states
        # imagined_traj = [beliefs, prior_states, prior_means, prior_std_devs]
        imag_feat = get_feat(prior_states)
        return imag_feat

    def update(self, replay_buffer, L, step):
        # obs (L, n, *img_size), action (L, n, acs_dim), reward (L, n), not_done (L, n)
        obs, action, reward, not_done = replay_buffer.sample_sequence(self.batch_size, self.seq_len)
        L.log('train/batch_reward', reward.mean(), step)

        # main, task_model is a TIA contains rssm_encoder
        # observation_encoder(obs) -> emb, representation(emb, prev_acs, prev_state) -> prior, post
        prior, post = self.task_model.encode(obs, action)
        feat = get_feat(post)

        # disen, disen_model is a TIA contains rssm_encoder
        prior_disen, post_disen = self.disen_model.encode(obs, action)
        feat_disen = get_feat(post_disen)

        # disen image pred
        image_pred_disen = self.disen_decoder(feat_disen)

        # joint image pred
        image_pred_joint, image_pred_joint_main, image_pred_joint_disen, mask_pred = self.joint_mask_decoder(feat, feat_disen)

        # reward pred
        reward_pred = self.task_reward(feat)

        # optimize disen reward predictor till optimal
        for _ in range(self.reward_opt_num):
            reward_pred_disen = self.disen_reward(feat_disen.detach())
            reward_like_disen = reward_pred_disen.log_prob(reward)
            reward_loss_disen = -reward_like_disen.mean()
            self.disen_reward_optimizer.zero_grad()
            reward_loss_disen.backward()
            self.disen_reward_optimizer.step()

        # disen reward pred with optimal reward predictor
        reward_pred_disen = self.disen_reward(feat_disen)
        reward_like_disen = torch.mean(reward_pred_disen.log_prob(reward))

        # main model loss
        likes = dict()
        likes['image'] = torch.mean(image_pred_joint.log_prob(obs))
        likes['reward'] = torch.mean(reward_pred.log_prob(reward)) * self.reward_scale

        prior_dist = get_dist(prior)
        post_dist = get_dist(post)
        div = torch.mean(torch.distributions.kl_divergence(prior_dist, post_dist))
        div = torch.max(div, self.free_nats)

        model_loss = self.kl_scale * div - sum(likes.values())

        # disen model loss with reward negative gradient
        likes_disen = dict()
        likes_disen['image'] = torch.mean(image_pred_joint.log_prob(obs))
        likes_disen['disen_only'] = torch.mean(image_pred_disen.log_prob(obs))

        reward_like_disen = reward_pred_disen.log_prob(reward)
        reward_like_disen = torch.mean(reward_like_disen)
        reward_loss_disen = -reward_like_disen

        prior_dist_disen = get_dist(prior_disen)
        post_dist_disen = get_dist(post_disen)
        div_disen = torch.mean(torch.distributions.kl_divergence(
            post_dist_disen, prior_dist_disen))
        div_disen = torch.max(div_disen, self.free_nats)

        model_loss_disen = div_disen * self.disen_kl_scale + \
            reward_like_disen * self.disen_neg_rew_scale - \
            likes_disen['image'] - likes_disen['disen_only'] * self.disen_rec_scale

        decode_loss = model_loss_disen + model_loss
        self.task_optimizer.zero_grad()
        self.disen_optimizer.zero_grad()
        self.joint_mask_decoder_optimizer.zero_grad()
        model_loss.backward()
        model_loss_disen.backward()
        decode_loss.backward()
        self.task_optimizer.step()
        self.disen_optimizer.step()
        self.joint_mask_decoder_optimizer.step()

        # update value and action
        image_feat = self.task_model.imagine_ahead(post, self.actor)
        rewards = self.task_reward(image_feat).mode()
        pcont = self.discount * torch.ones_like(rewards).to(self.device)
        values = self.critic(image_feat).mode()
        returns = util.lambda_return(rewards[:-1], values[:-1],
                               bootstrap=values[-1], discount=self.discount, lambda_=self.disclam)
        self.update_critic(image_feat, returns, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor(returns, L, step)

    def update_critic(self, imag_feat, returns, L, step):
        value_pred = self.critic(imag_feat)[:-1]
        target = returns.detach()
        critic_loss = -torch.mean(self.discount * value_pred.log_prob(target))
        L.log('train_critic/loss', critic_loss, step)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
    def update_actor(self, returns, L, step):
        actor_loss = -torch.mean(self.discount * returns)
        L.log('train_actor/loss', actor_loss, step)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        if self.decoder is not None:
            torch.save(
                self.decoder.state_dict(),
                '%s/decoder_%s.pt' % (model_dir, step)
            )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        if self.decoder is not None:
            self.decoder.load_state_dict(
                torch.load('%s/decoder_%s.pt' % (model_dir, step))
            )