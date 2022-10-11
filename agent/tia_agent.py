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
from agent.base_agent import AgentSACBase
from utils.pytorch_util import weight_init

LOG_FREQ = 10000


class TIA(nn.Module):
    """
    TIA
    """
    def __init__(
        self, obs_shape, z_dim,
        encoder, encoder_target, device, output_type="continuous"
    ):
        super(TIA, self).__init__()

        self.encoder = encoder

        self.encoder_target = encoder_target
        self.device = device

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

    def encode(
        self, x, actions, ema=False
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
            device=self.device, dtype=prev_actions.dtype
        ).unsqueeze(0)
        prev_actions = torch.cat([prev_act, prev_actions], dim=0)
        # Embed the pixel observation
        prev_state = self.encoder.representation.initial_state(
            batch_b, device=self.device
        )
        # Rollout model by taking the same series of actions as the real model
        if ema:
            with torch.no_grad():
                embeds = self.encoder_target.observation_encoder(x)
                prior, post = self.encoder_target.rollout.\
                    rollout_representation(batch_t, embeds,
                                           prev_actions, prev_state)
        else:
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
        buildin_encoder: bool = False,
    ):
        super(AgentSACBase, self).__init__(obs_shape, action_shape, device, hidden_dim, discount, init_temperature, alpha_lr, alpha_beta, actor_lr, actor_beta, actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr, critic_beta, critic_tau, critic_target_update_freq, encoder_type, encoder_feature_dim, encoder_tau, num_layers, num_filters)
        self.grad_clip = grad_clip
        self.reward_opt_num = reward_opt_num
        self.reward_scale = reward_scale
        self.free_nats = free_nats
        self.kl_scale = kl_scale
        self.disen_kl_scale = disen_kl_scale
        self.disen_neg_rew_scale = disen_neg_rew_scale
        self.disen_rec_scale = disen_rec_scale
        feature_dim = stochastic_size + deterministic_size

        # distractor dynamic model
        self.disen_model = make_rssm_encoder(
            encoder_type, obs_shape, action_shape, encoder_feature_dim,
            stochastic_size, deterministic_size, num_layers,
            num_filters, hidden_dim, output_logits=True
        ).to(self.device)
        self.disen_reward = DenseDecoder((), 2, feature_dim, num_units).to(self.device)
        self.disen_decoder = self._build_decoder(decoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters)

        # task dynamic model
        self.task_model = make_rssm_encoder(
            encoder_type, obs_shape, action_shape, encoder_feature_dim,
            stochastic_size, deterministic_size, num_layers,
            num_filters, hidden_dim, output_logits=True
        ).to(self.device)
        self.task_reward = DenseDecoder((), 2, feature_dim, num_units).to(self.device)

        self.disen_optimizer = torch.optim.Adam(list(self.disen_model.parameters()) + list(self.disen_decoder.parameters()), lr=encoder_lr)
        self.disen_reward_optimizer = torch.optim.Adam(self.disen_reward.parameters(), lr=disen_reward_lr)
        self.task_optimizer = torch.optim.Adam(list(self.task_model.parameters()) + list(self.task_reward.parameters()), lr=encoder_lr)
        # joint decode
        self.joint_mask_decoder = self._build_main_decoder()
        self.joint_mask_decoder_optimizer = torch.optim.Adam(self.joint_mask_decoder.parameters(), lr=encoder_lr)
        self.train()
        self.critic_target.train()


    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.decoder is not None:
            self.decoder.train(training)

    def _build_main_decoder(self, obs_shape, feature_dim, num_layers, num_filters, dtype):
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

    def update_decoder(self, obs, target_obs, L, step):
        h = self.critic.encoder(obs)

        if target_obs.dim() == 4:
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = util.preprocess_obs(target_obs)
        rec_obs = self.decoder(h)
        rec_loss = F.mse_loss(target_obs, rec_obs)

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        loss = rec_loss + self.decoder_latent_lambda * latent_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        L.log('train_ae/ae_loss', loss, step)

        self.decoder.log(L, step, log_freq=LOG_FREQ)

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        L.log('train/batch_reward', reward.mean(), step)

        # main
        embed = self.task_model.encoder(obs)
        post, prior = self.task_model.dynamics.observe(embed, action)
        feat = self.task_model.dynamics.get_feat(post)

        # disen
        embed_disen = self.disen_model.encoder(obs)
        post_disen, prior_disen = self.disen_model.dynamics.observe(embed_disen, action)
        feat_disen = self.disen_model.dynamics.get_feat(post_disen)

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

        prior_dist = self.task_model.dynamics.get_dist(prior)
        post_dist = self.task_model.dynamics.get_dist(post)
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

        prior_dist_disen = self.disen_model.dynamics.get_dist(prior_disen)
        post_dist_disen = self.disen_model.dynamics.get_dist(post_disen)
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
        image_feat = ...
        next_image_feat = ...
        reward = self.task_reward(image_feat).mode()
        self.update_critic(image_feat, action, reward, next_image_feat, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(image_feat, L, step)

        if step % self.critic_target_update_freq == 0:
            util.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            util.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            util.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

        if self.decoder is not None and step % self.decoder_update_freq == 0:
            self.update_decoder(obs, obs, L, step)

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