import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import copy
import math
from typing import Any, Dict, Optional, Type, Union, List

import utils.util as util
from model.decoder import make_decoder
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
    def __init__(self, 
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
        encoder_type: str = 'pixel',
        encoder_feature_dim: int = 50,
        encoder_tau: float = 0.005,
        encoder_lr: float = 1e-3,
        decoder_type: str = 'pixel',
        decoder_lr: float = 1e-3,
        decoder_update_freq: int = 1,
        decoder_latent_lambda: float = 0.0,
        decoder_weight_lambda: float = 0.0,
        num_layers: int = 4,
        num_filters: int = 32
    ):
        super(AgentSACBase, self).__init__(obs_shape, action_shape, device, hidden_dim, discount, init_temperature, alpha_lr, alpha_beta, actor_lr, actor_beta, actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr, critic_beta, critic_tau, critic_target_update_freq, encoder_type, encoder_feature_dim, encoder_tau, num_layers, num_filters)

        self.decoder_update_freq = decoder_update_freq
        self.decoder_latent_lambda = decoder_latent_lambda

        # distractor dynamic model
        self._build_dis_models()

        # task dynamic model
        self._build_task_models()

        # joint decode
        self._build_main_decoder()

        self.train()
        self.critic_target.train()


    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.decoder is not None:
            self.decoder.train(training)

    def _build_decoder(self, encoder_lr, decoder_lr, decoder_weight_lambda, decoder_type, obs_shape, encoder_feature_dim, num_layers, num_filters):
        self.decoder = None
        if decoder_type != 'identity':
            # create decoder
            self.decoder = make_decoder(
                decoder_type, obs_shape, encoder_feature_dim, num_layers,
                num_filters
            ).to(self.device)
            self.decoder.apply(weight_init)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            # optimizer for decoder
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(),
                lr=decoder_lr,
                weight_decay=decoder_weight_lambda
            )

    def _build_dis_models(self):
        self.dis_encoder = ...
        self.dis_dynamics = ...
        self.dis_decoder = ...
        self.dis_reward = ...
        self.dis_optimizer = torch.optim.Adam(
            list(self.dis_encoder.parameters()) + 
            list(self.dis_dynamics.parameters()) + 
            list(self.dis_decoder.parameters())
            )
        self.dis_reward_optimizer = torch.optim.Adam(
            self.dis_reward.parameters()
        )

    def _build_task_models(self):
        self.task_encoder = ...
        self.task_dynamics = ...
        self.task_reward = ...
        self.task_optimizer = torch.optim.Adam(
            list(self.task_reward.parameters()) + 
            list(self.task_dynamics.parameters()) + 
            list(self.task_reward.parameters())
            )

    def _build_main_decoder(self):
        self.main_mask_decoder = ...
        self.dis_mask_decoder = ...
        self.joint_mask_decoder = ...
        self.joint_mask_decoder_optimizer = torch.optim.Adam(self.joint_mask_decoder.parameters())

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
        embed = self.task_encoder(obs)
        post, prior = self.task_dynamics(embed, action)



        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

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