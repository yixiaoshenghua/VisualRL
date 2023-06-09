import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from typing import Any, Dict, Optional, Type, Union, List

import utils.util as util
from model.actor import Actor
from model.critic import Critic
from utils.pytorch_util import weight_init

LOG_FREQ = 10000

class AgentBase:
    def __init__(
        self, 
        obs_shape: int, 
        action_shape: int, 
        action_range: float, 
        device: Union[torch.device, str], 
        hidden_dim: int = 256,
        discount: float = 0.99,
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
        num_layers: int = 4,
        num_filters: int = 32,
        builtin_encoder: bool = True,
        ):
        self.action_range = action_range
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq

        self.critic = self._build_critic(obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, device, builtin_encoder)
        self.critic_target = self._build_critic_target(obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, builtin_encoder)
        self.actor = self._build_actor(obs_shape, action_shape, hidden_dim, encoder_type, encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, device, builtin_encoder)
        # build optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

    def _build_actor(self, obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, device, builtin_encoder=True):
        actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, device, builtin_encoder
        ).to(self.device)
        
        # tie encoders between actor and critic
        actor.encoder.copy_conv_weights_from(self.critic.encoder)
        return actor

    def _build_critic(self, obs_shape, action_shape, hidden_dim, encoder_type, encoder_feature_dim,    
            num_layers, num_filters, device, builtin_encoder=True):
        critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, device, builtin_encoder
        ).to(self.device)
        return critic


    def _build_critic_target(self, obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, builtin_encoder=True):
        critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, builtin_encoder
        ).to(self.device)
        critic_target.load_state_dict(self.critic.state_dict())
        return critic_target

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            mu = mu.clamp(*self.action_range)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            pi = pi.clamp(*self.action_range)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)
        return critic_loss.item()

    def update_actor(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), self.target_entropy.item(), entropy.mean().item()

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        loss_dict={}

        loss_dict['train/batch_reward'] = reward.mean()

        critic_loss = self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        loss_dict['train_critic/loss'] = critic_loss

        if step % self.actor_update_freq == 0:
            actor_loss, target_entropy, entropy = self.update_actor(obs, step)
            loss_dict['train_actor/loss'] = actor_loss
            loss_dict['train_actor/target_entropy'] = target_entropy
            loss_dict['train_actor/entropy'] = entropy

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

        return loss_dict

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )


class AgentSACBase(AgentBase):
    def __init__(
        self, 
        obs_shape: int,
        action_shape: int,
        action_range: float, 
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
        num_layers: int = 4,
        num_filters: int = 32,
        builtin_encoder: bool = True,
    ):
        super().__init__(obs_shape, action_shape, action_range, device, hidden_dim, discount, actor_lr, actor_beta, actor_log_std_min, actor_log_std_max, 
                                        actor_update_freq, critic_lr, critic_beta, critic_tau, critic_target_update_freq, encoder_type, encoder_feature_dim, encoder_tau, 
                                        num_layers, num_filters, builtin_encoder)

        self.log_alpha = self._build_log_alpha(init_temperature, action_shape, alpha_lr, alpha_beta)
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

    def _build_log_alpha(self, init_temperature, action_shape, alpha_lr, alpha_beta):
        log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)
        return log_alpha        

    def train(self, training=True):
        super().train(training)
        if self.decoder is not None:
            self.decoder.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update_actor_and_alpha(self, obs, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        return actor_loss.item(), self.target_entropy.item(), entropy.mean().item(), alpha_loss.item(), self.alpha.item()

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        loss_dict = {}

        loss_dict['train/batch_reward'] = reward.mean()

        critic_loss = self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        loss_dict['train_critic/loss'] = critic_loss

        if step % self.actor_update_freq == 0:
            actor_loss, target_entropy, entropy, alpha_loss, alpha = self.update_actor_and_alpha(obs, step)
            loss_dict['train_actor/loss'] = actor_loss
            loss_dict['train_actor/target_entropy'] = target_entropy
            loss_dict['train_actor/entropy'] = entropy
            loss_dict['train_alpha/loss'] = alpha_loss
            loss_dict['train_alpha/value'] = alpha

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

