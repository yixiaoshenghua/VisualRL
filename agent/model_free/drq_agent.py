import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import copy
import math
from typing import Any, Dict, Optional, Type, Union, List

import utils.util as util
from model.decoder import make_decoder
from model.actor import Actor
from model.critic import Critic
from agent.model_free.base_agent import AgentSACBase
from utils.pytorch_util import weight_init
import utils.data_augs as rad

class AgentDrQ(AgentSACBase):
    """Data regularized Q: actor-critic method for learning from pixels."""
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
        detach_encoder: bool = False,
        batch_size: int = 64,
        builtin_encoder: bool = True,
        ):
        super(AgentSACBase, self).__init__(obs_shape, action_shape, device, hidden_dim, discount, init_temperature, alpha_lr, alpha_beta, actor_lr, actor_beta, actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr, critic_beta, critic_tau, critic_target_update_freq, encoder_type, encoder_feature_dim, encoder_tau, num_layers, num_filters, builtin_encoder)
        self.action_range = action_range
        self.batch_size = batch_size
        self.train()
        self.critic_target.train()

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
        if obs.shape[-1] != self.image_size:
            obs = util.center_crop_image(obs, self.image_size)
 
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(
                obs, compute_log_pi=False
            )
            pi = pi.clamp(*self.action_range)
            return pi.cpu().data.numpy().flatten()
        
    def update_critic(self, obs, obs_aug, action, reward, next_obs,
                      next_obs_aug, not_done, L, step):
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (not_done * self.discount * target_V)

            dist_aug = self.actor(next_obs_aug)
            next_action_aug = dist_aug.rsample()
            log_prob_aug = dist_aug.log_prob(next_action_aug).sum(-1,
                                                                  keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs_aug,
                                                      next_action_aug)
            target_V = torch.min(
                target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug
            target_Q_aug = reward + (not_done * self.discount * target_V)

            target_Q = (target_Q + target_Q_aug) / 2

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        Q1_aug, Q2_aug = self.critic(obs_aug, action)

        critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(
            Q2_aug, target_Q)

        L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug = replay_buffer.sample_aug(
            self.batch_size)

        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, obs_aug, action, reward, next_obs,
                           next_obs_aug, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            util.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

    def save_DRQ(self, model_dir, step):
        params = dict(actor=self.actor, critic=self.critic)
        torch.save(
            params, f"{model_dir}/drq_{step}.pt"
        )