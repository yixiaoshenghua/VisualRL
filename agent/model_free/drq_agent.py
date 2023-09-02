import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import copy
import math
from typing import Any, Dict, Optional, Type, Union, List

import utils.util as util
from utils.replay_buffer import make_replay_buffer
from model.decoder import make_decoder
from model.actor import Actor
from model.critic import Critic
from agent.model_free.base_agent import AgentSACBase
from utils.pytorch_util import weight_init
import utils.data_augs as rad

class AgentDrQ(AgentSACBase):
    """Data regularized Q: actor-critic method for learning from pixels."""
    def __init__(self,
        args,
        obs_shape: int,
        action_shape: int,
        action_range: float,
        device: Union[torch.device, str],
        init_temperature: float = 0.01,
        alpha_lr: float = 1e-3,
        alpha_beta: float = 0.9
        ):
        super().__init__(obs_shape, action_shape, action_range, device, init_temperature, alpha_lr, alpha_beta)
        self.image_size = obs_shape[-1]
        self.decoder = None
        self.train()
        self.critic_target.train()
        self.data_buffer = make_replay_buffer(args, action_shape, device)

    def select_action(self, obs):
        obs = obs['image']
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            mu = mu.clamp(*self.action_range)
            return mu.cpu().data.numpy().flatten()
    
    def sample_action(self, obs):
        obs = obs['image']
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
        
    def critic_loss(self, obs, obs_aug, action, reward, next_obs, next_obs_aug, not_done):
        loss_dict, log_dict = {}, {}
        with torch.no_grad():
            _, next_action, log_prob, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (not_done * self.discount * target_V)

            _, next_action_aug, log_prob_aug, _ = self.actor(next_obs)
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
        
        loss_dict['critic_loss'] = critic_loss
        log_dict['train/critic_loss'] = critic_loss.item()
        return loss_dict, log_dict

    def update_critic(self, obs, obs_aug, action, reward, next_obs, next_obs_aug, not_done):
        critic_loss_dict, critic_log_dict = self.critic_loss(obs, obs_aug, action, reward, next_obs, next_obs_aug, not_done)
        # Optimize the critic
        critic_loss = critic_loss_dict['critic_loss']
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # self.critic.log(L, step)
        return critic_log_dict

    def update(self, L, step):
        loss_dict = {}
        obs, action, reward, next_obs, not_done, obs_aug, next_obs_aug = self.data_buffer.sample_aug()
        loss_dict['train/batch_reward'] = reward.mean()

        critic_log_dict = self.update_critic(obs, obs_aug, action, reward, next_obs, next_obs_aug, not_done)
        loss_dict.update(critic_log_dict)

        if step % self.actor_update_freq == 0:
            actor_log_dict = self.update_actor_and_alpha(obs, step)
            loss_dict.update(actor_log_dict)

        if step % self.critic_target_update_freq == 0:
            util.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)
            
        return loss_dict

    def save_DRQ(self, model_dir, step):
        params = dict(actor=self.actor, critic=self.critic)
        torch.save(
            params, f"{model_dir}/drq_{step}.pt"
        )
