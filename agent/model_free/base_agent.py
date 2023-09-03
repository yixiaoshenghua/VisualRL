import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from typing import Any, Dict, Optional, Type, Union, List

import utils.util as util
from utils.replay_buffer import make_replay_buffer
from model.actor import Actor
from model.critic import Critic
from utils.pytorch_util import weight_init

LOG_FREQ = 10000

class AgentBase:
    def __init__(
        self, 
        args,
        obs_shape: int, 
        action_shape: int, 
        device: Union[torch.device, str], 
        ):
        self.args = args
        self.device = device

        self.hidden_dim = self.args.hidden_dim
        self.discount = self.args.discount
        self.actor_lr = self.args.actor_lr
        self.actor_beta = self.args.actor_beta
        self.actor_log_std_min = self.args.actor_log_std_min
        self.actor_log_std_max = self.args.actor_log_std_max
        self.actor_update_freq = self.args.actor_update_freq
        self.critic_lr = self.args.critic_lr
        self.critic_beta = self.args.critic_beta
        self.critic_tau = self.args.critic_tau
        self.critic_target_update_freq = self.args.critic_target_update_freq
        self.encoder_type = self.args.encoder_type
        self.encoder_feature_dim = self.args.encoder_feature_dim
        self.encoder_tau = self.args.encoder_tau
        self.num_layers = self.args.num_layers
        self.num_filters = self.args.num_filters
        self.builtin_encoder = self.args.builtin_encoder

        # CURL doesn't add the tanh nonlinearity to the output of the fc layer
        self.output_logits = True if args.agent.lower() == 'curl' else False

        self.critic = self._build_critic(args, obs_shape, action_shape, 
                                         self.hidden_dim, 
                                         self.encoder_type,
                                         self.encoder_feature_dim, 
                                         self.num_layers, 
                                         self.num_filters, 
                                         self.output_logits,
                                         self.builtin_encoder)
        self.critic_target = self._build_critic_target(args, obs_shape, action_shape, 
                                                       self.hidden_dim, 
                                                       self.encoder_type,
                                                       self.encoder_feature_dim, 
                                                       self.num_layers, 
                                                       self.num_filters, 
                                                       self.output_logits,
                                                       self.builtin_encoder)
        self.actor = self._build_actor(args, obs_shape, action_shape, 
                                       self.hidden_dim, 
                                       self.encoder_type, 
                                       self.encoder_feature_dim, 
                                       self.actor_log_std_min, 
                                       self.actor_log_std_max,
                                       self.num_layers, 
                                       self.num_filters, 
                                       self.output_logits,
                                       self.builtin_encoder)
        # build optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr, betas=(self.actor_beta, 0.999)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, betas=(self.critic_beta, 0.999)
        )
        self.data_buffer = make_replay_buffer(args, action_shape, device)

    def _build_actor(self, args, obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, output_logits, builtin_encoder=True):
        actor = Actor(
            args, obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, output_logits, builtin_encoder
        ).to(self.device)
        
        # tie encoders between actor and critic
        actor.encoder.copy_conv_weights_from(self.critic.encoder)
        return actor

    def _build_critic(self, args, obs_shape, action_shape, hidden_dim, encoder_type, encoder_feature_dim,    
            num_layers, num_filters, output_logits, builtin_encoder=True):
        critic = Critic(
            args, obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, output_logits, builtin_encoder
        ).to(self.device)
        return critic


    def _build_critic_target(self, args, obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, output_logits, builtin_encoder=True):
        critic_target = Critic(
            args, obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, output_logits, builtin_encoder
        ).to(self.device)
        critic_target.load_state_dict(self.critic.state_dict())
        return critic_target

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def reset(self):
        return
    
    def select_action(self, obs):
        obs = obs['image']
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        obs = obs['image']
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def critic_loss(self, obs, action, reward, next_obs, not_done):
        loss_dict, log_dict = {}, {}
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
        
        loss_dict['critic_loss'] = critic_loss
        log_dict['train/critic_loss'] = critic_loss.item()
        return loss_dict, log_dict

    def update_critic(self, obs, action, reward, next_obs, not_done):
        critic_loss_dict, critic_log_dict = self.critic_loss(obs, action, reward, next_obs, not_done)
        
        # Optimize the critic
        critic_loss = critic_loss_dict['critic_loss']
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return critic_log_dict
    
    def actor_loss(self, obs):
        loss_dict, log_dict = {}, {}
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        
        # Save the losses
        loss_dict['actor_loss'] = actor_loss
        # Log the results of the actor
        log_dict['train/actor_loss'] = actor_loss.item()
        log_dict['train/entropy'] = entropy.mean()
        return loss_dict, log_dict

    def update_actor(self, obs):
        actor_loss_dict, actor_log_dict = self.actor_loss(obs)

        # optimize the actor
        actor_loss = actor_loss_dict['actor_loss']
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_log_dict

    def update(self, step):
        loss_dict={}
        obs, action, reward, next_obs, not_done = self.data_buffer.sample()
        loss_dict['train/batch_reward'] = reward.mean()

        critic_log_dict = self.update_critic(obs, action, reward, next_obs, not_done)
        loss_dict.update(critic_log_dict)

        if step % self.actor_update_freq == 0:
            actor_log_dict = self.update_actor(obs)
            loss_dict.update(actor_log_dict)

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
        args,
        obs_shape: int,
        action_shape: int,
        device: Union[torch.device, str],
        init_temperature: float = 0.01,
        alpha_lr: float = 1e-3,
        alpha_beta: float = 0.9
    ):
        super().__init__(args, obs_shape, action_shape, device)

        self.init_temperature = init_temperature
        self.alpha_lr = alpha_lr
        self.alpha_beta = alpha_beta

        self.log_alpha = self._build_log_alpha(init_temperature, 
                                               action_shape)
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )
        self.data_buffer = make_replay_buffer(args, action_shape, device)

    def _build_log_alpha(self, init_temperature, action_shape):
        log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)
        return log_alpha        

    def train(self, training=True):
        super().train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def actor_loss(self, obs):
        loss_dict, log_dict = {}, {}
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        
        # Save the losses
        loss_dict['actor_loss'] = actor_loss
        loss_dict['alpha_loss'] = alpha_loss
        # Log the results of actor and alpha
        log_dict['train/actor_loss'] = actor_loss.item()
        log_dict['train/entropy'] = entropy.mean()
        log_dict['train/alpha_loss'] = alpha_loss.item()
        log_dict['train/alpha'] = self.alpha
        return loss_dict, log_dict
    
    def update_actor_and_alpha(self, obs):
        actor_loss_dict, actor_log_dict = self.actor_loss(obs)
        actor_loss = actor_loss_dict['actor_loss']
        alpha_loss = actor_loss_dict['alpha_loss']
        
        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # optimize alpha
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return actor_log_dict

    def update(self, replay_buffer, L, step):
        loss_dict = {}
        obs, action, reward, next_obs, not_done = replay_buffer.sample()
        loss_dict['train/batch_reward'] = reward.mean()

        critic_log_dict = self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        loss_dict.update(critic_log_dict)

        if step % self.actor_update_freq == 0:
            actor_log_dict = self.update_actor_and_alpha(obs)
            loss_dict.update(actor_log_dict)

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
