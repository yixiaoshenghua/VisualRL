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
        obs_shape: int, action_shape: int, action_range: list, device: Union[torch.device, str], 
        agent, 
        encoder_type, encoder_feature_dim, encoder_tau, num_layers, num_filters, hidden_dim, builtin_encoder, 
        actor_lr, actor_beta, actor_log_std_min, actor_log_std_max, actor_update_freq, 
        critic_lr, critic_beta, critic_tau, critic_target_update_freq, 
        pre_transform_image_size, image_size, framestack, 
        buffer_size, batch_size, 
        discount, action_repeat, max_videos_to_save
        ):
        # ------------global config------------
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.action_range = action_range
        self.device = device
        # ------------agent config------------
        self.agent = agent
        # encoder
        self.encoder_type = encoder_type
        self.encoder_feature_dim = encoder_feature_dim
        self.encoder_tau = encoder_tau
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.hidden_dim = hidden_dim
        self.builtin_encoder = builtin_encoder
        # actor
        self.actor_lr = actor_lr
        self.actor_beta = actor_beta
        self.actor_log_std_min = actor_log_std_min
        self.actor_log_std_max = actor_log_std_max
        self.actor_update_freq = actor_update_freq
        # critic
        self.critic_lr = critic_lr
        self.critic_beta = critic_beta
        self.critic_tau = critic_tau
        self.critic_target_update_freq = critic_target_update_freq
        # constants
        self.discount = discount
        self.action_repeat = action_repeat
        self.max_videos_to_save = max_videos_to_save

        # CURL doesn't add the tanh nonlinearity to the output of the fc layer
        self.output_logits = True if self.agent == 'curl' else False

        self.critic = self._build_critic()
        self.critic_target = self._build_critic_target()
        self.actor = self._build_actor()
        # build optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr, betas=(self.actor_beta, 0.999)
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr, betas=(self.critic_beta, 0.999)
        )
        self.data_buffer = make_replay_buffer(
            action_shape, device, 
            agent, 
            pre_transform_image_size, image_size, framestack, 
            buffer_size, batch_size
        )

    def _build_actor(self):
        actor = Actor(
            self.agent, 
            self.obs_shape, 
            self.action_shape, 
            self.hidden_dim, 
            self.encoder_type, 
            self.encoder_feature_dim, 
            self.actor_log_std_min, 
            self.actor_log_std_max,
            self.num_layers, 
            self.num_filters, 
            self.output_logits,
            self.builtin_encoder
        ).to(self.device)
        
        # tie encoders between actor and critic
        actor.encoder.copy_conv_weights_from(self.critic.encoder)
        return actor

    def _build_critic(self):
        critic = Critic(
            self.agent, 
            self.obs_shape, 
            self.action_shape, 
            self.hidden_dim, 
            self.encoder_type,
            self.encoder_feature_dim, 
            self.num_layers, 
            self.num_filters, 
            self.output_logits,
            self.builtin_encoder
        ).to(self.device)
        return critic

    def _build_critic_target(self):
        critic_target = Critic(
            self.agent, 
            self.obs_shape, 
            self.action_shape, 
            self.hidden_dim, 
            self.encoder_type,
            self.encoder_feature_dim, 
            self.num_layers, 
            self.num_filters, 
            self.output_logits,
            self.builtin_encoder
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
    
    def act_and_collect_data(self, env, collect_steps, step, logger, log_interval=100, num_updates=1):
        obs = env.reset()
        done = False
        episode_rewards = [0.0]
        self.step = step

        for i in range(collect_steps):
            for _ in range(num_updates):
                loss_dict = self.update(self.step)
            
            if self.step % log_interval == 0:
                print('------------------------------------------')
                print('step: ', self.step)
                logger.log_scalars(loss_dict, self.step)
            
            with torch.no_grad():
                action = self.sample_action(obs)
            next_obs, rew, done, _ = env.step(action)
            self.data_buffer.add(obs, action, rew, next_obs, done)

            episode_rewards[-1] += rew

            if done:
                obs = env.reset()
                done = False
                if i != collect_steps-1:
                    episode_rewards.append(0.0)
            else:
                obs = next_obs 
            self.step += self.action_repeat

        return np.array(episode_rewards)
    
    def evaluate(self, env, eval_episodes, render=False):
        episode_rew = np.zeros((eval_episodes))
        video_images = [[] for _ in range(eval_episodes)]
        self.train(False)

        for i in range(eval_episodes):
            obs = env.reset()
            done = False

            while not done:
                with torch.no_grad():
                    action = self.select_action(obs)
                next_obs, rew, done, _ = env.step(action)
                episode_rew[i] += rew

                if render:
                    video_images[i].append(obs['image'].transpose(1, 2, 0).copy())
                obs = next_obs
        
        self.train(True)
        return episode_rew, np.array(video_images[:self.max_videos_to_save])#, pred_videos # (T, H, W, C)

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
        obs_shape: int, action_shape: int, action_range: list, device: Union[torch.device, str], 
        agent, 
        encoder_type, encoder_feature_dim, encoder_tau, num_layers, num_filters, hidden_dim, builtin_encoder, 
        actor_lr, actor_beta, actor_log_std_min, actor_log_std_max, actor_update_freq, 
        critic_lr, critic_beta, critic_tau, critic_target_update_freq, 
        pre_transform_image_size, image_size, framestack, 
        buffer_size, batch_size, 
        discount, action_repeat, max_videos_to_save, 
        init_temperature, alpha_lr, alpha_beta
    ):
        super().__init__(
            obs_shape, action_shape, action_range, device, 
            agent, 
            encoder_type, encoder_feature_dim, encoder_tau, num_layers, num_filters, hidden_dim, builtin_encoder, 
            actor_lr, actor_beta, actor_log_std_min, actor_log_std_max, actor_update_freq, 
            critic_lr, critic_beta, critic_tau, critic_target_update_freq, 
            pre_transform_image_size, image_size, framestack, 
            buffer_size, batch_size, 
            discount, action_repeat, max_videos_to_save
        )

        self.init_temperature = init_temperature
        self.alpha_lr = alpha_lr
        self.alpha_beta = alpha_beta

        self.log_alpha = self._build_log_alpha()
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.alpha_lr, betas=(self.alpha_beta, 0.999)
        )

    def _build_log_alpha(self):
        log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(self.action_shape)
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
