import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from typing import Any, Dict, Optional, Type, Union, List

import utils.util as util
from model.decoder import make_decoder
from model.actor import Actor
from model.critic import Critic
from model.transition_model import make_transition_model
from agent.model_free.base_agent import AgentSACBase
from utils.pytorch_util import weight_init


LOG_FREQ = 10000

class AgentDBC(AgentSACBase):
    def __init__(
        self, 
        obs_shape: int,
        action_shape: int,
        action_range: float,
        device: Union[torch.device, str],
        transition_model_type: str,
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
        decoder_lr: float = 1e-3,
        decoder_update_freq: int = 1,
        decoder_latent_lambda: float = 0.0,
        decoder_weight_lambda: float = 0.0,
        num_layers: int = 4,
        num_filters: int = 32,
        bisim_coef: float = 0.5,
        builtin_encoder: bool = True
    ):
        super().__init__(obs_shape, 
                         action_shape, 
                         action_range, 
                         device, 
                         hidden_dim,
                         discount, 
                         init_temperature, 
                         alpha_lr, 
                         alpha_beta, 
                         actor_lr, 
                         actor_beta, 
                         actor_log_std_min, 
                         actor_log_std_max, 
                         actor_update_freq, 
                         critic_lr, 
                         critic_beta, 
                         critic_tau, 
                         critic_target_update_freq, 
                         encoder_type, 
                         encoder_feature_dim, 
                         encoder_tau, 
                         num_layers, 
                         num_filters, 
                         builtin_encoder)

        self.decoder_update_freq = self.args.decoder_update_freq
        self.decoder_latent_lambda = self.args.decoder_latent_lambda
        self.transition_model_type = self.args.transition_model_type
        self.bisim_coef = self.args.bisim_coef

        self.transition_model = make_transition_model(
            self.args.transition_model_type, 
            self.args.encoder_feature_dim, 
            action_shape
        ).to(device)

        self.reward_decoder = nn.Sequential(
            nn.Linear(self.args.encoder_feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        ).to(device)

        # optimizer for decoder
        self.decoder_optimizer = torch.optim.Adam(
            list(self.reward_decoder.parameters()) + list(self.transition_model.parameters()),
            lr=float(self.args.decoder_lr),
            weight_decay=self.args.decoder_weight_lambda
        )

        # optimizer for critic encoder for reconstruction loss
        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(), lr=float(self.args.encoder_lr)
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=False)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)
        return critic_loss

    def update_encoder(self, obs, action, reward, step):
        h = self.critic.encoder(obs)            

        # Sample random states across episodes at random
        batch_size = obs.size(0)
        perm = np.random.permutation(batch_size)
        h2 = h[perm]

        with torch.no_grad():
            # action, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            pred_next_latent_mu1, pred_next_latent_sigma1 = self.transition_model(torch.cat([h, action], dim=1))
            # reward = self.reward_decoder(pred_next_latent_mu1)
            reward2 = reward[perm]
        if pred_next_latent_sigma1 is None:
            pred_next_latent_sigma1 = torch.zeros_like(pred_next_latent_mu1)
        if pred_next_latent_mu1.ndim == 2:  # shape (B, Z), no ensemble
            pred_next_latent_mu2 = pred_next_latent_mu1[perm]
            pred_next_latent_sigma2 = pred_next_latent_sigma1[perm]
        elif pred_next_latent_mu1.ndim == 3:  # shape (B, E, Z), using an ensemble
            pred_next_latent_mu2 = pred_next_latent_mu1[:, perm]
            pred_next_latent_sigma2 = pred_next_latent_sigma1[:, perm]
        else:
            raise NotImplementedError

        z_dist = F.smooth_l1_loss(h, h2, reduction='none')
        r_dist = F.smooth_l1_loss(reward, reward2, reduction='none')
        if self.transition_model_type == '':
            transition_dist = F.smooth_l1_loss(pred_next_latent_mu1, pred_next_latent_mu2, reduction='none')
        else:
            transition_dist = torch.sqrt(
                (pred_next_latent_mu1 - pred_next_latent_mu2).pow(2) +
                (pred_next_latent_sigma1 - pred_next_latent_sigma2).pow(2)
            )
            # transition_dist  = F.smooth_l1_loss(pred_next_latent_mu1, pred_next_latent_mu2, reduction='none') \
            #     +  F.smooth_l1_loss(pred_next_latent_sigma1, pred_next_latent_sigma2, reduction='none')
        # bisimulation metric
        bisimilarity = r_dist + self.discount * transition_dist
        loss = (z_dist - bisimilarity).pow(2).mean()
        return loss

    def update_transition_reward_model(self, obs, action, next_obs, reward, step):
        h = self.critic.encoder(obs)
        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(torch.cat([h, action], dim=1))
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        next_h = self.critic.encoder(next_obs)
        diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma
        loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))

        pred_next_latent = self.transition_model.sample_prediction(torch.cat([h, action], dim=1))
        pred_next_reward = self.reward_decoder(pred_next_latent)
        reward_loss = F.mse_loss(pred_next_reward, reward)
        total_loss = loss + reward_loss
        return loss, total_loss

    def update(self, replay_buffer, L, step):
        # There is a current reward in original DBC repo
        # obs, action, _, reward, next_obs, not_done = replay_buffer.sample()
        obs, action, reward, next_obs, not_done = replay_buffer.sample()

        loss_dict = {}

        loss_dict['train/batch_reward'] = reward.mean()

        critic_loss = self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        loss_dict['train_critic/loss'] = critic_loss

        transition_loss, transition_reward_loss = self.update_transition_reward_model(obs, action, next_obs, reward, step)
        loss_dict['train_ae/transition_loss'] = transition_loss

        encoder_loss = self.update_encoder(obs, action, reward, step)
        loss_dict['train_ae/encoder_loss'] = encoder_loss

        total_loss = self.bisim_coef * encoder_loss + transition_reward_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

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

            
    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.reward_decoder.state_dict(),
            '%s/reward_decoder_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        self.reward_decoder.load_state_dict(
            torch.load('%s/reward_decoder_%s.pt' % (model_dir, step))
        )
