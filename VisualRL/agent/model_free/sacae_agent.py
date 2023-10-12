import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


LOG_FREQ = 10000

class AgentSACAE(AgentSACBase):
    def __init__(
        self, 
        obs_shape: int, action_shape: int, action_range: list, device: Union[torch.device, str], 
        agent, 
        encoder_type, encoder_feature_dim, encoder_tau, num_layers, num_filters, hidden_dim, builtin_encoder, 
        actor_lr, actor_beta, actor_log_std_min, actor_log_std_max, actor_update_freq, 
        critic_lr, critic_beta, critic_tau, critic_target_update_freq, 
        pre_transform_image_size, image_size, framestack, 
        buffer_size, batch_size, 
        discount, 
        init_temperature, alpha_lr, alpha_beta, 
        encoder_lr: float = 1e-3,
        decoder_type: str = 'pixel',
        decoder_lr: float = 1e-3,
        decoder_update_freq: int = 1,
        decoder_latent_lambda: float = 0.0,
        decoder_weight_lambda: float = 0.0
    ):
        super().__init__(
            obs_shape, action_shape, action_range, device, 
            agent, 
            encoder_type, encoder_feature_dim, encoder_tau, num_layers, num_filters, hidden_dim, builtin_encoder, 
            actor_lr, actor_beta, actor_log_std_min, actor_log_std_max, actor_update_freq, 
            critic_lr, critic_beta, critic_tau, critic_target_update_freq, 
            pre_transform_image_size, image_size, framestack, 
            buffer_size, batch_size, 
            discount, 
            init_temperature, alpha_lr, alpha_beta
        )

        self.encoder_lr = encoder_lr
        self.decoder_type = decoder_type
        self.decoder_lr = decoder_lr
        self.decoder_update_freq = decoder_update_freq
        self.decoder_latent_lambda = decoder_latent_lambda
        self.decoder_weight_lambda = decoder_weight_lambda

        self.decoder = self._build_decoder(decoder_type, obs_shape, self.encoder_feature_dim, self.num_layers, self.num_filters)
        if decoder_type != 'identity':
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

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.decoder is not None:
            self.decoder.train(training)

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

    def decoder_loss(self, obs, target_obs):
        loss_dict, log_dict = {}, {}
        h = self.critic.encoder(obs)

        if target_obs.dim() == 4:
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = util.ae_preprocess_obs(target_obs)
        rec_obs = self.decoder(h)
        rec_loss = F.mse_loss(target_obs, rec_obs)

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        loss = rec_loss + self.decoder_latent_lambda * latent_loss
        loss_dict['ae_loss'] = loss
        log_dict['train/ae_loss'] = loss.item()
        return loss_dict, log_dict

    def update_decoder(self, obs, target_obs):
        decoder_loss_dict, decoder_log_dict = self.decoder_loss(obs, target_obs)
        loss = decoder_loss_dict['ae_loss']

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        # self.decoder.log(L, step, log_freq=LOG_FREQ)
        return decoder_log_dict

    def update(self, step):
        loss_dict = {}
        obs, action, reward, next_obs, not_done = self.data_buffer.sample()
        loss_dict['train/batch_reward'] = reward.mean()

        critic_log_dict = self.update_critic(obs, action, reward, next_obs, not_done)
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

        if self.decoder is not None and step % self.decoder_update_freq == 0:
            decoder_log_dict = self.update_decoder(obs, obs)
            loss_dict.update(decoder_log_dict)
        
        return loss_dict

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
