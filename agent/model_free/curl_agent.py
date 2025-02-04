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
from model.transition_model import make_transition_model
from agent.model_free.base_agent import AgentSACBase
from utils.pytorch_util import weight_init
import utils.data_augs as rad 

LOG_FREQ = 10000

class CURL(nn.Module):
    """
    CURL
    """

    def __init__(self, obs_shape, z_dim, batch_size, critic, critic_target, output_type="continuous"):
        super(CURL, self).__init__()
        self.batch_size = batch_size

        self.encoder = critic.encoder

        self.encoder_target = critic_target.encoder 

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

    def encode(self, x, detach=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)

        if detach:
            z_out = z_out.detach()
        return z_out

    #def update_target(self):
    #    utils.soft_update_params(self.encoder, self.encoder_target, 0.05)

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits


class AgentCURL(AgentSACBase):
    def __init__(
        self, 
        args,
        obs_shape: int,
        action_shape: int,
        device: Union[torch.device, str],
        init_temperature: float = 0.01,
        alpha_lr: float = 1e-3,
        alpha_beta: float = 0.9,
        encoder_lr: float = 1e-3,
        cpc_update_freq: int = 1,
        log_interval: int = 100,
        detach_encoder: bool = False,
        curl_latent_dim: int = 128,
        data_augs: str = ''
    ):
        super().__init__(args, obs_shape, action_shape, device, init_temperature, alpha_lr, alpha_beta)
        
        self.encoder_lr = encoder_lr
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.detach_encoder = detach_encoder
        self.curl_latent_dim = curl_latent_dim
        self.image_size = obs_shape[-1]
        self.data_augs = data_augs

        self.augs_funcs = {}

        aug_to_func = {
                'crop':rad.random_crop,
                'grayscale':rad.random_grayscale,
                'cutout':rad.random_cutout,
                'cutout_color':rad.random_cutout_color,
                'flip':rad.random_flip,
                'rotate':rad.random_rotation,
                'rand_conv':rad.random_convolution,
                'color_jitter':rad.random_color_jitter,
                'translate':rad.random_translate,
                'no_aug':rad.no_aug,
            }

        for aug_name in self.data_augs.split('-'):
            assert aug_name in aug_to_func, 'invalid data aug string'
            self.augs_funcs[aug_name] = aug_to_func[aug_name]

        if self.encoder_type == 'pixel':
            # create CURL encoder (the 128 batch size is probably unnecessary)
            self.CURL = CURL(obs_shape, 
                             self.encoder_feature_dim,
                             self.curl_latent_dim, 
                             self.critic,
                             self.critic_target, 
                             output_type='continuous').to(self.device)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            self.cpc_optimizer = torch.optim.Adam(
                self.CURL.parameters(), lr=encoder_lr
            )
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        self.train()
        self.critic_target.train()

        self.data_buffer = make_replay_buffer(args, action_shape, device)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_type == 'pixel':
            self.CURL.train(training)
    
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
        if obs.shape[-1] != self.image_size:
            obs = util.center_crop_image(obs, self.image_size)
 
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
        current_Q1, current_Q2 = self.critic(
            obs, action, detach_encoder=self.detach_encoder)
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

        # self.critic.log(L, step)
        return critic_log_dict
    
    def cpc_loss(self, obs_anchor, obs_pos, cpc_kwargs):
        loss_dict, log_dict = {}, {}
        z_a = self.CURL.encode(obs_anchor)
        z_pos = self.CURL.encode(obs_pos, ema=True)
        
        logits = self.CURL.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        cpc_loss = self.cross_entropy_loss(logits, labels)

        loss_dict['cpc_loss'] = cpc_loss
        log_dict['train/cpc_loss'] = cpc_loss.item()
        return loss_dict, log_dict
    
    def update_cpc(self, obs_anchor, obs_pos, cpc_kwargs):
        # time flips 
        """
        time_pos = cpc_kwargs["time_pos"]
        time_anchor= cpc_kwargs["time_anchor"]
        obs_anchor = torch.cat((obs_anchor, time_anchor), 0)
        obs_pos = torch.cat((obs_anchor, time_pos), 0)
        """
        cpc_loss_dict, cpc_log_dict = self.cpc_loss(obs_anchor, obs_pos, cpc_kwargs)

        loss = cpc_loss_dict['cpc_loss']
        self.encoder_optimizer.zero_grad()
        self.cpc_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.cpc_optimizer.step()
        
        return cpc_log_dict

    def update(self, step):
        loss_dict = {}
        if self.encoder_type == 'pixel':
            obs, action, reward, next_obs, not_done, cpc_kwargs = self.data_buffer.sample_cpc()#self.augs_funcs)
        else:
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
        if step % self.cpc_update_freq == 0 and self.encoder_type == 'pixel':
            obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
            cpc_log_dict = self.update_cpc(obs_anchor, obs_pos,cpc_kwargs)
            loss_dict.update(cpc_log_dict)
        
        return loss_dict

    def save_curl(self, model_dir, step):
        torch.save(
            self.CURL.state_dict(), '%s/curl_%s.pt' % (model_dir, step)
        )
