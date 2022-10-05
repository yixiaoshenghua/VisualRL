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
from model.recurrent_state_space_model import get_feat, get_dist, flatten_states, RSSMState
from model.actor import Actor
from model.critic import Critic
from agent.base_agent import AgentSACBase
from utils.pytorch_util import weight_init


LOG_FREQ = 10000

class DRIBO(nn.Module):
    """
    DRIBO
    """
    def __init__(
        self, obs_shape, z_dim,
        encoder, encoder_target, device, output_type="continuous"
    ):
        super(DRIBO, self).__init__()

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

    def multi_view_encode(self, obs, z, actions, ema=False):
        batch_shape = obs.shape[:-3]
        img_shape = obs.shape[-3:]
        # Obtain prev states

        def unsqueeze_states(rssm_states, dim):
            return RSSMState(
                rssm_states.mean.unsqueeze(dim),
                rssm_states.std.unsqueeze(dim),
                rssm_states.stoch.unsqueeze(dim),
                rssm_states.deter.unsqueeze(dim),
            )

        def cat_states(rssm_states: list, dim):
            return RSSMState(
                torch.cat([state.mean for state in rssm_states], dim=dim),
                torch.cat([state.std for state in rssm_states], dim=dim),
                torch.cat([state.stoch for state in rssm_states], dim=dim),
                torch.cat([state.deter for state in rssm_states], dim=dim),
            )

        prev_states = z[:-1]
        prev_state = self.encoder.representation.initial_state(
            batch_shape[1], device=self.device
        )
        prev_state = unsqueeze_states(prev_state, dim=0)
        prev_states = cat_states([prev_state, prev_states], dim=0)
        # Obtain prev actions
        prev_actions = actions[:-1]
        prev_act = torch.zeros(
            batch_shape[1], self.encoder.action_shape,
            device=self.device, dtype=prev_actions.dtype
        ).unsqueeze(0)
        prev_actions = torch.cat([prev_act, prev_actions], dim=0)

        # Flatten inputs
        flatten_batch = np.prod(batch_shape)
        obs = torch.reshape(obs, (-1, *img_shape))
        prev_states = flatten_states(prev_states, flatten_batch)
        prev_actions = torch.reshape(prev_actions, (flatten_batch, -1))

        if ema:
            with torch.no_grad():
                states = self.encoder_target(obs, prev_actions, prev_states)
        else:
            states = self.encoder(obs, prev_actions, prev_states)
        return states

    def compute_logits(self, z1, z2):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z1 (W z2.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy
          with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z2.T)  # (z_dim, B)
        logits = torch.matmul(z1, Wz)  # (B, B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

    def compute_skl(self, z1_dist, z2_dist):
        kl_1_2 = torch.mean(
            torch.distributions.kl.kl_divergence(z1_dist, z2_dist)
        )
        kl_2_1 = torch.mean(
            torch.distributions.kl.kl_divergence(z2_dist, z1_dist)
        )
        skl = (kl_1_2 + kl_2_1) / 2.
        return skl

    def compute_multi_view_skl(self, obs1, obs2, z1, z2, actions, ema=False):
        z1_ = self.multi_view_encode(obs1, z2, actions)
        z2_ = self.multi_view_encode(obs2, z1, actions, ema=ema)
        z1_dist = get_dist(z1_)
        z2_dist = get_dist(z2_)

        kl_1_2 = torch.mean(
            torch.distributions.kl.kl_divergence(z1_dist, z2_dist)
        )
        kl_2_1 = torch.mean(
            torch.distributions.kl.kl_divergence(z2_dist, z1_dist)
        )
        skl = (kl_1_2 + kl_2_1) / 2.
        return skl

    def compute_kl_balancing(
        self, z1_prior, z1_post
    ):
        def get_dist(rssm_state: RSSMState, ema=False):
            if not ema:
                return td.independent.Independent(
                    td.Normal(rssm_state.mean, rssm_state.std), 1
                )
            else:
                return td.independent.Independent(
                    td.Normal(
                        rssm_state.mean.detach(), rssm_state.std.detach()
                    ), 1
                )
        kl_t = 0.8 * torch.mean(
            torch.distributions.kl.kl_divergence(
                get_dist(z1_post, ema=True), get_dist(z1_prior)
            )
        )
        kl_q = 0.2 * torch.mean(
            torch.distributions.kl.kl_divergence(
                get_dist(z1_post), get_dist(z1_prior, ema=True)
            )
        )
        return kl_t + kl_q



class AgentDRIBO(AgentSACBase):
    """DRIBO representation learning with SAC."""
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
        encoder_feature_dim: int = 50,
        encoder_tau: float = 0.005,
        num_layers: int = 4,
        num_filters: int = 32,
        stochastic_size: int = 30,
        deterministic_size: int = 200,
        mib_update_freq: int = 1,
        log_interval: int = 100,
        multi_view_skl: bool = False,
        mib_batch_size: int = 10,
        mib_seq_len: int = 50,
        beta_start_value: float = 1e-3,
        beta_end_value: int = 1,
        grad_clip: float = 500,
        kl_balancing: bool = False,
    ):
        super(AgentSACBase, self).__init__(obs_shape, action_shape, device, hidden_dim, discount, init_temperature, alpha_lr, alpha_beta, actor_lr, actor_beta, actor_log_std_min, actor_log_std_max, actor_update_freq, critic_lr, critic_beta, critic_tau, critic_target_update_freq, encoder_type, encoder_feature_dim, encoder_tau, num_layers, num_filters)

        self.mib_update_freq = mib_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.encoder_type = encoder_type
        self.multi_view_skl = multi_view_skl
        self.batch_size = mib_batch_size
        self.seq_len = mib_seq_len
        self.grad_clip = grad_clip
        self.kl_balancing = kl_balancing