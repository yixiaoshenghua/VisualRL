import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from model.encoder import make_encoder
from utils.pytorch_util import weight_init

LOG_FREQ = 10000

class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters, output_logits, builtin_encoder=True
    ):
        super().__init__()
        self.builtin_encoder = builtin_encoder

        if self.builtin_encoder:
            self.encoder = make_encoder(
                encoder_type, obs_shape, encoder_feature_dim, num_layers,
                num_filters, output_logits=output_logits
            )

        self.Q1 = QFunction(
            encoder_feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            encoder_feature_dim, action_shape[0], hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        if self.builtin_encoder:
            obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)