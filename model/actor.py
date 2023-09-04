import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from model.encoder import make_encoder
from utils.pytorch_util import gaussian_logprob, squash, weight_init, drq_weight_init
from utils.util import SquashedNormal

LOG_FREQ = 10000

class Actor(nn.Module):
    """CNN actor network"""
    def __init__(
        self, args, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters, output_logits, builtin_encoder=True
    ):
        super().__init__()
        self.args = args
        self.drq = args.agent.lower() == "drq"
        self.builtin_encoder = builtin_encoder
        
        if self.builtin_encoder:
            self.encoder = make_encoder(
                encoder_type, obs_shape, encoder_feature_dim, num_layers,
                num_filters, output_logits=output_logits
            )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        if self.drq:
            self.trunk = nn.Sequential(
            nn.Linear(encoder_feature_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )
            self.apply(drq_weight_init)
        else:
            self.trunk = nn.Sequential(
                nn.Linear(encoder_feature_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, 2 * action_shape[0])
            )
            self.apply(weight_init)

        self.outputs = dict()

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False
    ):
        if self.builtin_encoder:
            obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            if not self.drq:
                noise = torch.randn_like(mu)
                pi = mu + noise * std
            else:
                dist = SquashedNormal(mu, std)
                return dist
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std