import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed

import utils.util as util
from model.encoder import make_encoder
import utils.data_augs as rad 
from sac import Actor, Critic, weight_init

LOG_FREQ = 10000


class RadSacAgentDDP(object):
    """RAD with SAC, multi-GPU."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        num_layers=4,
        num_filters=32,
        log_interval=100,
        detach_encoder=False,
        latent_dim=128,
        data_augs='',
        rank=0,
        print_param_check=False,
        action_range=[-1,1],
        image_channel=3,
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.latent_dim = latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type
        self.data_augs = data_augs
        self.action_range = action_range
        self.image_channel = image_channel
        
        self.rank = rank
        self.world_size = torch.distributed.get_world_size()
        self._print_param_check = print_param_check
        device_ids = [self.device.index]  # list of length 1, my device

        self.augs_funcs = {}

        aug_to_func = {
                'crop':rad.random_crop,
                'translate':rad.random_translate,
                'window':rad.random_window,
                'grayscale':rad.random_grayscale,
                'cutout':rad.random_cutout,
                'cutout_color':rad.random_cutout_color,
                'flip':rad.random_flip,
                'rotate':rad.random_rotation,
                'rand_conv':rad.random_convolution,
                'color_jitter':rad.random_color_jitter,
                'no_aug':rad.no_aug,
            }

        for aug_name in self.data_augs.split('-'):
            assert aug_name in aug_to_func, 'invalid data aug string'
            self.augs_funcs[aug_name] = aug_to_func[aug_name]

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, image_channel
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, image_channel
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, image_channel
        ).to(device)


        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and CURL and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # Wrap models with DDP for training, before making optimizer.
        self.actor = DDP(self.actor, device_ids=device_ids, find_unused_parameters=True)  # Only this process's GPU.
        self.critic = DDP(self.critic, device_ids=device_ids, find_unused_parameters=True)
        # will do alpha manually.
        
        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            action = mu.cpu().data
            return action.numpy().flatten()

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = util.center_crop_image(obs, self.image_size)
 
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(
            obs, action, detach_encoder=self.detach_encoder)
        critic_loss = 0
        critic_loss = critic_loss + F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        if step % self.log_interval == 0 and self.rank == 0:
            L.log('train_critic/loss', critic_loss, step)


        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()  # DDP automatically all-reduces gradient.
        if 'pixel' in self.encoder_type:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1000000)
        self.critic_optimizer.step()

        if self.rank == 0:
            self.critic.module.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0 and self.rank == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * \
            (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0 and self.rank == 0:
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if 'pixel' in self.encoder_type:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1000000)
        self.actor_optimizer.step()

        if self.rank == 0:
            self.actor.module.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0 and self.rank == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        if self._print_param_check:
            print(f"RANK: {self.rank} after update, before all reduce logalpha: {self.log_alpha}")
        torch.distributed.all_reduce(self.log_alpha.data)
        self.log_alpha.data /= self.world_size  # Might need to turn off grad or smth.
        if self._print_param_check:
            print(f"RANK: {self.rank} after all reduce logalpha: {self.log_alpha}")

    def update(self, replay_buffer, L, step):
        if self._print_param_check:
            critic_Q1_param_check = [self.critic.Q1.parameters()][0].data.flatten()[:5]
            actor_trunk_param_check = [self.actor.trunk.parameters()][0].data.flatten()[:5]
            encoder_param_check = [self.critic.encoder.parameters()][0].data.flatten()[:5]
            print(f"RANK: {self.rank} before update some params:",
                f"\n\t critic Q1.params[0][:5]: {critic_Q1_param_check}",
                f"\n\t actor trunk.params[0][:5]: {actor_trunk_param_check}",
                f"\n\t encoder.params[0][:5]: {encoder_param_check}",
                "\n\n",
            )

        if 'pixel' in self.encoder_type:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_rad(self.augs_funcs)
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()
    
        if step % self.log_interval == 0 and self.rank == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            util.soft_update_params(
                self.critic.module.Q1, self.critic_target.Q1, self.critic_tau
            )
            util.soft_update_params(
                self.critic.module.Q2, self.critic_target.Q2, self.critic_tau
            )
            util.soft_update_params(
                self.critic.module.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

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
 
