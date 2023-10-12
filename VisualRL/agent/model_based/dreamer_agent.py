import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import tqdm
from torchvision import transforms
from PIL import Image
from utils.replay_buffer import make_replay_buffer
from model.models import RSSM, ConvEncoder, ConvDecoder, DenseDecoder, ActionDecoder
from .worldmodel import WorldModel
from utils.util import *


class AgentDreamer:

    def __init__(
            self, 
            obs_shape, action_shape, action_range, device, 
            agent, 
            actor_grad, actor_dist, actor_min_std, actor_init_std, 
            slow_target, slow_target_update, slow_target_fraction, 
            actor_lr, critic_lr, 
            imagine_horizon, discount, td_lambda, 
            action_noise, exploration_decay, exploration_min, 
            pre_transform_image_size, image_size, frame_stack, 
            buffer_size, batch_size, train_seq_length, init_steps, action_repeat, 
            stoch_size, deter_size, hidden_size, obs_embed_size, num_units, 
            cnn_activation_function, dense_activation_function, 
            discrete, 
            use_disc_model, disc_loss_coeff, reward_dist, discount_dist, 
            model_lr, adam_epsilon, adam_weight_decay, grad_clip_norm, 
            kl_alpha, kl_balance, free_nats, kl_loss_coeff, 
            policy_checkpoint_path, wm_checkpoint_path, max_videos_to_save, 
            restore=False
    ):
        # ------------global config------------
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.action_size = action_shape[0]
        self.action_range = action_range
        self.device = device
        # ------------agent config------------
        self.agent = agent
        self.actor_grad = actor_grad
        self.actor_dist = actor_dist
        self.actor_min_std = actor_min_std
        self.actor_init_std = actor_init_std
        # slow target
        self.slow_target = slow_target
        self.slow_target_update = slow_target_update
        self.slow_target_fraction = slow_target_fraction
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.imagine_horizon = imagine_horizon
        self.discount = discount
        self.td_lambda = td_lambda
        # action exploration
        self.action_noise = action_noise
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        # replay buffer
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.train_seq_length = train_seq_length
        self.action_repeat = action_repeat
        # world model
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.hidden_size = hidden_size
        self.obs_embed_size = obs_embed_size
        self.num_units = num_units
        self.cnn_activation_function = cnn_activation_function
        self.dense_activation_function = dense_activation_function
        self.discrete = discrete
        self.use_disc_model = use_disc_model
        self.disc_loss_coeff = disc_loss_coeff
        self.reward_dist = reward_dist
        self.discount_dist = discount_dist
        self.model_lr = model_lr
        self.adam_epsilon = adam_epsilon
        self.adam_weight_decay = adam_weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.kl_alpha = kl_alpha
        self.kl_balance = kl_balance
        self.free_nats = free_nats
        self.kl_loss_coeff = kl_loss_coeff
        # save and restore
        self.max_videos_to_save = max_videos_to_save
        self.restore = restore
        self.restore_policy_path = policy_checkpoint_path #FIXME: Add these to config
        self.restore_wm_path = wm_checkpoint_path #FIXME: Add these to config
        
        if self.actor_grad == 'auto':
            self.actor_grad = 'dynamics' if self.agent == 'dreamerv1' else 'reinforce'
        self.data_buffer = make_replay_buffer(
            self.action_shape, self.device, 
            agent, 
            pre_transform_image_size, image_size, frame_stack, 
            self.buffer_size, self.batch_size, self.train_seq_length
        )
        
        # if self.args.combine_offline_datadir is not None:
        #     self.data_buffer.add_from_files(self.args.combine_offline_datadir)
        self.step = init_steps
        self._build_model(restore=self.restore)
        self.train(training=True)
    
    def train(self, training: bool = True):
        self.training = training
        self.world_model.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def _build_model(self, restore):

        self.world_model = WorldModel(
            self.action_size, self.obs_shape, self.device, 
            self.stoch_size, self.deter_size, self.hidden_size, self.obs_embed_size, self.num_units, 
            self.cnn_activation_function, self.dense_activation_function, 
            self.discrete, 
            self.use_disc_model, self.disc_loss_coeff, self.reward_dist, self.discount_dist, 
            self.model_lr, self.adam_epsilon, self.adam_weight_decay, self.grad_clip_norm, 
            self.kl_alpha, self.kl_balance, self.free_nats, self.kl_loss_coeff,
            self.batch_size, self.train_seq_length, 
            render_image=False
        )

        self.actor = ActionDecoder(
            action_size = self.action_size,
            stoch_size = self.stoch_size,
            deter_size = self.deter_size,
            units = self.num_units,
            n_layers = 4,
            dist = self.actor_dist,
            min_std = self.actor_min_std,
            init_std  = self.actor_init_std,
            activation = self.dense_activation_function,
            discrete = self.discrete
        ).to(self.device)

        self.critic  = DenseDecoder(
            stoch_size = self.stoch_size,
            deter_size = self.deter_size,
            output_shape = (1,),
            n_layers = 3,
            units = self.num_units,
            activation= self.dense_activation_function,
            dist = 'normal',
            discrete = self.discrete
        ).to(self.device) 

        if self.slow_target:
            self.target_critic = DenseDecoder(
                stoch_size = self.stoch_size,
                deter_size = self.deter_size,
                output_shape = (1,),
                n_layers = 3,
                units = self.num_units,
                activation = self.dense_activation_function,
                dist = 'normal',
                discrete = self.discrete
            ).to(self.device)

            self._updates = 0

        # if self.args.use_disc_model:
        self.world_model_modules = self.world_model.world_model_modules#[self.world_model.rssm, self.world_model.obs_encoder, self.world_model.obs_decoder, self.world_model.reward_model, self.world_model.discount_model]
        # else:
        #     self.world_model_modules = [self.world_model.rssm, self.world_model.obs_encoder, self.world_model.obs_decoder, self.world_model.reward_model]

        self.critic_opt = optim.Adam(self.critic.parameters(), self.critic_lr, eps=self.adam_epsilon, weight_decay=self.adam_weight_decay)
        self.actor_opt = optim.Adam(self.actor.parameters(), self.actor_lr, eps=self.adam_epsilon, weight_decay=self.adam_weight_decay)

        self.value_modules = [self.critic]
        self.actor_modules = [self.actor]

        if restore:
            self.load(self.restore_policy_path)
            self.world_model.load(self.restore_wm_path)

    def actor_loss(self):
        loss_dict, log_dict = {}, {}
        with torch.no_grad():
            posterior = self.world_model.rssm.detach_state(self.world_model.rssm.seq_to_batch(self.world_model.posterior))

        with FreezeParameters(self.world_model_modules):
            self.imag_feat, imag_actions, imag_rews = self.imagine(posterior, self.imagine_horizon)

        # self.imag_feat = imag_feats

        with FreezeParameters(self.world_model_modules + self.value_modules):
            # imag_rew_dist = self.world_model.reward_model(self.imag_feat)
            imag_val_dist = self.critic(self.imag_feat)

            # imag_rews = imag_rew_dist.mean
            imag_vals = imag_val_dist.mean
            if self.use_disc_model:
                imag_disc_dist = self.world_model.discount_model(self.imag_feat)
                discounts = imag_disc_dist.mean().detach()
            else:
                discounts = self.discount * torch.ones_like(imag_rews).detach()

        self.returns = compute_return(imag_rews[:-1], imag_vals[:-1], discounts[:-1], \
                                         self.td_lambda, imag_vals[-1])

        discounts = torch.cat([torch.ones_like(discounts[:1]), discounts[1:-1]], 0)
        self.discounts = torch.cumprod(discounts, 0).detach()
        actor_loss = -torch.mean(self.discounts * self.returns)

        loss_dict['actor_loss'] = actor_loss
        log_dict['train/actor_loss'] = actor_loss.item()
        return loss_dict, log_dict

    def critic_loss(self):
        # States:         [z0]    [z1]    [z2]     z3
        # Rewards:        [r0]    [r1]    [r2]     r3
        # Values:         [v0]    [v1]    [v2]     v3
        # Weights:        [ 1]    [w1]    [w2]     w3
        # Targets:        [t0]    [t1]    [t2]
        # Loss:                l0        l1        l2
        loss_dict, log_dict = {}, {}
        with torch.no_grad():
            value_feat = self.imag_feat[:-1].detach()
            # discount   = self.discounts.detach()
            value_targ = self.returns.detach()

        value_dist = self.critic(value_feat)  
        value_loss = -torch.mean(self.discounts * value_dist.log_prob(value_targ).unsqueeze(-1))

        loss_dict['value_loss'] = value_loss
        log_dict['train/value_loss'] = value_loss.item()

        return loss_dict, log_dict

    def update_actor(self):
        ac_loss_dict, ac_log_dict = self.actor_loss()
        actor_loss = ac_loss_dict['actor_loss']
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        self.actor_opt.step()
        return ac_log_dict
    
    def update_critic(self):
        val_loss_dict, val_log_dict = self.critic_loss()
        value_loss = val_loss_dict['value_loss']
        self.critic_opt.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_opt.step()
        return val_log_dict

    def target(self, feat, reward, disc):
        if self.slow_target:
            value = self.target_critic(feat).mean
        else:
            value = self.critic(feat).mean
        target = compute_return(reward[:-1], value[:-1], disc[:-1], \
                                self.td_lambda, value[-1])
        weight = torch.cumprod(torch.cat([torch.ones_like(disc[:1]), disc[:-1]], 0), 0).detach()
        return target, weight

    def update_slow_target(self):
        if self.slow_target:
            if self._updates % self.slow_target_update == 0:
                mix = 1.0 if self._updates == 0 else float(
                    self.slow_target_fraction)
                for s, d in zip(self.critic.parameters(), self.target_critic.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1

    def update(self):
        log_dict = {}
        obs, acs, rews, nonterms = self.data_buffer.sample_dreamer()
        # obs, acs, rews, terms = self.data_buffer.sample()
        # obs  = torch.tensor(obs, dtype=torch.float32).to(self.device)
        # acs  = torch.tensor(acs, dtype=torch.float32).to(self.device)
        # rews = torch.tensor(rews, dtype=torch.float32).to(self.device).unsqueeze(-1)
        # nonterms = torch.tensor((1.0-terms), dtype=torch.float32).to(self.device).unsqueeze(-1)

        wm_log_dict = self.world_model.update_world_model(obs, acs, rews, nonterms)
        ac_log_dict = self.update_actor()
        val_log_dict = self.update_critic()
        self.update_slow_target()

        log_dict.update(wm_log_dict)
        log_dict.update(ac_log_dict)
        log_dict.update(val_log_dict)

        return log_dict

    def sample_action(self, obs, explore=True):
        features = self.world_model.set_obs(obs, return_feat=True)
        action = self.actor(features, deter=not explore) 

        action_noise = self.action_noise
        if self.exploration_decay:
            action_noise *= 0.5 ** (self.step/self.exploration_decay)
        if self.exploration_min:
            action_noise = max(self.exploration_min, action_noise)
        action = self.actor.add_exploration(action, action_noise)

        self.world_model.prev_action = action
        return action[0].cpu().numpy()
    
    def select_action(self, obs, explore=False):
        features = self.world_model.set_obs(obs, return_feat=True)
        action = self.actor(features, deter=not explore)

        self.world_model.prev_action = action
        return action[0].cpu().numpy()

    def act_and_collect_data(self, env, collect_steps, step, logger, log_interval=1000, num_updates=None):

        obs = env.reset()
        done = False
        rssm_state = self.world_model.reset()
        episode_rewards = [0.0]

        for i in range(collect_steps):
            self.step += self.action_repeat
            with torch.no_grad():
                action = self.sample_action(obs, explore=True)
            next_obs, rew, done, _ = env.step(action)
            self.data_buffer.add(obs, action, rew, next_obs, done)

            episode_rewards[-1] += rew

            if done:
                obs = env.reset()
                done = False
                rssm_state = self.world_model.reset()
                if i != collect_steps-1:
                    episode_rewards.append(0.0)
            else:
                obs = next_obs 

        return np.array(episode_rewards)

    def imagine(self, prev_state, horizon):

        feature = self.world_model.set_state(prev_state, return_feat=True)
        features = [feature]
        actions = [torch.zeros_like(self.actor(feature.detach()))]
        rewards = [self.world_model.reward_model(feature).mean]

        for t in range(horizon):
            action = self.actor(feature.detach())
            feature, rew, _, _ = self.world_model.step(action, return_feat=True)
            actions.append(action)
            features.append(feature)
            rewards.append(rew)

        features = torch.stack(features, dim=0)
        actions = torch.stack(actions, dim=0)
        rewards = torch.stack(rewards, dim=0)

        return features, actions, rewards

    def evaluate(self, env, eval_episodes, render=False):

        episode_rew = np.zeros((eval_episodes))

        video_images = [[] for _ in range(eval_episodes)]
        self.train(False)

        for i in tqdm.tqdm(range(eval_episodes), desc='evaluating'):
            obs = env.reset()
            done = False
            rssm_state = self.world_model.reset()

            while not done:
                with torch.no_grad():
                    action = self.select_action(obs)
                # action = action[0].cpu().numpy()
                next_obs, rew, done, _ = env.step(action)
                episode_rew[i] += rew

                if render:
                    video_images[i].append(obs['image'].transpose(1, 2, 0).copy())
                obs = next_obs

        # video prediction
        # obs, acs, rews, terms = self.data_buffer.sample()
        # obs  = torch.tensor(obs, dtype=torch.float32).to(self.device)
        # acs  = torch.tensor(acs, dtype=torch.float32).to(self.device)
        # nonterms = torch.tensor((1.0-terms), dtype=torch.float32).to(self.device).unsqueeze(-1)
        # pred_videos = self.video_pred(obs, acs, nonterms)
        
        self.train(True)
        return episode_rew, np.array(video_images[:self.max_videos_to_save])#, pred_videos # (T, H, W, C)

    def collect_random_episodes(self, env, seed_steps):

        obs = env.reset()
        done = False
        seed_episode_rews = [0.0]

        for i in range(seed_steps):
            action = env.action_space.sample()
            next_obs, rew, done, _ = env.step(action)
            
            self.data_buffer.add(obs, action, rew, next_obs, done)
            seed_episode_rews[-1] += rew
            if done:
                obs = env.reset()
                if i!= seed_steps-1:
                    seed_episode_rews.append(0.0)
                done=False  
            else:
                obs = next_obs

        return np.array(seed_episode_rews)

    def save(self, save_policy_path):

        torch.save(
            {'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_opt.state_dict(),
            'critic_optimizer': self.critic_opt.state_dict()}, save_policy_path)
        

    def load(self, policy_ckpt_path):

        checkpoint = torch.load(policy_ckpt_path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_opt.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_opt.load_state_dict(checkpoint['critic_optimizer'])

    def set_step(self, step):
        self.step = step

    @torch.no_grad()
    def video_pred(self, obs, acs, nonterms):
        '''
        Log images reconstructions
        '''
        T = obs.shape[0]
        true_obs = preprocess_obs(obs)
        obs = preprocess_obs(obs)
        obs_embed = self.world_model.obs_encoder(obs[1:]) # (T-1, n, e)
        
        init_state = self.world_model.rssm.init_state(4, self.device)
        _, states = self.world_model.rssm.observe_rollout(obs_embed[:5, :4], acs[:5, :4], nonterms[:5, :4], init_state, 5) # (5, 4, ...)
        recon = self.world_model.obs_decoder(self.world_model.rssm.get_feat(states)).mean # (5, 4, 3, 64, 64)

        init = {k: v[-1, :] for k, v in states.items()} # get the last posterior and imagine
        prior = self.world_model.rssm.imagine_rollout(acs[5:, :4], nonterms[5:, :4], init, T-5) # (45, 4, ...)
        features = self.world_model.rssm.get_feat(prior)
        openl = self.world_model.obs_decoder(features).mean # (45, 4, 3, 64, 64)
        
        # select 6 envs, do 5 frames from data, rest reconstruct from dataset
        # so if dataset has 50 frames, 5 initial are real, 50-5 are imagined

        recon = recon.cpu()
        openl = openl.cpu()
        truth = true_obs[:, :4].cpu() + 0.5 # (50, 4, 3, 64, 64)

        if len(recon.shape)==3: #flat
            recon = recon.reshape(*recon.shape[:-1],*self.shape)
            openl = openl.reshape(*openl.shape[:-1],*self.shape)
            truth = truth.reshape(*truth.shape[:-1],*self.shape)


        model = torch.cat([recon[:5, :] + 0.5, openl + 0.5], 0)  # time
        error = (model - truth + 1) / 2
        video = torch.cat([truth, model, error], 3)  # on H
        T, B, C, H, W = video.shape  # time, batch, height, width, channels
        return video.permute(1, 0, 2, 3, 4)# reshape(T, C, H, B * W).permute(0, 2, 3, 1).numpy()
    
    def save_data(self, env, collect_steps, save_path):
        rews = self.act_and_collect_data(env, collect_steps)
        print(rews)
        np.savez(save_path, 
                 image=self.data_buffer.observations[self.data_buffer.idx-collect_steps:self.data_buffer.idx],
                 action=self.data_buffer.actions[self.data_buffer.idx-collect_steps:self.data_buffer.idx],
                 reward=self.data_buffer.rewards[self.data_buffer.idx-collect_steps:self.data_buffer.idx],
                 done=self.data_buffer.terminals[self.data_buffer.idx-collect_steps:self.data_buffer.idx])
        print("Save data to {}".format(save_path))