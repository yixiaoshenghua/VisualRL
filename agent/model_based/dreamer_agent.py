import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions

from utils.replay_buffer import MBReplayBuffer
from model.models import RSSM, ConvEncoder, ConvDecoder, DenseDecoder, ActionDecoder
from utils import FreezeParameters, compute_return, preprocess_obs


class AgentDreamer:

    def __init__(self, args, obs_shape, action_size, device, restore=False):

        self.args = args
        if self.args.actor_grad == 'auto':
            self.args.actor_grad = 'dynamics' if self.args.agent == 'Dreamerv1' else 'reinforce'
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.device = device
        self.restore = restore
        self.restore_path = args.restore_checkpoint_path
        self.data_buffer = MBReplayBuffer(self.args.buffer_size, self.obs_shape, self.action_size,
                                                    self.args.train_seq_len, self.args.batch_size)
        self.step = args.init_steps
        self._build_model(restore=self.restore)

        self.train()

    def _build_model(self, restore: bool):
        """Build model and optimizer
        
        Args:
            restore (bool): whether to restore from checkpoint
        """
        self.rssm = RSSM(
                    action_size = self.action_size,
                    stoch_size = self.args.stoch_size,
                    deter_size = self.args.deter_size,
                    hidden_size = self.args.hidden_size,
                    obs_embed_size = self.args.obs_embed_size,
                    activation = self.args.dense_activation_function,
                    discrete = self.args.discrete).to(self.device)
        self.actor = ActionDecoder(
                     action_size = self.action_size,
                     stoch_size = self.args.stoch_size,
                     deter_size = self.args.deter_size,
                     units = self.args.num_units,
                     n_layers = 4,
                     dist = self.args.actor_dist,
                     min_std = self.args.actor_min_std,
                     init_std = self.args.actor_init_std,
                     activation = self.args.dense_activation_function,
                     discrete = self.args.discrete).to(self.device)
        self.obs_encoder  = ConvEncoder(
                            input_shape = self.obs_shape,
                            embed_size = self.args.obs_embed_size,
                            activation = self.args.cnn_activation_function).to(self.device)
        self.obs_decoder  = ConvDecoder(
                            stoch_size = self.args.stoch_size,
                            deter_size = self.args.deter_size,
                            output_shape = self.obs_shape,
                            activation = self.args.cnn_activation_function,
                            discrete = self.args.discrete).to(self.device)
        self.reward_model = DenseDecoder(
                            stoch_size = self.args.stoch_size,
                            deter_size = self.args.deter_size,
                            output_shape = (1,),
                            n_layers = 2,
                            units = self.args.num_units,
                            activation = self.args.dense_activation_function,
                            dist = 'normal',
                            discrete = self.args.discrete).to(self.device)
        self.critic       = DenseDecoder(
                            stoch_size = self.args.stoch_size,
                            deter_size = self.args.deter_size,
                            output_shape = (1,),
                            n_layers = 3,
                            units = self.args.num_units,
                            activation = self.args.dense_activation_function,
                            dist = 'normal',
                            discrete = self.args.discrete).to(self.device) 
        if self.args.slow_target:
            self.target_critic = DenseDecoder(
                            stoch_size = self.args.stoch_size,
                            deter_size = self.args.deter_size,
                            output_shape = (1,),
                            n_layers = 3,
                            units = self.args.num_units,
                            activation = self.args.dense_activation_function,
                            dist = 'normal',
                            discrete = self.args.discrete).to(self.device) 
            self._updates = 0
        if self.args.use_disc_model:  
            self.discount_model = DenseDecoder(
                                stoch_size = self.args.stoch_size,
                                deter_size = self.args.deter_size,
                                output_shape = (1,),
                                n_layers = 2,
                                units = self.args.num_units,
                                activation = self.args.dense_activation_function,
                                dist = 'binary',
                                discrete = self.args.discrete).to(self.device)
        
        if self.args.use_disc_model:
            self.world_model_params = list(self.rssm.parameters()) + list(self.obs_encoder.parameters()) \
              + list(self.obs_decoder.parameters()) + list(self.reward_model.parameters()) + list(self.discount_model.parameters())
        else:
            self.world_model_params = list(self.rssm.parameters()) + list(self.obs_encoder.parameters()) \
              + list(self.obs_decoder.parameters()) + list(self.reward_model.parameters())
    
        self.world_model_opt = optim.Adam(self.world_model_params, self.args.model_learning_rate)
        self.critic_opt = optim.Adam(self.critic.parameters(), self.args.value_learning_rate)
        self.actor_opt = optim.Adam(self.actor.parameters(), self.args.actor_learning_rate)

        if self.args.use_disc_model:
            self.world_model_modules = [self.rssm, self.obs_encoder, self.obs_decoder, self.reward_model, self.discount_model]
        else:
            self.world_model_modules = [self.rssm, self.obs_encoder, self.obs_decoder, self.reward_model]
        self.value_modules = [self.critic]
        self.actor_modules = [self.actor]

        if restore:
            self.load_checkpoint(self.restore_path)

    def train(self, training: bool = True):
        self.training = training
        self.rssm.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.obs_encoder.train(training)
        self.obs_decoder.train(training)
        self.reward_model.train(training)
        if self.args.use_disc_model:
            self.discount_model.train(training)

    def reset(self):
        """Reset the agent state."""
        self.prev_state = self.rssm.init_state(1, self.device)
        self.prev_action = torch.zeros(1, self.action_size).to(self.device)

    @torch.no_grad()
    def select_action(self, obs, explore=False):
        obs = obs['image']
        obs = torch.tensor(obs.copy(), dtype=torch.float32).to(self.device).unsqueeze(0)
        obs_embed = self.obs_encoder(preprocess_obs(obs))
        _, posterior = self.rssm.observe_step(self.prev_state, self.prev_action, obs_embed)
        features = self.rssm.get_feat(posterior)
        action = self.actor(features, deter = not explore) 
        if explore:
            action = self.actor.add_exploration(action, self.args.action_noise)

        self.prev_state = posterior
        self.prev_action = action.clone().detach().to(dtype=torch.float32).to(self.device)
        
        return action.cpu().data.numpy().flatten()
    
    @torch.no_grad()
    def sample_action(self, obs, explore=True):
        obs = obs['image']
        obs = torch.tensor(obs.copy(), dtype=torch.float32).to(self.device).unsqueeze(0)
        obs_embed = self.obs_encoder(preprocess_obs(obs))
        _, posterior = self.rssm.observe_step(self.prev_state, self.prev_action, obs_embed)
        features = self.rssm.get_feat(posterior)
        action = self.actor(features, deter = not explore) 
        if explore:
            action = self.actor.add_exploration(action, self.args.action_noise)

        self.prev_state = posterior
        self.prev_action = action.clone().detach().to(dtype=torch.float32).to(self.device)
        # self.prev_action = torch.tensor(action, dtype=torch.float32).to(self.device)
        
        return action.cpu().data.numpy().flatten()

    def actor_loss(self):
        with torch.no_grad():
            posterior = self.rssm.detach_state(self.rssm.seq_to_batch(self.posterior))

        with FreezeParameters(self.world_model_modules):
            imag_states, imag_actions, imag_feats = self.imagine_rollout(posterior, self.args.imagine_horizon)

        self.imag_feat = self.rssm.get_feat(imag_states)

        with FreezeParameters(self.world_model_modules + self.value_modules):
            imag_rew_dist = self.reward_model(self.imag_feat)
            imag_val_dist = self.critic(self.imag_feat)

            imag_rews = imag_rew_dist.mean
            imag_vals = imag_val_dist.mean
            if self.args.use_disc_model:
                imag_disc_dist = self.discount_model(self.imag_feat)
                discounts = imag_disc_dist.mean().detach()
            else:
                discounts = self.args.discount * torch.ones_like(imag_rews).detach()

        self.returns = compute_return(imag_rews[:-1], imag_vals[:-1],discounts[:-1], \
                                         self.args.td_lambda, imag_vals[-1])

        discounts = torch.cat([torch.ones_like(discounts[:1]), discounts[1:-1]], 0)
        self.discounts = torch.cumprod(discounts, 0).detach()
        actor_loss = -torch.mean(self.discounts * self.returns)
        return actor_loss

    def target(self, feat, reward, disc):
        if self.args.slow_target:
            value = self.target_critic(feat).mean
        else:
            value = self.critic(feat).mean
        target = compute_return(reward[:-1], value[:-1], disc[:-1], \
                                self.args.td_lambda, value[-1])
        weight = torch.cumprod(torch.cat([torch.ones_like(disc[:1]), disc[1:-1]], 0).detach(), 0)
        return target, weight
    
    def imagine_rollout(self, prev_state, horizon):

        rssm_state = prev_state
        next_states = []
        features = []
        actions = []

        for t in range(horizon):
            feature = self.rssm.get_feat(rssm_state)
            action = self.actor(feature.detach())
            rssm_state = self.rssm.imagine_step(rssm_state, action)
            next_states.append(rssm_state)
            actions.append(action)
            features.append(feature)

        next_states = self.rssm.stack_states(next_states)
        features = torch.cat(features, dim=0)
        actions = torch.cat(actions, dim=0)

        return next_states, actions, features

    def _kl_loss(self, prior):
        prior_dist = self.rssm.get_dist(prior)
        post_dist = self.rssm.get_dist(self.posterior)

        if self.args.agent == 'Dreamerv2':
            post_no_grad = self.rssm.detach_state(self.posterior)
            prior_no_grad = self.rssm.detach_state(prior)
            
            kl_loss = self.args.kl_alpha * (torch.mean(distributions.kl.kl_divergence(
                               self.rssm.get_dist(post_no_grad), prior_dist)))
            kl_loss += (1-self.args.kl_alpha) * (torch.mean(distributions.kl.kl_divergence(
                               post_dist, self.rssm.get_dist(prior_no_grad))))
        else:
            kl_loss = torch.mean(distributions.kl.kl_divergence(post_dist, prior_dist))
            kl_loss = torch.max(kl_loss, kl_loss.new_full(kl_loss.size(), self.args.free_nats))
        return kl_loss

    def _reward_loss(self, rews, features):
        rew_dist = self.reward_model(features)
        rew_loss = -torch.mean(rew_dist.log_prob(rews[:-1]))
        return rew_loss

    def _obs_loss(self, obs, features):
        obs_dist = self.obs_decoder(features)
        obs_loss = -torch.mean(obs_dist.log_prob(obs[1:]))
        return obs_loss
    
    def _disc_loss(self, nonterms, features):
        disc_dist = self.discount_model(features)
        disc_loss = -torch.mean(disc_dist.log_prob(nonterms[:-1]))
        return disc_loss

    def world_model_loss(self, obs, acs, rews, nonterms):

        obs = preprocess_obs(obs)
        obs_embed = self.obs_encoder(obs[1:])
        init_state = self.rssm.init_state(self.args.batch_size, self.device)
        prior, self.posterior = self.rssm.observe_rollout(obs_embed, acs[:-1], nonterms[:-1], init_state, self.args.train_seq_len-1)
        features = self.rssm.get_feat(self.posterior)
        
        rew_loss = self._reward_loss(rews, features)
        obs_loss = self._obs_loss(obs, features)

        kl_loss = self._kl_loss(prior)

        if self.args.use_disc_model:
            disc_loss = self._disc_loss(nonterms, features)

        if self.args.use_disc_model:
            model_loss = self.args.kl_loss_coeff * kl_loss + obs_loss + rew_loss + self.args.disc_loss_coeff * disc_loss
        else:
            model_loss = self.args.kl_loss_coeff * kl_loss + obs_loss + rew_loss 
        
        return model_loss

    def update_world_model(self, obs, acs, rews, nonterms):
        model_loss = self.world_model_loss(obs, acs, rews, nonterms)
        self.world_model_opt.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.world_model_params, self.args.grad_clip_norm)
        self.world_model_opt.step()
        return model_loss
    
    def update_actor(self):
        actor_loss = self.actor_loss()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_clip_norm)
        self.actor_opt.step()
        return actor_loss
    
    def update_critic(self):
        with torch.no_grad():
            value_feat = self.imag_feat[:-1].detach()
            # discount   = self.discounts.detach() # TODO not use
            value_targ = self.returns.detach()

        value_dist = self.critic(value_feat)  
        value_loss = -torch.mean(self.discounts * value_dist.log_prob(value_targ).unsqueeze(-1))
        
        self.critic_opt.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.grad_clip_norm)
        self.critic_opt.step()
        return value_loss

    def update_slow_target(self):
        if self.args.slow_target:
            if self._updates % self.args.slow_target_update == 0:
                mix = 1.0 if self._updates == 0 else float(
                    self.args.slow_target_fraction)
                for s, d in zip(self.critic.parameters(), self.target_critic.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1

    def update(self):
        obs, acs, rews, terms = self.data_buffer.sample()
        obs  = torch.tensor(obs, dtype=torch.float32).to(self.device)
        acs  = torch.tensor(acs, dtype=torch.float32).to(self.device)
        rews = torch.tensor(rews, dtype=torch.float32).to(self.device).unsqueeze(-1)
        nonterms = torch.tensor((1.0-terms), dtype=torch.float32).to(self.device).unsqueeze(-1)

        model_loss = self.update_world_model(obs, acs, rews, nonterms)
        actor_loss = self.update_actor()
        value_loss = self.update_critic()
        self.update_slow_target()

        loss_dict = {}
        loss_dict['train/model_loss'] = model_loss.item()
        loss_dict['train/actor_loss'] = actor_loss.item()
        loss_dict['train/value_loss'] = value_loss.item()
        return loss_dict
        return model_loss.item(), actor_loss.item(), value_loss.item()

    def evaluate(self, env, eval_episodes, render=False):

        episode_rew = np.zeros((eval_episodes))

        video_images = [[] for _ in range(eval_episodes)]

        for i in range(eval_episodes):
            obs = env.reset()
            done = False
            prev_state = self.rssm.init_state(1, self.device)
            prev_action = torch.zeros(1, self.action_size).to(self.device)

            while not done:
                with torch.no_grad():
                    posterior, action = self.act_with_world_model(obs, prev_state, prev_action)
                action = action[0].cpu().numpy()
                next_obs, rew, done, _ = env.step(action)
                prev_state = posterior
                prev_action = torch.tensor(action, dtype=torch.float32).to(self.device).unsqueeze(0)

                episode_rew[i] += rew

                if render:
                    video_images[i].append(obs['image'].transpose(1, 2, 0).copy())
                obs = next_obs
        return episode_rew, np.array(video_images[:self.args.max_videos_to_save])

    def collect_random_episodes(self, env, seed_steps):

        obs = env.reset()
        done = False
        seed_episode_rews = [0.0]

        for i in range(seed_steps):
            action = env.action_space.sample()
            next_obs, rew, done, _ = env.step(action)
            
            self.data_buffer.add(obs, action, rew, done)
            seed_episode_rews[-1] += rew
            if done:
                obs = env.reset()
                if i != seed_steps - 1:
                    seed_episode_rews.append(0.0)
                done = False
            else:
                obs = next_obs

        return np.array(seed_episode_rews)

    def save_checkpoint(self, checkpoint_path):
        """Save the model parameters and optimizer to a file."""
        torch.save(
            {'rssm':          self.rssm.state_dict(),
            'actor':          self.actor.state_dict(),
            'reward_model':   self.reward_model.state_dict(),
            'obs_encoder':    self.obs_encoder.state_dict(),
            'obs_decoder':    self.obs_decoder.state_dict(),
            'discount_model': self.discount_model.state_dict() if self.args.use_disc_model else None,

            'world_model_optimizer': self.world_model_opt.state_dict(),
            'actor_optimizer':       self.actor_opt.state_dict(),
            'critic_optimizer':      self.critic_opt.state_dict(),
            }, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        """Load model parameters from a file."""
        checkpoint = torch.load(checkpoint_path)
        self.rssm.load_state_dict(checkpoint['rssm'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.reward_model.load_state_dict(checkpoint['reward_model'])
        self.obs_encoder.load_state_dict(checkpoint['obs_encoder'])
        self.obs_decoder.load_state_dict(checkpoint['obs_decoder'])
        if self.args.use_disc_model and (checkpoint['discount_model'] is not None):
            self.discount_model.load_state_dict(checkpoint['discount_model'])

        self.world_model_opt.load_state_dict(checkpoint['world_model_optimizer'])
        self.actor_opt.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_opt.load_state_dict(checkpoint['critic_optimizer'])

    def save_data_buffer(self, data_buffer_path):
        """Save the data buffer to a file."""
        self.data_buffer.save(data_buffer_path)

    def load_data_buffer(self, data_buffer_path):
        """Load the data buffer from a file."""
        self.data_buffer.load(data_buffer_path)

    def video_pred(self, obs, acs):
        '''
        Log images reconstructions
        '''
        obs = preprocess_obs(obs)
        obs_embed = self.obs_encoder(obs[1:])
        init_state = self.rssm.init_state(self.args.batch_size, self.device)
        states, _ = self.rssm.observe_rollout(obs_embed[:6, :5], acs[:6, :5], torch.ones((6, 5), device=self.device), init_state, self.args)
        recon = self.obs_decoder(self.rssm.get_feat(states)).mean[:6]
        init = {k: v[:, -1] for k, v in states.items()}
        prior = self.imagine_rollout(init, self.args.train_seq_len-5)
        features = self.rssm.get_feat(prior)
        openl = self.obs_decoder(features).mean
        
        # select 6 envs, do 5 frames from data, rest reconstruct from dataset
        # so if dataset has 50 frames, 5 initial are real, 50-5 are imagined

        recon = recon.cpu()
        openl = openl.cpu()
        truth = obs[:6].cpu() + 0.5

        if len(recon.shape) == 3: #flat
            recon = recon.reshape(*recon.shape[:-1],*self.shape)
            openl = openl.reshape(*openl.shape[:-1],*self.shape)
            truth = truth.reshape(*truth.shape[:-1],*self.shape)


        model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)  # time
        error = (model - truth + 1) / 2
        video = torch.cat([truth, model, error], 3)  # on H
        B, T, C, H, W = video.shape  # batch, time, height,width, channels
        return video.permute(1, 2, 3, 0, 4).reshape(T, C, H, B * W)
    

    '''
    # def actor_loss(self):

    #     with torch.no_grad():
    #         posterior = self.rssm.detach_state(self.rssm.seq_to_batch(self.posterior))

    #     with FreezeParameters(self.world_model_modules):
    #         # imag_states: (bs, 1:T+1, ...), imag_actions: (bs, 0:T, a_dim), imag_feats: (bs, 0:T, e_dim)
    #         imag_states, imag_actions, imag_feats = self.imagine_rollout(posterior, self.args.imagine_horizon)

    #     self.imag_feat = imag_feats # self.rssm.get_feat(imag_states)
    #     policy = self.actor(self.imag_feat.detach(), return_dist=True)
    #     action_entropy = policy.entropy().reshape((-1, 1))

    #     with FreezeParameters(self.world_model_modules + self.value_modules):
    #         imag_rew_dist = self.reward_model(self.imag_feat)

    #         imag_rews = imag_rew_dist.mean
    #         if self.args.use_disc_model:
    #             imag_disc_dist = self.discount_model(self.imag_feat)
    #             discounts = imag_disc_dist.mean().detach()
    #         else:
    #             discounts =  self.args.discount * torch.ones_like(imag_rews).detach()
    #         target, weight = self.target(self.imag_feat, imag_rews, discounts)
    #         self.discounts = weight
    #         self.returns = target

    #     if self.args.actor_grad == 'dynamics':
    #         objective = target
    #     elif self.args.actor_grad == 'reinforce':
    #         baseline = self.critic(self.imag_feat[:-1]).mean
    #         advantage = (target - baseline).detach()
    #         objective = policy.log_prob(imag_actions)[:-1].reshape((-1, 1)) * advantage
    #     elif self.args.actor_grad == 'both':
    #         baseline = self.critic(self.imag_feat[:-1]).mean
    #         advantage = (target - baseline).detach()
    #         objective = policy.log_prob(imag_actions)[:-1].reshape((-1, 1)) * advantage
    #         mix = schedule(self.args.actor_grad_mix, self.step)
    #         objective = mix * target + (1 - mix) * objective
    #     ent_scale = schedule(self.args.actor_ent, self.step)
    #     objective += ent_scale * action_entropy[:-1]
    #     actor_loss = -(weight * objective).mean()
    #     return actor_loss

    # def set_step(self, step):
    #     self.step = step

    
    def act_with_world_model(self, obs, prev_state, prev_action, explore=False):

        obs = obs['image']
        obs = torch.tensor(obs.copy(), dtype=torch.float32).to(self.device).unsqueeze(0)
        obs_embed = self.obs_encoder(preprocess_obs(obs))
        _, posterior = self.rssm.observe_step(prev_state, prev_action, obs_embed)
        features = self.rssm.get_feat(posterior)
        action = self.actor(features, deter = not explore) 
        if explore:
            action = self.actor.add_exploration(action, self.args.action_noise)

        return posterior, action

    def act_and_collect_data(self, env, collect_steps):

        obs = env.reset()
        done = False
        prev_state = self.rssm.init_state(1, self.device)
        prev_action = torch.zeros(1, self.action_size).to(self.device)

        episode_rewards = [0.0]

        for i in range(collect_steps):

            with torch.no_grad():
                posterior, action = self.act_with_world_model(obs, prev_state, prev_action, explore=True)
            action = action[0].cpu().numpy()
            next_obs, rew, done, _ = env.step(action)
            self.data_buffer.add(obs, action, rew, done)

            episode_rewards[-1] += rew

            if done:
                obs = env.reset()
                done = False
                prev_state = self.rssm.init_state(1, self.device)
                prev_action = torch.zeros(1, self.action_size).to(self.device)
                if i != collect_steps - 1:
                    episode_rewards.append(0.0)
            else:
                obs = next_obs 
                prev_state = posterior
                prev_action = torch.tensor(action, dtype=torch.float32).to(self.device).unsqueeze(0)

        return np.array(episode_rewards)
    '''