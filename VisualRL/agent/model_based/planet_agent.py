import torch
import torch.nn as nn
import numpy as np
import tqdm
from math import inf
from utils.replay_buffer import make_replay_buffer
from .worldmodel import WorldModel


class AgentPlaNet:
    def __init__(
            self, 
            obs_shape, action_shape, action_range, device, 
            agent, 
            planning_horizon, optimization_iters, candidates, top_candidates, 
            action_noise, 
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
        # planning
        self.planning_horizon = planning_horizon
        self.optimization_iters = optimization_iters
        self.candidates = candidates
        self.top_candidates = top_candidates
        # action exploration
        self.action_noise = action_noise
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

        self.data_buffer = make_replay_buffer(
            self.action_shape, self.device, 
            agent, 
            pre_transform_image_size, image_size, frame_stack, 
            self.buffer_size, self.batch_size, self.train_seq_length
        )

        self.step = init_steps
        self._build_model(restore=self.restore)
        self.train(training=True)

    def train(self, training: bool = True):
        self.training = training
        self.world_model.train(training)
    
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

        self.planner = MPCPlanner(
            self.action_size, self.planning_horizon, self.optimization_iters, 
            self.candidates, self.top_candidates, 
            self.world_model.rssm, self.world_model.reward_model, 
            self.action_range[0], self.action_range[1]
        )

        self.world_model_modules = self.world_model.world_model_modules

        if restore:
            self.load(self.restore_policy_path)
            self.world_model.load(self.restore_wm_path)
    
    def update(self):
        log_dict = {}
        # sample data from buffer
        obs, acs, rews, nonterms = self.data_buffer.sample_dreamer()

        # update world model
        wm_log_dict = self.world_model.update_world_model(obs, acs, rews, nonterms)

        log_dict.update(wm_log_dict)
        return log_dict
    
    def sample_action(self, obs, min_action=-inf, max_action=inf):
        posterior_state = self.world_model.set_obs(obs, return_feat=False)
        
        action = self.planner(posterior_state.copy())  # Get action from planner(q(s_t|o≤t,a<t), p)
        action = action + self.action_noise * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
        action.clamp_(min=min_action, max=max_action)  # Clip action range
        
        return action[0].cpu().numpy()
    
    def select_action(self, obs, min_action=-inf, max_action=inf):
        posterior_state = self.world_model.set_obs(obs, return_feat=False)
        
        action = self.planner(posterior_state.copy())  # Get action from planner(q(s_t|o≤t,a<t), p)
        action.clamp_(min=min_action, max=max_action)  # Clip action range
        
        return action[0].cpu().numpy()
    
    def act_and_collect_data(self, env, collect_steps, step, logger, log_interval=1000, num_updates=None):
        obs = env.reset()
        done = False
        rssm_state = self.world_model.reset()
        episode_rewards = [0.0]

        for i in range(collect_steps):
            self.step += self.action_repeat
            with torch.no_grad():
                action = self.sample_action(obs, self.action_range[0], self.action_range[1])
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
                    action = self.select_action(obs, self.action_range[0], self.action_range[1])
                next_obs, rew, done, _ = env.step(action)
                episode_rew[i] += rew

                if render:
                    video_images[i].append(obs['image'].transpose(1, 2, 0).copy())
                obs = next_obs
        
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


class MPCPlanner(nn.Module):
    __constants__ = ['action_size', 'planning_horizon', 'optimisation_iters', 'candidates', 'top_candidates', 'min_action', 'max_action']

    def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates, transition_model, reward_model, min_action=-inf, max_action=inf):
        super().__init__()
        self.transition_model, self.reward_model = transition_model, reward_model
        self.action_size, self.min_action, self.max_action = action_size, min_action, max_action
        self.planning_horizon = planning_horizon
        self.optimisation_iters = optimisation_iters
        self.candidates, self.top_candidates = candidates, top_candidates

    def forward(self, post_state):
        deter, stoch = post_state['deter'], post_state['stoch']
        B, H, Z = deter.size(0), deter.size(1), stoch.size(1)
        deter, stoch = deter.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H), stoch.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)
        post_state['deter'] = deter
        post_state['stoch'] = stoch
        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        action_mean, action_std_dev = torch.zeros(self.planning_horizon, B, 1, self.action_size, device=deter.device), torch.ones(self.planning_horizon, B, 1, self.action_size, device=deter.device)
        for _ in range(self.optimisation_iters):
            # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
            actions = (action_mean + action_std_dev * torch.randn(self.planning_horizon, B, self.candidates, self.action_size, device=action_mean.device)).view(self.planning_horizon, B * self.candidates, self.action_size)  # Sample actions (time x (batch x candidates) x actions)
            actions.clamp_(min=self.min_action, max=self.max_action)  # Clip action range
            # Sample next states
            prior = self.transition_model.imagine_rollout(actions, [1.0]*self.planning_horizon, post_state, self.planning_horizon)
            prior['stoch'] = prior['stoch'].view(-1, Z)
            prior['deter'] = prior['deter'].view(-1, H)
            prior = self.transition_model.get_feat(prior)
            # Calculate expected returns (technically sum of rewards over planning horizon)
            returns = self.reward_model(prior).view(self.planning_horizon, -1).sum(dim=0)
            # Re-fit belief to the K best action sequences
            _, topk = returns.reshape(B, self.candidates).topk(self.top_candidates, dim=1, largest=True, sorted=False)
            topk += self.candidates * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)  # Fix indices for unrolled actions
            best_actions = actions[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates, self.action_size)
            # Update belief with new means and standard deviations
            action_mean, action_std_dev = best_actions.mean(dim=2, keepdim=True), best_actions.std(dim=2, unbiased=False, keepdim=True)
        # Return first action mean µ_t
        return action_mean[0].squeeze(dim=1)