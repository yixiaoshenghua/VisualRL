import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions


from utils.replay_buffer import MBReplayBuffer
from model.models import RSSM, ConvEncoder, ConvDecoder, DenseDecoder, ActionDecoder, MaskConvDecoder, EnsembleMaskConvDecoder
from utils import *


class AgentTIA:

    def __init__(self, args, obs_shape, action_size, device, restore=False):

        self.args = args
        if self.args.actor_grad == 'auto':
            self.args.actor_grad = 'dynamics' if self.args.agent == 'Dreamerv1' else 'reinforce'
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.device = device
        self.restore = args.restore
        self.restore_path = args.checkpoint_path
        self.data_buffer = MBReplayBuffer(self.args.buffer_size, self.obs_shape, self.action_size,
                                                    self.args.train_seq_len, self.args.batch_size)
        self.step = args.seed_steps
        self._build_model(restore=self.restore)

    def _build_model(self, restore):

        self.main_rssm = RSSM(
                    action_size =self.action_size,
                    stoch_size = self.args.stoch_size,
                    deter_size = self.args.deter_size,
                    hidden_size = self.args.hidden_size,
                    obs_embed_size = self.args.obs_embed_size,
                    activation = self.args.dense_activation_function,
                    discrete = self.args.discrete,
                    future=self.args.rssm_attention,
                    reverse=self.args.rssm_reverse,
                    device=self.device).to(self.device)
        self.disen_rssm = RSSM(
                    action_size =self.action_size,
                    stoch_size = self.args.stoch_size,
                    deter_size = self.args.deter_size,
                    hidden_size = self.args.hidden_size,
                    obs_embed_size = self.args.obs_embed_size,
                    activation = self.args.dense_activation_function,
                    discrete = self.args.discrete,
                    future=self.args.rssm_attention,
                    reverse=self.args.rssm_reverse,
                    device=self.device).to(self.device)
        self.actor = ActionDecoder(
                     action_size = self.action_size,
                     stoch_size = self.args.stoch_size,
                     deter_size = self.args.deter_size,
                     units = self.args.num_units,
                     n_layers = 4,
                     dist = self.args.actor_dist,
                     min_std = self.args.actor_min_std,
                     init_std  = self.args.actor_init_std,
                     activation = self.args.dense_activation_function,
                     discrete = self.args.discrete).to(self.device)
        self.main_obs_encoder  = ConvEncoder(
                            input_shape= self.obs_shape,
                            embed_size = self.args.obs_embed_size,
                            activation =self.args.cnn_activation_function).to(self.device)
        self.main_obs_decoder  = MaskConvDecoder(
                            stoch_size = self.args.stoch_size,
                            deter_size = self.args.deter_size,
                            output_shape = (6,*self.obs_shape[1:]),
                            activation = self.args.cnn_activation_function,
                            discrete=self.args.discrete).to(self.device)
        self.disen_obs_encoder = ConvEncoder(
                            input_shape= self.obs_shape,
                            embed_size = self.args.obs_embed_size,
                            activation =self.args.cnn_activation_function).to(self.device)
        self.disen_obs_decoder = MaskConvDecoder(
                            stoch_size = self.args.stoch_size,
                            deter_size = self.args.deter_size,
                            output_shape = (6,*self.obs_shape[1:]),
                            activation = self.args.cnn_activation_function,
                            discrete=self.args.discrete).to(self.device)
        self.disen_only_obs_decoder = ConvDecoder(
                            stoch_size = self.args.stoch_size,
                            deter_size = self.args.deter_size,
                            output_shape=self.obs_shape,
                            activation = self.args.cnn_activation_function,
                            discrete=self.args.discrete).to(self.device)
        self.joint_obs_decoder = EnsembleMaskConvDecoder(
                            decoder1 = self.main_obs_decoder,
                            decoder2 = self.disen_obs_decoder).to(self.device)
        self.main_reward_model = DenseDecoder(
                            stoch_size = self.args.stoch_size,
                            deter_size = self.args.deter_size,
                            output_shape = (1,),
                            n_layers = 2,
                            units=self.args.num_units,
                            activation= self.args.dense_activation_function,
                            dist = 'normal',
                            discrete = self.args.discrete).to(self.device)
        self.disen_reward_model = DenseDecoder(
                            stoch_size = self.args.stoch_size,
                            deter_size = self.args.deter_size,
                            output_shape = (1,),
                            n_layers = 2,
                            units=self.args.num_units,
                            activation= self.args.dense_activation_function,
                            dist = 'normal',
                            discrete = self.args.discrete).to(self.device)
        self.critic  = DenseDecoder(
                            stoch_size = self.args.stoch_size,
                            deter_size = self.args.deter_size,
                            output_shape = (1,),
                            n_layers = 3,
                            units = self.args.num_units,
                            activation= self.args.dense_activation_function,
                            dist = 'normal',
                            discrete = self.args.discrete).to(self.device) 
        if self.args.slow_target:
            self.target_critic = DenseDecoder(
                            stoch_size = self.args.stoch_size,
                            deter_size = self.args.deter_size,
                            output_shape = (1,),
                            n_layers = 3,
                            units = self.args.num_units,
                            activation= self.args.dense_activation_function,
                            dist = 'normal',
                            discrete = self.args.discrete).to(self.device) 
            self._updates = 0
        if self.args.use_disc_model:  
            self.discount_model = DenseDecoder(
                                stoch_size = self.args.stoch_size,
                                deter_size = self.args.deter_size,
                                output_shape = (1,),
                                n_layers = 2,
                                units=self.args.num_units,
                                activation= self.args.dense_activation_function,
                                dist = 'binary',
                                discrete = self.args.discrete).to(self.device)
        
        if self.args.use_disc_model:
            self.world_model_params = list(self.main_rssm.parameters()) + list(self.main_obs_encoder.parameters()) \
              + list(self.main_reward_model.parameters()) + list(self.discount_model.parameters())
        else:
            self.world_model_params = list(self.main_rssm.parameters()) + list(self.main_obs_encoder.parameters()) \
              + list(self.main_reward_model.parameters())
        self.disen_model_params = list(self.disen_rssm.parameters()) + list(self.disen_obs_encoder.parameters()) + list(self.disen_only_obs_decoder.parameters())
    
        self.world_model_opt = optim.Adam(self.world_model_params, self.args.model_learning_rate)
        self.critic_opt = optim.Adam(self.critic.parameters(), self.args.value_learning_rate)
        self.actor_opt = optim.Adam(self.actor.parameters(), self.args.actor_learning_rate)
        self.decoder_opt = optim.Adam(self.joint_obs_decoder.parameters(), self.args.model_learning_rate)
        self.disen_model_opt = optim.Adam(self.disen_model_params, self.args.model_learning_rate)
        self.disen_reward_opt = optim.Adam(self.disen_reward_model.parameters(), self.args.disen_reward_learning_rate)

        if self.args.use_disc_model:
            self.world_model_modules = [self.main_rssm, self.main_obs_encoder, self.main_reward_model, self.discount_model]
        else:
            self.world_model_modules = [self.main_rssm, self.main_obs_encoder, self.main_reward_model]
        self.value_modules = [self.critic]
        self.actor_modules = [self.actor]
        self.disen_modules = [self.disen_obs_encoder, self.disen_rssm, self.disen_only_obs_decoder]
        self.decoder_modules = [self.joint_obs_decoder]
        self.disen_reward_modules = [self.disen_reward_model]


        if restore:
            self.load(self.restore_path)

    def actor_loss(self):
        loss_dict, log_dict = {}, {}
        with torch.no_grad():
            posterior = self.main_rssm.detach_state(self.main_rssm.seq_to_batch(self.posterior))

        with FreezeParameters(self.world_model_modules):
            imag_states, imag_actions, imag_feats = self.imagine(posterior, self.args.imagine_horizon)

        self.imag_feat = self.main_rssm.get_feat(imag_states)

        with FreezeParameters(self.world_model_modules + self.value_modules):
            imag_rew_dist = self.main_reward_model(self.imag_feat)
            imag_val_dist = self.critic(self.imag_feat)

            imag_rews = imag_rew_dist.mean
            imag_vals = imag_val_dist.mean
            if self.args.use_disc_model:
                imag_disc_dist = self.discount_model(self.imag_feat)
                discounts = imag_disc_dist.mean().detach()
            else:
                discounts =  self.args.discount * torch.ones_like(imag_rews).detach()

        self.returns = compute_return(imag_rews[:-1], imag_vals[:-1],discounts[:-1] \
                                         ,self.args.td_lambda, imag_vals[-1])

        discounts = torch.cat([torch.ones_like(discounts[:1]), discounts[1:-1]], 0)
        self.discounts = torch.cumprod(discounts, 0).detach()
        actor_loss = -torch.mean(self.discounts * self.returns)
        
        loss_dict['actor_loss'] = actor_loss
        log_dict['train/actor_loss'] = actor_loss.item()
        
        return loss_dict, log_dict

    def critic_loss(self):
        loss_dict, log_dict = {}, {}
        with torch.no_grad():
            value_feat = self.imag_feat[:-1].detach()
            discount   = self.discounts.detach()
            value_targ = self.returns.detach()

        value_dist = self.critic(value_feat)  
        value_loss = -torch.mean(self.discounts * value_dist.log_prob(value_targ).unsqueeze(-1))

        loss_dict['value_loss'] = value_loss
        log_dict['train/value_loss'] = value_loss.item()

        return loss_dict, log_dict

    def world_model_loss(self, obs, acs, rews, nonterms):
        loss_dict, log_dict = {}, {}
        obs = preprocess_obs(obs)
        # main
        main_obs_embed = self.main_obs_encoder(obs[1:]) # (T-1, n, e)
        main_init_state = self.main_rssm.init_state(self.args.batch_size, self.device)
        prior, self.posterior = self.main_rssm.observe_rollout(main_obs_embed, acs[:-1], nonterms[:-1], main_init_state, self.args.train_seq_len-1)
        features = self.main_rssm.get_feat(self.posterior)

        # disen
        disen_obs_embed = self.disen_obs_encoder(obs[1:]) # (T-1, n, e)
        disen_init_state = self.disen_rssm.init_state(self.args.batch_size, self.device)
        disen_prior, self.disen_posterior = self.disen_rssm.observe_rollout(disen_obs_embed, acs[:-1], nonterms[:-1], disen_init_state, self.args.train_seq_len-1)
        disen_features = self.disen_rssm.get_feat(self.disen_posterior)

        # disen image pred
        image_pred_disen = self.disen_only_obs_decoder(disen_features)

        # joint image pred
        image_pred_joint, image_pred_joint_main, image_pred_joint_disen, mask_pred = self.joint_obs_decoder(features, disen_features)

        # main reward pred
        rew_dist = self.main_reward_model(features)

        # optimize disen reward predictor till optimal
        for _ in range(self.args.num_reward_opt_iters):
            disen_rew_dist = self.disen_reward_model(disen_features.detach())
            disen_rew_loss = -torch.mean(disen_rew_dist.log_prob(rews[:-1]))
            self.disen_reward_opt.zero_grad()
            disen_rew_loss.backward()
            self.disen_reward_opt.step()

        # disen reward pred with optimal reward predictor
        disen_rew_dist = self.disen_reward_model(disen_features)

        # main model loss
        if self.args.use_disc_model:
            disc_dist = self.discount_model(features)

        kl_loss = self.main_rssm.get_kl_loss(prior, self.posterior, self.args.kl_alpha, self.args.free_nats, self.args.agent=='TIAv2')

        obs_loss = -torch.mean(image_pred_joint.log_prob(obs[1:])) 
        rew_loss = -torch.mean(rew_dist.log_prob(rews[:-1]))
        if self.args.use_disc_model:
            disc_loss = -torch.mean(disc_dist.log_prob(nonterms[:-1]))
            loss_dict['world_model/disc_loss'] = disc_loss
            log_dict['world_model/disc_loss'] = disc_loss.item()

        if self.args.use_disc_model:
            model_loss = self.args.kl_loss_coeff * kl_loss + obs_loss + rew_loss + self.args.disc_loss_coeff * disc_loss
        else:
            model_loss = self.args.kl_loss_coeff * kl_loss + obs_loss + rew_loss 
        
        # disen model loss
        disen_kl_loss = self.disen_rssm.get_kl_loss(disen_prior, self.disen_posterior, self.args.kl_alpha, self.args.free_nats, self.args.agent=='TIAv2')

        disen_obs_loss = -torch.mean(image_pred_joint.log_prob(obs[1:])) 
        disen_only_obs_loss = -torch.mean(image_pred_disen.log_prob(obs[1:]))
        disen_rew_loss = -torch.mean(disen_rew_dist.log_prob(rews[:-1]))

        disen_model_loss = self.args.disen_kl_loss_coeff * disen_kl_loss + disen_obs_loss \
            + disen_only_obs_loss * self.args.disen_rec_loss_coeff - disen_rew_loss * self.args.disen_neg_rew_loss_coeff
        
        decode_loss = model_loss + disen_model_loss
        
        loss_dict['world_model/kl_loss'] = kl_loss
        loss_dict['world_model/obs_loss'] = obs_loss
        loss_dict['world_model/rew_loss'] = rew_loss
        loss_dict['world_model/model_loss'] = model_loss
        log_dict['world_model/kl_loss'] = kl_loss.item()
        log_dict['world_model/obs_loss'] = obs_loss.item()
        log_dict['world_model/rew_loss'] = rew_loss.item()
        log_dict['world_model/model_loss'] = model_loss.item()
        loss_dict['world_model/decode_loss'] = decode_loss
        log_dict['world_model/decode_loss'] = decode_loss.item()

        loss_dict['disen_model/kl_loss'] = disen_kl_loss
        loss_dict['disen_model/obs_loss'] = disen_obs_loss
        loss_dict['disen_model/only_obs_loss'] = disen_only_obs_loss
        loss_dict['disen_model/rew_loss'] = disen_rew_loss
        loss_dict['disen_model/model_loss'] = disen_model_loss
        log_dict['disen_model/kl_loss'] = disen_kl_loss.item()
        log_dict['disen_model/obs_loss'] = disen_obs_loss.item()
        log_dict['disen_model/only_obs_loss'] = disen_only_obs_loss.item()
        log_dict['disen_model/rew_loss'] = disen_rew_loss.item()
        log_dict['disen_model/model_loss'] = disen_model_loss.item()

        return loss_dict, log_dict

    def update_world_model(self, obs, acs, rews, nonterms):
        wm_loss_dict, wm_log_dict = self.world_model_loss(obs, acs, rews, nonterms)
        main_model_loss = wm_loss_dict['world_model/model_loss']
        disen_model_loss = wm_loss_dict['disen_model/model_loss']
        decode_loss = wm_loss_dict['world_model/decode_loss']
        self.world_model_opt.zero_grad()
        self.disen_model_opt.zero_grad()
        self.decoder_opt.zero_grad()
        main_model_loss.backward(retain_graph=True)
        disen_model_loss.backward(retain_graph=True)
        decode_loss.backward()
        nn.utils.clip_grad_norm_(self.world_model_params, self.args.grad_clip_norm)
        nn.utils.clip_grad_norm_(self.disen_model_params, self.args.grad_clip_norm)
        nn.utils.clip_grad_norm_(self.joint_obs_decoder.parameters(), self.args.grad_clip_norm)
        self.world_model_opt.step()
        self.disen_model_opt.step()
        self.decoder_opt.step()
        return wm_log_dict
    
    def update_actor(self):
        ac_loss_dict, ac_log_dict = self.actor_loss()
        actor_loss = ac_loss_dict['actor_loss']
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_clip_norm)
        self.actor_opt.step()
        return ac_log_dict
    
    def update_critic(self):
        val_loss_dict, val_log_dict = self.critic_loss()
        value_loss = val_loss_dict['value_loss']
        self.critic_opt.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.args.grad_clip_norm)
        self.critic_opt.step()
        return val_log_dict

    def target(self, feat, reward, disc):
        if self.args.slow_target:
            value = self.target_critic(feat).mean
        else:
            value = self.critic(feat).mean
        target = compute_return(reward[:-1], value[:-1], disc[:-1], \
                                self.args.td_lambda, value[-1])
        weight = torch.cumprod(torch.cat([torch.ones_like(disc[:1]), disc[1:-1]], 0).detach(), 0)
        return target, weight

    def update_slow_target(self):
        if self.args.slow_target:
            if self._updates % self.args.slow_target_update == 0:
                mix = 1.0 if self._updates == 0 else float(
                    self.args.slow_target_fraction)
                for s, d in zip(self.critic.parameters(), self.target_critic.parameters()):
                    d.data = mix * s.data + (1 - mix) * d.data
            self._updates += 1

    def update(self):
        log_dict = {}
        obs, acs, rews, terms = self.data_buffer.sample()
        obs  = torch.tensor(obs, dtype=torch.float32).to(self.device)
        acs  = torch.tensor(acs, dtype=torch.float32).to(self.device)
        rews = torch.tensor(rews, dtype=torch.float32).to(self.device).unsqueeze(-1)
        nonterms = torch.tensor((1.0-terms), dtype=torch.float32).to(self.device).unsqueeze(-1)

        wm_log_dict = self.update_world_model(obs, acs, rews, nonterms)
        ac_log_dict = self.update_actor()
        val_log_dict = self.update_critic()
        self.update_slow_target()

        log_dict.update(wm_log_dict)
        log_dict.update(ac_log_dict)
        log_dict.update(val_log_dict)

        return log_dict

    def act_with_world_model(self, obs, prev_state, prev_action, explore=False):

        obs = obs['image']
        obs  = torch.tensor(obs.copy(), dtype=torch.float32).to(self.device).unsqueeze(0)
        obs_embed = self.main_obs_encoder(preprocess_obs(obs))
        _, posterior = self.main_rssm.observe_step(prev_state, prev_action, obs_embed)
        features = self.main_rssm.get_feat(posterior)
        action = self.actor(features, deter=not explore) 
        if explore:
            action = self.actor.add_exploration(action, self.args.action_noise)

        return  posterior, action

    def act_and_collect_data(self, env, collect_steps):

        obs = env.reset()
        done = False
        prev_state = self.main_rssm.init_state(1, self.device)
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
                prev_state = self.main_rssm.init_state(1, self.device)
                prev_action = torch.zeros(1, self.action_size).to(self.device)
                if i!= collect_steps-1:
                    episode_rewards.append(0.0)
            else:
                obs = next_obs 
                prev_state = posterior
                prev_action = torch.tensor(action, dtype=torch.float32).to(self.device).unsqueeze(0)

        return np.array(episode_rewards)

    def imagine(self, prev_state, horizon):

        rssm_state = prev_state
        next_states = []
        features = []
        actions = []

        for t in range(horizon):
            feature = self.main_rssm.get_feat(rssm_state)
            action = self.actor(feature.detach())
            rssm_state = self.main_rssm.imagine_step(rssm_state, action)
            next_states.append(rssm_state)
            actions.append(action)
            features.append(feature)

        next_states = self.main_rssm.stack_states(next_states)
        features = torch.cat(features, dim=0)
        actions = torch.cat(actions, dim=0)

        return next_states, actions, features

    def evaluate(self, env, eval_episodes, render=False):

        episode_rew = np.zeros((eval_episodes))

        video_images = [[] for _ in range(eval_episodes)]

        for i in range(eval_episodes):
            obs = env.reset()
            done = False
            prev_state = self.main_rssm.init_state(1, self.device)
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

        # video prediction
        obs, acs, rews, terms = self.data_buffer.sample()
        obs  = torch.tensor(obs, dtype=torch.float32).to(self.device)
        acs  = torch.tensor(acs, dtype=torch.float32).to(self.device)
        nonterms = torch.tensor((1.0-terms), dtype=torch.float32).to(self.device).unsqueeze(-1)
        pred_videos = self.video_pred(obs, acs, nonterms)

        return episode_rew, np.array(video_images[:self.args.max_videos_to_save]), pred_videos # (T, H, W, C)

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
                if i!= seed_steps-1:
                    seed_episode_rews.append(0.0)
                done=False  
            else:
                obs = next_obs

        return np.array(seed_episode_rews)

    def save(self, save_path):

        torch.save(
            {'main_rssm' : self.main_rssm.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'main_reward_model': self.main_reward_model.state_dict(),
            'main_obs_encoder': self.main_obs_encoder.state_dict(),
            'disen_rssm': self.disen_rssm.state_dict(),
            'disen_obs_encoder': self.disen_obs_encoder.state_dict(),
            'disen_only_decoder':self.disen_only_obs_decoder.state_dict(),
            'disen_reward_model': self.disen_reward_model.state_dict(),
            'joint_obs_decoder': self.joint_obs_decoder.state_dict(),
            'discount_model': self.discount_model.state_dict() if self.args.use_disc_model else None,
            'actor_optimizer': self.actor_opt.state_dict(),
            'critic_optimizer': self.critic_opt.state_dict(),
            'world_model_optimizer': self.world_model_opt.state_dict(),
            'disen_model_optimizer': self.disen_model_opt.state_dict(),
            'disen_reward_optimizer': self.disen_reward_opt.state_dict(),
            'decoder_optimizer': self.decoder_opt.state_dict()}, save_path)

    def load(self, ckpt_path):

        checkpoint = torch.load(ckpt_path)
        self.main_rssm.load_state_dict(checkpoint['main_rssm'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.main_reward_model.load_state_dict(checkpoint['main_reward_model'])
        self.main_obs_encoder.load_state_dict(checkpoint['main_obs_encoder'])
        self.disen_rssm.load_state_dict(checkpoint['disen_rssm'])
        self.disen_reward_model.load_state_dict(checkpoint['disen_reward_model'])
        self.disen_obs_encoder.load_state_dict(checkpoint['disen_obs_encoder'])
        self.disen_only_obs_decoder.load_state_dict(checkpoint['disen_only_obs_decoder'])
        self.joint_obs_decoder.load_state_dict(checkpoint['joint_obs_decoder'])
        if self.args.use_disc_model and (checkpoint['discount_model'] is not None):
            self.discount_model.load_state_dict(checkpoint['discount_model'])

        self.world_model_opt.load_state_dict(checkpoint['world_model_optimizer'])
        self.actor_opt.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_opt.load_state_dict(checkpoint['critic_optimizer'])
        self.disen_model_opt.load_state_dict(checkpoint['disen_model_optimizer'])
        self.disen_reward_opt.load_state_dict(checkpoint['disen_reward_optimizer'])
        self.decoder_opt.load_state_dict(checkpoint['decoder_optimizer'])

    def set_step(self, step):
        self.step = step

    @torch.no_grad()
    def video_pred(self, obs, acs, nonterms):
        '''
        Log images reconstructions
        '''
        T = obs.shape[0]
        obs = preprocess_obs(obs)
        # main decoder
        main_obs_embed = self.main_obs_encoder(obs[1:])
        main_init_observe_state = self.main_rssm.init_state(4, self.device)
        _, main_posterior = self.main_rssm.observe_rollout(main_obs_embed[:5, :4], acs[:5, :4], nonterms[:5, :4], main_init_observe_state, 5) # (5, 4, ...)
        main_observe_features = self.main_rssm.get_feat(main_posterior)

        # disen decoder
        disen_obs_embed = self.disen_obs_encoder(obs[1:])
        disen_init_observe_state = self.disen_rssm.init_state(4, self.device)
        _, disen_posterior = self.disen_rssm.observe_rollout(disen_obs_embed[:5, :4], acs[:5, :4], nonterms[:5, :4], disen_init_observe_state, 5) # (5, 4, ...)
        disen_observe_features = self.disen_rssm.get_feat(disen_posterior)

        
        image_recon_joint, image_recon_joint_main, image_recon_joint_disen, mask_recon = self.joint_obs_decoder(main_observe_features, disen_observe_features)

        # joint log
        joint_recon = image_recon_joint.mean # (5, 4, 3, 64, 64)

        main_init_imagine_state = {k: v[-1, :] for k, v in main_posterior.items()} # get the last posterior and imagine
        disen_init_imagine_state = {k: v[-1, :] for k, v in disen_posterior.items()}
        main_prior = self.main_rssm.imagine_rollout(acs[5:, :4], nonterms[5:, :4], main_init_imagine_state, T-5) # (45, 4, ...)
        disen_prior = self.disen_rssm.imagine_rollout(acs[5:, :4], nonterms[5:, :4], disen_init_imagine_state, T-5) # (45, 4, ...)
        main_imagine_features = self.main_rssm.get_feat(main_prior)
        disen_imagine_features = self.disen_rssm.get_feat(disen_prior)
        image_pred_joint, _, _, mask_pred = self.joint_obs_decoder(main_imagine_features, disen_imagine_features) 
        joint_pred = image_pred_joint.mean # (45, 4, 3, 64, 64)

        # main log
        main_recon = image_recon_joint_main.mean # (5, 4, 3, 64, 64)
        main_pred, _ = self.main_obs_decoder(main_imagine_features)
        main_pred = main_pred.mean # (45, 4, 3, 64, 64)

        # disen log
        disen_recon = image_recon_joint_disen.mean # (5, 4, 3, 64, 64)
        disen_pred, _ = self.disen_obs_decoder(disen_imagine_features)
        disen_pred = disen_pred.mean # (45, 4, 3, 64, 64)

        # disen only log
        disen_only_recon = self.disen_only_obs_decoder(disen_observe_features).mean # (5, 4, 3, 64, 64)
        disen_only_pred = self.disen_only_obs_decoder(disen_imagine_features).mean # (5, 4, 3, 64, 64)

        # select 4 envs, do 5 frames from data, rest reconstruct from dataset
        # so if dataset has 50 frames, 5 initial are real, 50-5 are imagined

        mask_recon = mask_recon.cpu()
        mask_pred = mask_pred.cpu()
        joint_recon = joint_recon.cpu()
        joint_pred = joint_pred.cpu()
        main_recon = main_recon.cpu()
        main_pred = main_pred.cpu()
        disen_recon = disen_recon.cpu()
        disen_pred = disen_pred.cpu()
        disen_only_recon = disen_only_recon.cpu()
        disen_only_pred = disen_only_pred.cpu()
        truth = obs[:, :4].cpu() + 0.5 # (50, 4, 3, 64, 64)

        if len(joint_recon.shape)==3: #flat
            mask_recon = mask_recon.reshape(*mask_recon.shape[:-1],1,*self.shape[1:])
            mask_pred = mask_pred.reshape(*mask_pred.shape[:-1],1,*self.shape[1:])
            joint_recon = joint_recon.reshape(*joint_recon.shape[:-1],*self.shape)
            joint_pred = joint_pred.reshape(*joint_pred.shape[:-1],*self.shape)
            main_recon = main_recon.reshape(*main_recon.shape[:-1],*self.shape)
            main_pred = main_pred.reshape(*main_pred.shape[:-1],*self.shape)
            disen_recon = disen_recon.reshape(*disen_recon.shape[:-1],*self.shape)
            disen_pred = disen_pred.reshape(*disen_pred.shape[:-1],*self.shape)
            disen_only_recon = disen_only_recon.reshape(*disen_only_recon.shape,*self.shape)
            disen_only_pred = disen_only_pred.reshape(*disen_only_pred.shape,*self.shape)
            truth = truth.reshape(*truth.shape[:-1],*self.shape)


        mask = torch.cat([mask_recon[:5, :] + 0.5, mask_pred + 0.5], 0)
        mask = torch.cat([mask, mask, mask], dim=2)
        joint_model = torch.cat([joint_recon[:5, :] + 0.5, joint_pred + 0.5], 0)  # time
        joint_error = (joint_model - truth + 1) / 2
        joint_video = torch.cat([truth, joint_model, joint_error, mask], 3)  # on H
        main_model = torch.cat([main_recon[:5, :] + 0.5, main_pred + 0.5], 0)  # time
        main_error = (main_model - truth + 1) / 2
        main_video = torch.cat([truth, main_model, main_error], 3)
        disen_model = torch.cat([disen_recon[:5, :] + 0.5, disen_pred + 0.5], 0)  # time
        disen_error = (disen_model - truth + 1) / 2
        disen_video = torch.cat([truth, disen_model, disen_error], 3)  # on H
        disen_only_model = torch.cat([disen_only_recon[:5, :] + 0.5, disen_only_pred + 0.5], 0)  # time
        disen_only_error = (disen_only_model - truth + 1) / 2
        disen_only_video = torch.cat([truth, disen_only_model, disen_only_error], 3)  # on H
        # T, B, C, H, W = joint_video.shape  # time, batch, height, width, channels
        
        return {'joint':joint_video.permute(1, 0, 2, 3, 4), 'main':main_video.permute(1, 0, 2, 3, 4), 'disen':disen_video.permute(1, 0, 2, 3, 4), 'disen_only':disen_only_video.permute(1, 0, 2, 3, 4)}
    