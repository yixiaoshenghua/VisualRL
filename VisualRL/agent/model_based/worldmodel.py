import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
from torchvision import transforms
from PIL import Image
from utils.replay_buffer import ReplayBuffer
from model.models import RSSM, ConvEncoder, ConvDecoder, DenseDecoder, ActionDecoder
from utils import *


class WorldModel: 
    ''' Simulated and learnable environment'''
    def __init__(
            self, 
            action_size, obs_shape, device, 
            stoch_size, deter_size, hidden_size, obs_embed_size, num_units, 
            cnn_activation_function, dense_activation_function, 
            discrete, 
            use_disc_model, disc_loss_coeff, reward_dist, disc_dist, 
            model_lr, adam_epsilon, adam_weight_decay, grad_clip_norm, 
            kl_alpha, kl_balance, free_nats, kl_loss_coeff,
            batch_size, train_seq_length, 
            render_image=False
    ):
        self.action_size = action_size
        self.obs_shape = obs_shape
        self.device = device
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
        self.disc_dist = disc_dist
        self.model_lr = model_lr
        self.adam_epsilon = adam_epsilon
        self.adam_weight_decay = adam_weight_decay
        self.grad_clip_norm = grad_clip_norm
        self.kl_alpha = kl_alpha
        self.kl_balance = kl_balance
        self.free_nats = free_nats
        self.kl_loss_coeff = kl_loss_coeff
        self.batch_size = batch_size
        self.train_seq_length = train_seq_length
        self.render_image = render_image

        self.rssm = RSSM(
                    action_size = self.action_size,
                    stoch_size = self.stoch_size,
                    deter_size = self.deter_size,
                    hidden_size = self.hidden_size,
                    obs_embed_size = self.obs_embed_size,
                    activation = self.dense_activation_function,
                    discrete = self.discrete).to(self.device)
        self.obs_encoder  = ConvEncoder(
                                input_shape= self.obs_shape,
                                embed_size = self.obs_embed_size,
                                activation =self.cnn_activation_function).to(self.device)
        self.obs_decoder  = ConvDecoder(
                            stoch_size = self.stoch_size,
                            deter_size = self.deter_size,
                            output_shape=self.obs_shape,
                            activation = self.cnn_activation_function,
                            discrete=self.discrete).to(self.device)
        self.reward_model = DenseDecoder(
                            stoch_size = self.stoch_size,
                            deter_size = self.deter_size,
                            output_shape = (1,),
                            n_layers = 2,
                            units=self.num_units,
                            activation= self.dense_activation_function,
                            dist = self.reward_dist,
                            discrete = self.discrete).to(self.device)
        if self.use_disc_model:  
            self.discount_model = DenseDecoder(
                                stoch_size = self.stoch_size,
                                deter_size = self.deter_size,
                                output_shape = (1,),
                                n_layers = 2,
                                units=self.num_units,
                                activation= self.dense_activation_function,
                                dist = self.disc_dist,
                                discrete = self.discrete).to(self.device)
        
        if self.use_disc_model:
            self.world_model_params = list(self.rssm.parameters()) + list(self.obs_encoder.parameters()) \
              + list(self.obs_decoder.parameters()) + list(self.reward_model.parameters()) + list(self.discount_model.parameters())
        else:
            self.world_model_params = list(self.rssm.parameters()) + list(self.obs_encoder.parameters()) \
              + list(self.obs_decoder.parameters()) + list(self.reward_model.parameters())
        self.world_model_opt = optim.Adam(self.world_model_params, self.model_lr, eps=self.adam_epsilon, weight_decay=self.adam_weight_decay)

        if self.use_disc_model:
            self.world_model_modules = [self.rssm, self.obs_encoder, self.obs_decoder, self.reward_model, self.discount_model]
        else:
            self.world_model_modules = [self.rssm, self.obs_encoder, self.obs_decoder, self.reward_model]
        
    def update_world_model(self, obs, acs, rews, nonterms):
        wm_loss_dict, wm_log_dict = self.world_model_loss(obs, acs, rews, nonterms)
        model_loss = wm_loss_dict['model_loss']
        self.world_model_opt.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(self.world_model_params, self.grad_clip_norm)
        self.world_model_opt.step()
        return wm_log_dict
    
    def world_model_loss(self, obs, acs, rews, nonterms):
        loss_dict, log_dict = {}, {}
        
        obs = preprocess_obs(obs)
        obs_embed = self.obs_encoder(obs) # (T, n, e)

        init_state = self.rssm.init_state(self.batch_size, self.device)
        prior, self.posterior = self.rssm.observe_rollout(obs_embed, acs, nonterms, init_state, self.train_seq_length)
        features = self.rssm.get_feat(self.posterior)
        
        rew_dist = self.reward_model(features)
        obs_dist = self.obs_decoder(features)
        if self.use_disc_model:
            disc_dist = self.discount_model(features)

        prior_dist = self.rssm.get_dist(prior)
        prior_ent = torch.mean(prior_dist.entropy())
        post_dist = self.rssm.get_dist(self.posterior)
        post_ent = torch.mean(post_dist.entropy())
        
        div_loss = self.rssm.get_div_loss(prior, self.posterior, self.kl_alpha, self.free_nats, self.kl_balance)#, div_type=self.args.div_type)

        obs_loss = -torch.mean(obs_dist.log_prob(obs)) 
        rew_loss = F.mse_loss(rew_dist, rews, reduction='none').mean() if self.reward_dist == 'none' else -torch.mean(rew_dist.log_prob(rews))
        model_loss = self.kl_loss_coeff * div_loss + obs_loss + rew_loss
        if self.use_disc_model:
            disc_loss = -torch.mean(disc_dist.log_prob(nonterms))
            model_loss += self.disc_loss_coeff * disc_loss
            loss_dict['disc_loss'] = disc_loss
            log_dict['disc_loss'] = disc_loss.item()
        
        loss_dict['div_loss'] = div_loss
        loss_dict['obs_loss'] = obs_loss
        loss_dict['rew_loss'] = rew_loss
        loss_dict['model_loss'] = model_loss
        log_dict['world_model/div_loss'] = div_loss.item()
        log_dict['world_model/obs_loss'] = obs_loss.item()
        log_dict['world_model/rew_loss'] = rew_loss.item()
        log_dict['world_model/model_loss'] = model_loss.item()
        log_dict['world_model/prior_ent'] = prior_ent.item()
        log_dict['world_model/post_ent'] = post_ent.item()
        log_dict['world_model/mi_gain'] = prior_ent.item() - post_ent.item()

        return loss_dict, log_dict

    def train(self, training):
        for module in self.world_model_modules:
            if training:
                module.train()
            else:
                module.eval()
    
    def save(self, save_path):
        torch.save(
            {'rssm' : self.rssm.state_dict(),
            'reward_model': self.reward_model.state_dict(),
            'obs_encoder': self.obs_encoder.state_dict(),
            'obs_decoder': self.obs_decoder.state_dict(),
            'discount_model': self.discount_model.state_dict() if self.use_disc_model else None,
            'world_model_optimizer': self.world_model_opt.state_dict(),}, save_path)

    def load(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        self.rssm.load_state_dict(checkpoint['rssm'])
        self.reward_model.load_state_dict(checkpoint['reward_model'])
        self.obs_encoder.load_state_dict(checkpoint['obs_encoder'])
        self.obs_decoder.load_state_dict(checkpoint['obs_decoder'])
        if self.use_disc_model and (checkpoint['discount_model'] is not None):
            self.discount_model.load_state_dict(checkpoint['discount_model'])

        self.world_model_opt.load_state_dict(checkpoint['world_model_optimizer'])
    
    
    def step(self, action, return_feat=False):
        if len(action.shape) == 1:
            action = action.reshape((-1, self.action_size))
        if type(action) == np.ndarray:
            action = torch.FloatTensor(action, device=self.device)
        self.rssm_state = self.rssm.imagine_step(self.rssm_state, action)
        self.prev_action = action
        obs = self.rssm_state if not self.render_image else self.render()
        reward = self.reward_model(self.rssm.get_feat(self.rssm_state)).mean
        done = False
        info = {}
        obs = self.rssm.get_feat(obs) if return_feat else obs
        return obs, reward, done, info
    
    
    def reset(self):
        self.obs = None
        self.rssm_state = self.rssm.init_state(1, self.device)
        self.prev_action = torch.zeros(1, self.action_size).to(self.device)
        return self.rssm_state
    
    
    def set_obs(self, obs, return_feat=False):
        self.obs = obs
        obs = obs['image']
        obs  = torch.tensor(obs.copy(), dtype=torch.float32).to(self.device).unsqueeze(0)
        obs = preprocess_obs(obs)
        obs_embed = self.obs_encoder(obs)
        _, self.rssm_state = self.rssm.observe_step(self.rssm_state, self.prev_action, obs_embed)
        if return_feat:
            return self.rssm.get_feat(self.rssm_state)
        else:
            return self.rssm_state
    
    
    def set_state(self, state, return_feat=False):
        self.rssm_state = state
        if return_feat:
            return self.rssm.get_feat(self.rssm_state)
        else:
            return self.rssm_state
    
    
    def render(self):
        obs = self.obs_decoder(self.rssm.get_feat(self.rssm_state)).mean
        return obs