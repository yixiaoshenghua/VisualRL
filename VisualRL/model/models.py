import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from torch.distributions import constraints
from torch.distributions.transformed_distribution import TransformedDistribution

_str_to_activation = {
    'relu': nn.ReLU(),
    'elu' : nn.ELU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


class RSSM(nn.Module):

    def __init__(self, action_size, stoch_size, deter_size, hidden_size, obs_embed_size, activation, discrete=False):

        super().__init__()

        self.action_size = action_size
        self.stoch_size  = stoch_size   
        self.deter_size  = deter_size   # GRU hidden units
        self.hidden_size = hidden_size  # intermediate fc_layers hidden units 
        self.embedding_size = obs_embed_size
        self.discrete = discrete

        self.act_fn = _str_to_activation[activation]
        self.rnn = nn.GRUCell(self.deter_size, self.deter_size)

        self.fc_state_action = nn.Linear(self.stoch_size + self.action_size, self.deter_size) if not self.discrete \
                            else nn.Linear(self.stoch_size*self.discrete + self.action_size, self.deter_size)
        self.fc_embed_prior = nn.Linear(self.deter_size, self.hidden_size)
        self.fc_state_prior  = nn.Linear(self.hidden_size, 2*self.stoch_size) if not self.discrete \
                            else nn.Linear(self.hidden_size, self.stoch_size*self.discrete)
        self.fc_embed_posterior = nn.Linear(self.embedding_size + self.deter_size, self.hidden_size)
        self.fc_state_posterior = nn.Linear(self.hidden_size, 2*self.stoch_size) if not self.discrete \
                            else nn.Linear(self.hidden_size, self.stoch_size*self.discrete)

    def init_state(self, batch_size, device):
        if self.discrete:
            return dict(
                logit = torch.zeros(batch_size, self.stoch_size, self.discrete).to(device),
                stoch = torch.zeros(batch_size, self.stoch_size, self.discrete).to(device),
                deter = torch.zeros(batch_size, self.deter_size).to(device))
        else:   
            return dict(
                mean = torch.zeros(batch_size, self.stoch_size).to(device),
                std  = torch.zeros(batch_size, self.stoch_size).to(device),
                stoch = torch.zeros(batch_size, self.stoch_size).to(device),
                deter = torch.zeros(batch_size, self.deter_size).to(device))

    def get_feat(self, state):
        stoch = state['stoch']
        if self.discrete:
            stoch = stoch.reshape(*stoch.shape[:-2], self.stoch_size * self.discrete)
        return torch.cat([stoch, state['deter']], -1)

    def get_dist(self, state):
        if self.discrete:
            logit = state['logit'].float()
            distribution = OneHotDist(logit) # Quantization
        else:
            mean, std = state['mean'], state['std']
            distribution = distributions.Normal(mean, std)
        distribution = Independent(distribution, 1)
        return distribution

    def observe_step(self, prev_state, prev_action, obs_embed, nonterm=1.0):

        prior = self.imagine_step(prev_state, prev_action, nonterm) # h_t, \hat{z}_t <- p_{phi}(\hat{z}_t | f_{phi}([h_{t-1}, z_{t-1}], a_{t-1}))
        posterior_embed = self.act_fn(self.fc_embed_posterior(torch.cat([obs_embed, prior['deter']], dim=-1))) # z_t ~ q_{phi}(z_t | h_t, x_t)
        posterior = self.fc_state_posterior(posterior_embed) 
        stats = self.suff_stats_layer(posterior)
        stoch = self.get_dist(stats).rsample()

        posterior = {'stoch': stoch, 'deter': prior['deter'], **stats}
        return prior, posterior

    def imagine_step(self, prev_state, prev_action, nonterm=1.0):
        prev_stoch = prev_state['stoch'] # (B, s)
        if self.discrete:
            prev_stoch = prev_stoch.reshape(*prev_stoch.shape[:-2], self.stoch_size*self.discrete)
        state_action = self.act_fn(self.fc_state_action(torch.cat([prev_stoch*nonterm, prev_action], dim=-1))) # za_{t-1}=[z_{t-1}, a_{t-1}]
        deter = self.rnn(state_action, prev_state['deter']*nonterm) # (B, h) h_t = f_{phi}(h_{t-1}, za_{t-1})
        prior_embed = self.act_fn(self.fc_embed_prior(deter))
        prior = self.fc_state_prior(prior_embed)
        stats = self.suff_stats_layer(prior)
        stoch = self.get_dist(stats).rsample()
        prior = {'stoch': stoch, 'deter': deter, **stats}
        return prior

    def suff_stats_layer(self, x):
        '''
        x: the output logit by the prior or posterior network
        NOTE: we can add some complex distribution transformations here like normalizing flows.
        '''
        if self.discrete:
            logit = x.reshape(*x.shape[:-1], self.stoch_size, self.discrete)
            return {'logit': logit}
        else:
            mean, std = x.chunk(2, -1)
            std = F.softplus(std) + 0.1
            return {'mean': mean, 'std': std}
    
    # def suff_stats_ensemble(self, x):
    #     '''Plan2Explore'''
    #     stats = []
    #     for i in range(1):
    #         prior_embed = self.act_fn(self.fc_embed_prior[i](x))
    #         prior = self.fc_state_prior[i](prior_embed)
    #         stat = self.suff_stats_layer(prior) # -> mean, std
    #         stats.append(stat)
    #     stats = {k: torch.stack([stat[k] for stat in stats]) for k in stats[0].keys()}
    #     return stats # (ensemble, B, ...)

    def observe_rollout(self, obs_embed, actions, nonterms, prev_state, horizon):
        
        priors = []
        posteriors = []

        for t in range(horizon):
            prev_action = actions[t] * nonterms[t]
            prior_state, posterior_state = self.observe_step(prev_state, prev_action, obs_embed[t], nonterms[t])
            priors.append(prior_state)
            posteriors.append(posterior_state)
            prev_state = posterior_state

        priors = self.stack_states(priors, dim=0)
        posteriors = self.stack_states(posteriors, dim=0)

        return priors, posteriors

    def imagine_rollout(self, actions, nonterms, prev_state, horizon):
        priors = []

        for t in range(horizon):
            prev_action = actions[t] * nonterms[t]
            prior_state = self.imagine_step(prev_state, prev_action, nonterms[t]) # (B, ...)
            priors.append(prior_state)
            prev_state = prior_state

        priors = self.stack_states(priors, dim=0)

        return priors

    def get_div_loss(self, prior, posterior, kl_alpha, free_nats, kl_balancing):
        divergence = distributions.kl.kl_divergence
        prior_dist = self.get_dist(prior)
        post_dist = self.get_dist(posterior)
        if kl_balancing:
            post_no_grad = self.detach_state(posterior)
            prior_no_grad = self.detach_state(prior)
            
            # kl_balancing
            div_loss = kl_alpha * (torch.mean(divergence(
                            self.get_dist(post_no_grad), prior_dist)))
            div_loss += (1-kl_alpha) * (torch.mean(divergence(
                            post_dist, self.get_dist(prior_no_grad))))
        else:
            div_loss = torch.mean(divergence(post_dist, prior_dist))
            div_loss = torch.max(div_loss, div_loss.new_full(div_loss.size(), free_nats))
        return div_loss

    def stack_states(self, states, dim=0): # (T, B, ...)
        if self.discrete:
            return dict(
                logit  = torch.stack([state['logit'] for state in states], dim=dim),
                stoch = torch.stack([state['stoch'] for state in states], dim=dim),
                deter = torch.stack([state['deter'] for state in states], dim=dim))
        else:
            return dict(
                mean = torch.stack([state['mean'] for state in states], dim=dim),
                std  = torch.stack([state['std'] for state in states], dim=dim),
                stoch = torch.stack([state['stoch'] for state in states], dim=dim),
                deter = torch.stack([state['deter'] for state in states], dim=dim))

    def detach_state(self, state):
        if self.discrete:
            return dict(
                logit = state['logit'].detach(),
                stoch = state['stoch'].detach(),
                deter = state['deter'].detach())
        else:
            return dict(
                mean = state['mean'].detach(),
                std  = state['std'].detach(),
                stoch = state['stoch'].detach(),
                deter = state['deter'].detach())

    def seq_to_batch(self, state):
        if self.discrete:
            return dict(
                logit = torch.reshape(state['logit'], (state['logit'].shape[0]* state['logit'].shape[1], *state['logit'].shape[2:])),
                stoch = torch.reshape(state['stoch'], (state['stoch'].shape[0]* state['stoch'].shape[1], *state['stoch'].shape[2:])),
                deter = torch.reshape(state['deter'], (state['deter'].shape[0]* state['deter'].shape[1], *state['deter'].shape[2:])))
        else:
            return dict(
                mean = torch.reshape(state['mean'], (state['mean'].shape[0]* state['mean'].shape[1], *state['mean'].shape[2:])),
                std = torch.reshape(state['std'], (state['std'].shape[0]* state['std'].shape[1], *state['std'].shape[2:])),
                stoch = torch.reshape(state['stoch'], (state['stoch'].shape[0]* state['stoch'].shape[1], *state['stoch'].shape[2:])),
                deter = torch.reshape(state['deter'], (state['deter'].shape[0]* state['deter'].shape[1], *state['deter'].shape[2:])))

class ConvEncoder(nn.Module):

    def __init__(self, input_shape, embed_size, activation, depth=32):

        super().__init__()

        self.input_shape = input_shape
        self.act_fn = _str_to_activation[activation]
        self.depth = depth
        self.kernels = [4, 4, 4, 4]

        self.embed_size = embed_size
        
        layers = []
        for i, kernel_size in enumerate(self.kernels):
            in_ch = input_shape[0] if i==0 else self.depth * (2 ** (i-1))
            out_ch = self.depth * (2 ** i)
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size, stride=2))
            layers.append(self.act_fn)

        self.conv_block = nn.Sequential(*layers)
        self.fc = nn.Identity() if self.embed_size == 1024 else nn.Linear(1024, self.embed_size)

    def forward(self, inputs):
        reshaped = inputs.reshape(-1, *self.input_shape)
        embed = self.conv_block(reshaped)
        embed = torch.reshape(embed, (*inputs.shape[:-3], -1))
        embed = self.fc(embed)

        return embed

class ConvDecoder(nn.Module):
 
    def __init__(self, stoch_size, deter_size, output_shape, activation, depth=32, discrete=False):

        super().__init__()

        self.output_shape = output_shape
        self.depth = depth
        self.kernels = [5, 5, 6, 6]
        self.act_fn = _str_to_activation[activation]
        
        self.dense = nn.Linear(stoch_size + deter_size, 32*self.depth) if not discrete \
                else nn.Linear(stoch_size * discrete + deter_size, 32*self.depth)

        layers = []
        for i, kernel_size in enumerate(self.kernels):
            in_ch = 32*self.depth if i==0 else self.depth * (2 ** (len(self.kernels)-1-i))
            out_ch = output_shape[0] if i== len(self.kernels)-1 else self.depth * (2 ** (len(self.kernels)-2-i))
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=2))
            if i!=len(self.kernels)-1:
                layers.append(self.act_fn)

        self.convtranspose = nn.Sequential(*layers)

    def forward(self, features):
        out_batch_shape = features.shape[:-1]
        out = self.dense(features)
        out = torch.reshape(out, [-1, 32*self.depth, 1, 1])
        out = self.convtranspose(out)
        mean = torch.reshape(out, (*out_batch_shape, *self.output_shape))

        out_dist = distributions.independent.Independent(
            distributions.Normal(mean, 1), len(self.output_shape))

        return out_dist

class MaskConvDecoder(nn.Module):
 
    def __init__(self, stoch_size, deter_size, output_shape, activation, depth=32, discrete=False):

        super().__init__()

        self.output_shape = output_shape # (6, size, size)
        self.depth = depth
        self.kernels = [5, 5, 6, 6]
        self.act_fn = _str_to_activation[activation]
        
        self.dense = nn.Linear(stoch_size + deter_size, 32*self.depth) if not discrete \
                else nn.Linear(stoch_size * discrete + deter_size, 32*self.depth)

        layers = []
        for i, kernel_size in enumerate(self.kernels):
            in_ch = 32*self.depth if i==0 else self.depth * (2 ** (len(self.kernels)-1-i))
            out_ch = output_shape[0] if i== len(self.kernels)-1 else self.depth * (2 ** (len(self.kernels)-2-i))
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=2))
            if i!=len(self.kernels)-1:
                layers.append(self.act_fn)

        self.convtranspose = nn.Sequential(*layers)

    def forward(self, features):
        out_batch_shape = features.shape[:-1] # (T, B)
        out = self.dense(features)
        out = torch.reshape(out, [-1, 32*self.depth, 1, 1])
        out = self.convtranspose(out)
        mean, mask = torch.split(out, [self.output_shape[0]//2]*2, dim=1) # origin: (out, [3,3], dim=1), 默认channel=3
        mean = torch.reshape(mean, (*out_batch_shape, self.output_shape[0]//2, *self.output_shape[1:])) # (T, B, 3, size, size)
        mask = torch.reshape(mask, (*out_batch_shape, self.output_shape[0]//2, *self.output_shape[1:])) # (T, B, 3, size, size)
        
        out_dist = distributions.independent.Independent(
            distributions.Normal(mean, 1), len(self.output_shape))

        return out_dist, mask

class EnsembleMaskConvDecoder(nn.Module):

    def __init__(self, decoder1, decoder2):
        super().__init__()
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.output_shape = decoder1.output_shape # (3, size, size)
        self.mask_conv = nn.Sequential(nn.Conv2d(self.output_shape[0], 1, 1, stride=1), # origin: nn.Conv2d(6, 1, 1, stride=1)
                                       nn.Sigmoid())

    def forward(self, features1, features2):
        pred1, mask1 = self.decoder1(features1) # (T, B, 3, size, size)
        pred2, mask2 = self.decoder2(features2)
        mean1 = pred1.mean
        mean2 = pred2.mean
        mask_feat = torch.cat([mask1, mask2], dim=2) # (T, B, 6, size, size)
        mask_feat = torch.reshape(mask_feat, [-1, *self.output_shape]) # origin: [-1, 6, *self.output_shape[1:]
        mask = self.mask_conv(mask_feat) # (T*B, 1, size, size)
        mask = torch.reshape(mask, [*mask1.shape[:2], *mask.shape[1:]]) # # (T, B, 1, size, size)
        mean = mean1 * mask + mean2 * (1 - mask)
        out_dist = distributions.independent.Independent(
            distributions.Normal(mean, 1), len(self.output_shape))
        return out_dist, pred1, pred2, mask
    

# used for reward and value models
class DenseDecoder(nn.Module):

    def __init__(self, stoch_size, deter_size, output_shape, n_layers, units, activation, dist, discrete = False, action_size=None):

        super().__init__()

        self.input_size = stoch_size + deter_size if not discrete else stoch_size * discrete + deter_size
        if action_size is not None:
            self.input_size += action_size
        self.output_shape = output_shape
        self.n_layers = n_layers
        self.units = units
        self.act_fn = _str_to_activation[activation]
        self.dist = dist

        layers=[]

        for i in range(self.n_layers):
            in_ch = self.input_size if i==0 else self.units
            out_ch = self.units
            layers.append(nn.Linear(in_ch, out_ch))
            layers.append(self.act_fn) 

        layers.append(nn.Linear(self.units, int(np.prod(self.output_shape))))

        self.model = nn.Sequential(*layers)

    def forward(self, features):

        out = self.model(features)

        if self.dist == 'normal':
            return distributions.independent.Independent(
                distributions.Normal(out, 1), len(self.output_shape))
        if self.dist == 'binary':
            return Bernoulli(distributions.independent.Independent(
                distributions.Bernoulli(logits=out), len(self.output_shape)))
        if self.dist == 'none':
            return out

        raise NotImplementedError(self.dist)

class ActionDecoder(nn.Module):

    def __init__(self, action_size, stoch_size, deter_size, n_layers, units, 
                        activation, dist='trunc_normal', min_std=1e-4, init_std=5, mean_scale=5, discrete = False):

        super().__init__()

        self.action_size = action_size
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.discrete = discrete
        self.units = units  
        self.act_fn = _str_to_activation[activation]
        self.n_layers = n_layers
        self.dist = dist

        self._min_std = min_std
        self._init_std = init_std
        self._mean_scale = mean_scale

        layers = []
        for i in range(self.n_layers):
            if i == 0:
                in_ch = self.stoch_size + self.deter_size if not self.discrete else self.stoch_size * self.discrete + self.deter_size
            else:
                in_ch = self.units
            out_ch = self.units
            layers.append(nn.Linear(in_ch, out_ch))
            layers.append(self.act_fn)

        layers.append(nn.Linear(self.units, 2*self.action_size))
        self.action_model = nn.Sequential(*layers)

    def forward(self, features, deter=False, return_dist=False):

        out = self.action_model(features)
        mean, std = torch.chunk(out, 2, dim=-1) 

        if self.dist == 'trunc_normal':
            std = 2 * torch.sigmoid((std + self._init_std) / 2) + self._min_std

            dist = TruncNormalDist(torch.tanh(mean), std, -1, 1)
            dist = Independent(dist, 1)
        elif self.dist == 'tanh_normal':
            raw_init_std = np.log(np.exp(self._init_std) - 1)
            action_mean = self._mean_scale * torch.tanh(mean / self._mean_scale)
            action_std = F.softplus(std + raw_init_std) + self._min_std

            dist = distributions.Normal(action_mean, action_std)
            dist = TransformedDistribution(dist, TanhBijector())
            dist = distributions.independent.Independent(dist, 1)
            dist = SampleDist(dist)

        if return_dist:
            return dist

        if deter:
            return dist.mode()
        else:
            return dist.rsample()

    def add_exploration(self, action, action_noise=0.3):

        return torch.clamp(distributions.Normal(action, action_noise).rsample(), -1, 1)

class TanhBijector(distributions.Transform):

    def __init__(self):
        super().__init__()
        self.bijective = True
        self.domain = constraints.real
        self.codomain = constraints.interval(-1.0, 1.0)

    @property
    def sign(self): 
        return 1.

    def _call(self, x): 
        return torch.tanh(x)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    def _inverse(self, y: torch.Tensor):
        y = torch.where(
            (torch.abs(y) <= 1.),
            torch.clamp(y, -0.99999997, 0.99999997),
            y)
        y = self.atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2. * (np.log(2) - x - F.softplus(-2. * x))

class SampleDist:

    def __init__(self, dist, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        sample = self._dist.rsample(self._samples)
        return torch.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        return self._dist.sample()
    
    def rsample(self):
        return self._dist.rsample()

class OneHotDist(distributions.OneHotCategorical):

    def __init__(self, logits=None, probs=None, dtype=None, validate_args=False):
        self._sample_dtype = dtype or torch.float32  # FIXME tipo
        super().__init__(probs=probs, logits=logits,
                         validate_args=False)  # todo verify args, ignore for now due to error

        # FIXME event_shape -1 for now,because I think could be empty
        # if so, tf uses logits or probs shape[-1]
        self._mode = F.one_hot(torch.argmax(self.logits, -1), self.event_shape[-1]).float()  # fixme dtype

    @property
    def name(self):
        return 'OneHotDist'

    @property
    def mode(self):
        return self._mode

    def sample(self, sample_shape=(), seed=None):  # note doenst have rsample
        # Straight through biased gradient estimator.
        # FIXME seed is not possible here

        sample = super().sample(sample_shape)  # .type(self._sample_dtype) #FIXME
        probs = super().probs
        while len(probs.shape) < len(sample.shape):  # adds dims on 0
            probs = probs[None]

        sample += (probs - probs.detach())  # .type(self._sample_dtype)
        return sample

    def rsample(self, sample_shape=(), seed=None):  # note doenst have rsample
        return self.sample(sample_shape, seed=seed)

    # custom log_prob more stable
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        value, logits = torch.broadcast_tensors(value, self.logits)
        indices = value.max(-1)[1]

        # reshapes are not cool
        ret = -F.cross_entropy(logits.reshape(-1, *self.event_shape), indices.reshape(-1).detach(), reduction='none')

        return torch.reshape(ret, logits.shape[:-1])

import math
from numbers import Number
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)

class TruncatedStandardNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        'a': constraints.real,
        'b': constraints.real,
    }
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError('Truncation bounds types are different')
        if any((self.a >= self.b).view(-1, ).tolist()):
            raise ValueError('Incorrect truncation range')
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        self._lpbb_m_lpaa_d_Z = (self._little_phi_b * self.b - self._little_phi_a * self.a) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._mode = torch.minimum(torch.maximum(torch.ones_like(self.a), self.a), self.b)  # TODO pull request?
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def mode(self):
        return self._mode

    @property
    def variance(self):
        return self._variance

    # @property #In pytorch is a function
    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x ** 2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value ** 2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        # icdf is numerically unstable; as a consequence, so is rsample.
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)


class TruncatedNormal(TruncatedStandardNormal):
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True

    def __init__(self, loc, scale, scalar_a, scalar_b, validate_args=True):
        self.loc, self.scale, a, b = broadcast_all(loc, scale, scalar_a, scalar_b)
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._mode = torch.clamp(self.loc, scalar_a, scalar_b)  # pull request?
        self._variance = self._variance * self.scale ** 2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        return super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale

class TruncNormalDist(TruncatedNormal):

    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale, low, high)
        self._clip = clip
        self._mult = mult

        self.low = low
        self.high = high

    def sample(self, *args, **kwargs):
        event = super().rsample(*args, **kwargs)
        if self._clip:

            clipped = torch.clamp(event, self.low + self._clip, self.high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event

class Independent(distributions.Independent):
    @property
    def mode(self):
        return self.base_dist.mode
    
    def rsample(self):
        return self.base_dist.rsample()
    

class Bernoulli:

    def __init__(self, dist=None):
        super().__init__()
        self._dist = dist
        self.mean = dist.mean

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def entropy(self):
        return self._dist.entropy()

    def mode(self):
        _mode = torch.round(self._dist.mean)
        return _mode.detach() +self._dist.mean - self._dist.mean.detach()

    def sample(self, sample_shape=()):
        return self._dist.rsample(sample_shape)

    def log_prob(self, x):
        _logits = self._dist.base_dist.logits
        log_probs0 = -F.softplus(_logits)
        log_probs1 = -F.softplus(-_logits)

        return log_probs0 * (1-x) + log_probs1 * x