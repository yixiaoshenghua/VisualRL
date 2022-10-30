import torch
import torch.nn as nn
import numpy as np
from encoder import OUT_DIM
import torch.distributions as td

class PixelDecoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = OUT_DIM[num_layers]

        self.fc = nn.Linear(
            feature_dim, num_filters * self.out_dim * self.out_dim
        )

        self.deconvs = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.deconvs.append(
                nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1)
            )
        self.deconvs.append(
            nn.ConvTranspose2d(
                num_filters, obs_shape[0], 3, stride=2, output_padding=1
            )
        )

        self.outputs = dict()

    def forward(self, h):
        h = torch.relu(self.fc(h))
        self.outputs['fc'] = h

        deconv = h.view(-1, self.num_filters, self.out_dim, self.out_dim)
        self.outputs['deconv1'] = deconv

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
            self.outputs['deconv%s' % (i + 1)] = deconv

        obs = self.deconvs[-1](deconv)
        self.outputs['obs'] = obs

        return obs

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_decoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_decoder/%s_i' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param(
                'train_decoder/deconv%s' % (i + 1), self.deconvs[i], step
            )
        L.log_param('train_decoder/fc', self.fc, step)





class MaskDecoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32):
        super(MaskDecoder, self).__init__()
        self.num_layers = num_layers
        self.obs_shape = obs_shape
        if obs_shape[-1] == 64:
            self.out_dim = 32 * num_filters
            self.kernel_sizes = [5, 5, 6, 6]
        elif obs_shape[-1] == 32:
            self.out_dim = 8 * num_filters
            self.kernel_sizes = [3, 3, 3, 4]
        elif obs_shape[-1] == 84:
            self.out_dim = 72 * num_filters
            self.kernel_sizes = [7, 6, 6, 6]
        output_channels = [4*num_filters, 2*num_filters, 1*num_filters, 3+obs_shape[0]]
        self.fc = nn.Linear(
            feature_dim, self.out_dim
        )
        self.deconvs = nn.ModuleList()
        last_dim = self.out_dim
        for i in range(len(self.kernel_sizes)):
            self.deconvs.append(
                nn.ConvTranspose2d(last_dim, output_channels[i], self.kernel_sizes[i], stride=2),
            )
            last_dim = output_channels[i]
        self.outputs = dict()

    def forward(self, x):
        h = self.fc(x)
        self.outputs['fc'] = h

        deconv = h.view((-1, self.out_dim, 1, 1))
        self.outputs['deconv'] = deconv

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
            self.outputs['deconv%s' % (i + 1)] = deconv

        obs = self.deconvs[-1](deconv)
        self.outputs['obs'] = obs

        mean, mask = torch.split(obs, [3, 3], dim=1)
        mean = mean.reshape(list(x.shape[:-1])+list(self.obs_shape))
        mask = mask.reshape(list(x.shape[:-1])+list(self.obs_shape))
        self.outputs['mean'] = mean
        self.outputs['mask'] = mask

        return td.Independent(td.Normal(mean, 1.0), len(self.obs_shape)), mask

class EnsembleMaskDecoder(nn.Module):
    """
    ensemble two convdecoder with <Normal, mask> outputs
    """
    def __init__(self, decoder1, decoder2, dtype):
        super(EnsembleMaskDecoder, self).__init__()
        self.decoder1 = decoder1
        self.decoder2 = decoder2
        self.obs_shape = decoder1.obs_shape
        self.dtype = dtype
        self.conv = nn.Conv2d(2 * self.obs_shape[0], 1, 1, stride=2)
        self.outputs = dict()

    def forward(self, feat1, feat2):
        pred1, mask1 = self.decoder1(feat1)
        pred2, mask2 = self.decoder2(feat2)
        mean1, mean2 = pred1.base_distribution.loc, pred2.base_distribution.loc
        mask_feat = torch.cat([mask1, mask2], dim=1)
        self.outputs['mask_feat'] = mask_feat
        mask = torch.sigmoid(self.conv(mask_feat))
        self.outputs['mask'] = mask
        mask_use1, mask_use2 = mask, 1 - mask
        mean = mean1 * mask_use1.to(self.dtype) + mean2 * mask_use2.to(self.dtype)
        self.outputs['mean'] = mean
        return td.Independent(td.Normal(mean, 1.0), len(self.obs_shape)), pred1, pred2, mask_use1.to(self.dtype)

class DenseDecoder(nn.Module):
    def __init__(self, shape, num_layers, input_dim, num_units, dist='normal', act=nn.ELU):
        super(DenseDecoder, self).__init__()
        self.shape = shape
        self.num_layers = num_layers
        self.num_units = num_units
        self.dist = dist
        self.act = act()
        self.outputs = dict()
        self.net = nn.ModuleList()
        last_dim = input_dim
        for i in range(self.num_layers):
            self.net.append(nn.Linear(last_dim, self.num_units))
            last_dim = self.num_units
        self.net.append(nn.Linear(last_dim, np.prod(self.shape)))

    def forward(self, x):
        h = x
        for i in range(self.num_layers):
            h = self.act(self.net[i](h))
        h = self.net[-1](h)
        h = h.reshape(list(x.shape[:-1]) + list(self.shape))
        self.outputs['h'] = h
        if self.dist == 'normal':
            return td.Independent(td.Normal(h, 1.0), len(self.shape))
        elif self.dist == 'binary':
            return td.Independent(td.Bernoulli(h), len(self.shape))
        raise NotImplementedError(self.dist)


_AVAILABLE_DECODERS = {'pixel': PixelDecoder, 'mask': MaskDecoder, 'ensemble': EnsembleMaskDecoder, 'dense': DenseDecoder}


def make_decoder(
    decoder_type, obs_shape, feature_dim, num_layers, num_filters
):
    assert decoder_type in _AVAILABLE_DECODERS
    return _AVAILABLE_DECODERS[decoder_type](
        obs_shape, feature_dim, num_layers, num_filters
    )