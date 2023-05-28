import torch
import numpy as np
import torch.nn as nn
import os
from collections import deque, namedtuple
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
import kornia
import copy
from utils.data_augs import center_crop_images, random_crop

class MBReplayBuffer:

    def __init__(self, size, obs_shape, action_size, seq_len, batch_size):

        self.size = size
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.idx = 0
        self.full = False
        self.observations = np.empty((size, *obs_shape), dtype=np.uint8) 
        self.actions = np.empty((size, action_size), dtype=np.float32)
        self.rewards = np.empty((size,), dtype=np.float32) 
        self.terminals = np.empty((size,), dtype=np.float32)
        self.steps, self.episodes = 0, 0
    
    def add(self, obs, ac, rew, done):

        self.observations[self.idx] = obs['image']
        self.actions[self.idx] = ac
        self.rewards[self.idx] = rew
        self.terminals[self.idx] = done
        self.idx = (self.idx + 1) % self.size
        self.full = self.full or self.idx == 0
        self.steps += 1 
        self.episodes = self.episodes + (1 if done else 0)

    def _sample_idx(self, L):

        valid_idx = False
        while not valid_idx:
            idx = np.random.randint(0, self.size if self.full else self.idx - L)
            idxs = np.arange(idx, idx + L) % self.size
            valid_idx = not self.idx in idxs[1:] 
        return idxs

    def _retrieve_batch(self, idxs, n, L):
        
        vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
        observations = self.observations[vec_idxs]
        return observations.reshape(L, n, *observations.shape[1:]), self.actions[vec_idxs].reshape(L, n, -1), self.rewards[vec_idxs].reshape(L, n), self.terminals[vec_idxs].reshape(L, n)

    def sample(self):
        n = self.batch_size
        l = self.seq_len
        obs, acs, rews, terms= self._retrieve_batch(np.asarray([self._sample_idx(l) for _ in range(n)]), n, l)
        return obs, acs, rews, terms



class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""
    def __init__(
        self, obs_shape, action_shape, capacity, batch_size, device,
        path_len=None, image_size=84, transform=None
    ):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False
        self._path_len = path_len

    def add(self, obs, action, reward, next_obs, done):

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_proprio(self):

        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones

    def sample_cpc(self):
        # start = time.time()
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        pos = obses.copy()

        obses = random_crop(obses, self.image_size)
        next_obses = random_crop(next_obses, self.image_size)
        pos = random_crop(pos, self.image_size)

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        pos = torch.as_tensor(pos, device=self.device).float()
        cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos,
                          time_anchor=None, time_pos=None)

        return obses, actions, rewards, next_obses, not_dones, cpc_kwargs

    def _sample_sequential_idx(self, n, L):
        # Returns an index for a valid single chunk uniformly sampled from the
        # memory
        idx = np.random.randint(
            0, self.capacity - L if self.full else self.idx - L, size=n
        )
        pos_in_path = idx - idx // self._path_len * self._path_len
        idx[pos_in_path > self._path_len - L] = idx[
            pos_in_path > self._path_len - L
        ] // self._path_len * self._path_len + L
        idxs = np.zeros((n, L), dtype=np.int)
        for i in range(n):
            idxs[i] = np.arange(idx[i], idx[i] + L)
        return idxs.transpose().reshape(-1)

    def sample_multi_view(self, n, L):
        # start = time.time()
        idxs = self._sample_sequential_idx(n, L)
        obses = self.obses[idxs]
        pos = obses.copy()

        obses = random_crop(obses, out=self.image_size)
        pos = random_crop(pos, out=self.image_size)

        obses = torch.as_tensor(obses, device=self.device).float()\
            .reshape(L, n, *obses.shape[-3:])
        actions = torch.as_tensor(self.actions[idxs], device=self.device)\
            .reshape(L, n, -1)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)\
            .reshape(L, n)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)\
            .reshape(L, n)

        pos = torch.as_tensor(pos, device=self.device).float()\
            .reshape(L, n, *obses.shape[-3:])
        mib_kwargs = dict(view1=obses, view2=pos,
                          time_anchor=None, time_pos=None)

        return obses, actions, rewards, not_dones, mib_kwargs

    def sample_sequence(self, n, L):
        # start = time.time()
        idxs = self._sample_sequential_idx(n, L)
        obses = self.obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()\
            .reshape(L, n, *obses.shape[-3:])
        actions = torch.as_tensor(self.actions[idxs], device=self.device)\
            .reshape(L, n, -1)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)\
            .reshape(L, n)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)\
            .reshape(L, n)

        return obses, actions, rewards, not_dones

    def sample(self, k=False):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        if k:
            return obses, actions, rewards, next_obses, not_dones, torch.as_tensor(self.k_obses[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity


class ReplayBufferFLARE(Dataset):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device,
        image_size=84, pre_image_size=100, transform=None, frame_stack=3, 
        augment_target_same_rnd=True, camera_id=None):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.pre_image_size = pre_image_size
        self.transform = transform
        self.obs_shape = obs_shape
        self.camera_id = camera_id
        self.frame_stack = frame_stack
        self.number_channel = obs_shape[0]
        self.augment_target_same_rnd = augment_target_same_rnd

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        if len(obs_shape)==1:
            self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.eps_not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False


    def add(self, obs, action, reward, next_obs, done, eps_done):
        if len(obs.shape) == 1:
            np.copyto(self.obses[self.idx], obs)
            np.copyto(self.next_obses[self.idx], next_obs)
        else:
            np.copyto(self.obses[self.idx], obs[-1 * self.number_channel:, :, :])
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.eps_not_dones[self.idx], not eps_done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_proprio(self):
        
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
        
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones


    def sample_rad(self,aug_funcs):
        
        # augs specified as flags
        # curl_sac organizes flags into aug funcs
        # passes aug funcs into sampler

        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
        capacity = self.capacity if self.full else self.idx

        idxs_current = copy.deepcopy(idxs)
        # avoid using the last one
        idxs_current = [x-1 if not self.eps_not_dones[x] else x for x in idxs_current]
        idxs_next = [x if x+1 >= capacity or not self.eps_not_dones[x] else x+1 for x in idxs_current] 
        idxs_list = [copy.deepcopy(idxs_next), copy.deepcopy(idxs_current)]
        if self.frame_stack > 1:
            idxs_prev = idxs_current
            for t in range(1, self.frame_stack):
                idxs_prev = [x if x-1 < 0 or not self.eps_not_dones[x-1] else x-1 for x in idxs_prev]
                idxs_list.append(copy.deepcopy(idxs_prev))

        obses = []
        for t in range(self.frame_stack):
            obses.append(self.obses[idxs_list[-1 - 1 * t]]) #-1 to - self.frame_stak 
        obses = np.concatenate(obses, axis=1)
        pos = obses.copy()

        next_obses = []
        for t in range(self.frame_stack):
            next_obses.append(self.obses[idxs_list[-2 - 1 * t]]) 
        next_obses = np.concatenate(next_obses, axis=1)

        og_obses = center_crop_images(obses, self.pre_image_size)
        og_next_obses = center_crop_images(next_obses, self.pre_image_size)
        og_pos = center_crop_images(pos, self.pre_image_size)

        if aug_funcs:
            for aug,func in aug_funcs.items():
                # apply crop and cutout first
                if 'crop' in aug or 'cutout' in aug or 'window' in aug:
                    og_obses = func(obses, self.pre_image_size)
                    og_next_obses = func(next_obses, self.pre_image_size)
                    og_pos = func(pos, self.pre_image_size)
                if 'translate' in aug:
                    obses, rndm_idxs = func(og_obses, self.image_size, return_random_idxs=True)
                    if self.augment_target_same_rnd:
                        next_obses = func(og_next_obses, self.image_size, **rndm_idxs)
                    else:
                        next_obses = func(og_next_obses, self.image_size)
        
        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs_current], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs_current], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs_current], device=self.device)
        pos = torch.as_tensor(pos, device=self.device).float()

        obses = obses / 255.
        next_obses = next_obses / 255.
        pos = pos / 255.

        # other augmentations go here
        if aug_funcs:
            for aug,func in aug_funcs.items():
                # skip crop and cutout augs
                if 'crop' in aug or 'cutout' in aug or 'translate' in aug or 'window' in aug:
                    continue
                obses = func(obses)
                next_obses = func(next_obses)
                pos = func(pos)
        
        cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos, 
                          time_anchor=None, time_pos=None)
        
        return obses, actions, rewards, next_obses, not_dones, cpc_kwargs

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity 

#TODO: When to use this replay buffer?
# class ReplayBuffer(object):
#     """Buffer to store environment transitions."""
#     def __init__(self, obs_shape, action_shape, capacity, image_pad, device):
#         self.capacity = capacity
#         self.device = device

#         self.aug_trans = nn.Sequential(
#             nn.ReplicationPad2d(image_pad),
#             kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))

#         self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
#         self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
#         self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
#         self.rewards = np.empty((capacity, 1), dtype=np.float32)
#         self.not_dones = np.empty((capacity, 1), dtype=np.float32)
#         self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

#         self.idx = 0
#         self.full = False

#     def __len__(self):
#         return self.capacity if self.full else self.idx

#     def add(self, obs, action, reward, next_obs, done, done_no_max):
#         np.copyto(self.obses[self.idx], obs)
#         np.copyto(self.actions[self.idx], action)
#         np.copyto(self.rewards[self.idx], reward)
#         np.copyto(self.next_obses[self.idx], next_obs)
#         np.copyto(self.not_dones[self.idx], not done)
#         np.copyto(self.not_dones_no_max[self.idx], not done_no_max)

#         self.idx = (self.idx + 1) % self.capacity
#         self.full = self.full or self.idx == 0

#     def sample(self, batch_size):
#         idxs = np.random.randint(0,
#                                  self.capacity if self.full else self.idx,
#                                  size=batch_size)

#         obses = self.obses[idxs]
#         next_obses = self.next_obses[idxs]
#         obses_aug = obses.copy()
#         next_obses_aug = next_obses.copy()

#         obses = torch.as_tensor(obses, device=self.device).float()
#         next_obses = torch.as_tensor(next_obses, device=self.device).float()
#         obses_aug = torch.as_tensor(obses_aug, device=self.device).float()
#         next_obses_aug = torch.as_tensor(next_obses_aug,
#                                          device=self.device).float()
#         actions = torch.as_tensor(self.actions[idxs], device=self.device)
#         rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
#         not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs],
#                                            device=self.device)

#         obses = self.aug_trans(obses)
#         next_obses = self.aug_trans(next_obses)

#         obses_aug = self.aug_trans(obses_aug)
#         next_obses_aug = self.aug_trans(next_obses_aug)

#         return obses, actions, rewards, next_obses, not_dones_no_max, obses_aug, next_obses_aug
