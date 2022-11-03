import torch
import numpy as np
import torch.nn as nn
import gym
import os
import sys
from collections import deque, namedtuple
import random
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

def get_device(memory_limits=6000):
    import os
    os.system("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp.txt")
    memory_gpu = [int(x.split()[2]) for x in open("tmp.txt", "r").readlines()]
    os.system("rm tmp.txt")
    for i, memory in enumerate(memory_gpu):
        if memory > memory_limits:
            print("CUDA : {}, free memory : {} Mb".format(i, memory))
            return i
    print("Fail to assign cuda due to memory limitation!!!")
    return -1

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )

'''https://github.com/yusukeurakami/dreamer-pytorch/utils.py'''
def lambda_return(imged_reward, value_pred, bootstrap, discount=0.99, lambda_=0.95):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    next_values = torch.cat([value_pred[1:], bootstrap[None]], 0)
    discount_tensor = discount * torch.ones_like(imged_reward)  # pcont
    inputs = imged_reward + discount_tensor * next_values * (1 - lambda_)
    last = bootstrap
    indices = reversed(range(len(inputs)))
    outputs = []
    for index in indices:
        inp, disc = inputs[index], discount_tensor[index]
        last = inp + disc * lambda_ * last
        outputs.append(last)
    outputs = list(reversed(outputs))
    outputs = torch.stack(outputs, 0)
    returns = outputs
    return returns

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


def random_crop(imgs, out=84):
    """
        args:
        imgs: np.array shape (B,C,H,W)
        out: output size (e.g. 84)
        returns np.array
    """
    n, c, h, w = imgs.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = np.empty((n, c, out, out), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):

        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    return cropped


def grayscale(imgs):
    # imgs: b x c x h x w
    device = imgs.device
    b, c, h, w = imgs.shape
    frames = c // 3

    imgs = imgs.view([b, frames, 3, h, w])
    imgs = imgs[:, :, 0, ...] * 0.2989 + imgs[:, :, 1, ...] * 0.587 + \
        imgs[:, :, 2, ...] * 0.114

    imgs = imgs.type(torch.uint8).float()
    # assert len(imgs.shape) == 3, imgs.shape
    imgs = imgs[:, :, None, :, :]
    imgs = imgs * torch.ones(
        [1, 1, 3, 1, 1], dtype=imgs.dtype
    ).float().to(device)  # broadcast tiling
    return imgs


def random_grayscale(images, p=.3):
    """
        args:
        imgs: torch.tensor shape (B,C,H,W)
        device: cpu or cuda
        returns torch.tensor
    """
    device = images.device
    in_type = images.type()
    images = images * 255.
    images = images.type(torch.uint8)
    # images: [B, C, H, W]
    bs, channels, h, w = images.shape
    images = images.to(device)
    gray_images = grayscale(images)
    rnd = np.random.uniform(0., 1., size=(images.shape[0],))
    mask = rnd <= p
    mask = torch.from_numpy(mask)
    frames = images.shape[1] // 3
    images = images.view(*gray_images.shape)
    mask = mask[:, None] * torch.ones([1, frames]).type(mask.dtype)
    mask = mask.type(images.dtype).to(device)
    mask = mask[:, :, None, None, None]
    out = mask * gray_images + (1 - mask) * images
    out = out.view([bs, -1, h, w]).type(in_type) / 255.
    return out


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

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]
    return image


def center_crop_images(image, output_size):
    h, w = image.shape[2:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, :, top:top + new_h, left:left + new_w]
    return image


def center_translate(image, size):
    c, h, w = image.shape
    assert size >= h and size >= w
    outs = np.zeros((c, size, size), dtype=image.dtype)
    h1 = (size - h) // 2
    w1 = (size - w) // 2
    outs[:, h1:h1 + h, w1:w1 + w] = image
    return outs

class AgentDistractorEnv(gym.Wrapper):
    '''
    Combine the distractor envs with the main env with 2x2 observation
    '''
    def __init__(self, main_env, dis_envs, unchange=False):
        gym.Wrapper.__init__(self, main_env)
        self.dis_envs = dis_envs
        self._max_episode_steps = self.env._max_episode_steps
        self.unchange = unchange
        if self.unchange:
            self.dis_obs = []
            for env in self.dis_envs:
                obs = env.reset()
                self.dis_obs.append(obs)

    def reset(self):
        self.env.reset()
        for env in self.dis_envs:
            env.reset()
        return self._get_obs()
    
    def step(self, action):
        main_obs, reward, done, info = self.env.step(action)
        obs = self._combine_obs(main_obs)
        return obs, reward, done, info
            
    def _get_obs(self):
        obs = self.env.render(mode='rgb_array', camera_id=0).transpose(2, 0, 1)
        obs = self._combine_obs(obs)
        return obs

    def _combine_obs(self, main_obs):
        channel, height, width = main_obs.shape
        final_obs = np.zeros((channel, height*2, width*2), dtype=np.uint8)
        final_obs[:, :height, :width] = main_obs
        if not self.unchange:
            for i, env in enumerate(self.dis_envs):
                random_action = env.action_space.sample()
                obs, _, _, _ = env.step(random_action)
                if i == 0:
                    final_obs[:, height:, :width] = obs if not self.unchange else self.dis_obs[i]
                elif i == 1:
                    final_obs[:, :height, width:] = obs if not self.unchange else self.dis_obs[i]
                else:
                    final_obs[:, height:, width:] = obs if not self.unchange else self.dis_obs[i]
        else:
            for i in range(len(self.dis_envs)):
                if i == 0:
                    final_obs[:, height:, :width] = self.dis_obs[i]
                elif i == 1:
                    final_obs[:, :height, width:] = self.dis_obs[i]
                else:
                    final_obs[:, height:, width:] = self.dis_obs[i]
        final_obs = np.array(Image.fromarray(final_obs.transpose(1, 2, 0)).resize((height, width))).transpose(2, 0, 1)
        return final_obs

class AgentAlphaEnv(gym.Wrapper):
    '''
    Combine the distractor envs with the main env with alpha channel
    '''
    def __init__(self, main_env, alpha_env, unchange=False, alpha=0.4):
        gym.Wrapper.__init__(self, main_env)
        self.alpha_env = alpha_env
        self._max_episode_steps = self.env._max_episode_steps
        self.unchange = unchange
        self.alpha = alpha
        if self.unchange:
            self.dis_obs = self.alpha_env.reset()

    def reset(self):
        self.env.reset()
        self.alpha_env.reset()
        return self._get_obs()
    
    def step(self, action):
        main_obs, reward, done, info = self.env.step(action)
        obs = self._combine_obs(main_obs)
        return obs, reward, done, info
            
    def _get_obs(self):
        obs = self.env.render(mode='rgb_array', camera_id=0).transpose(2, 0, 1)
        obs = self._combine_obs(obs)
        return obs

    def _combine_obs(self, main_obs):
        if not self.unchange:
            alpha_obs, _, _, _ = self.alpha_env.step(self.alpha_env.action_space.sample())
        else:
            alpha_obs = self.dis_obs
        alpha_obs = Image.fromarray(alpha_obs.transpose(1, 2, 0)).convert('RGBA')
        main_obs = Image.fromarray(main_obs.transpose(1, 2, 0)).convert('RGBA')
        alpha_obs.putalpha(ImageEnhance.Brightness(alpha_obs.split()[-1]).enhance(self.alpha))
        final_obs = Image.alpha_composite(main_obs, alpha_obs).convert('RGB')
        final_obs = np.array(final_obs).transpose(2, 0, 1)
        return final_obs

RESERVED_NAMES = ("get", "items")


def tuple_itemgetter(i):
    def _tuple_itemgetter(obj):
        return tuple.__getitem__(obj, i)
    return _tuple_itemgetter


def namedarraytuple(typename, field_names, return_namedtuple_cls=False,
        classname_suffix=False):
    """
    Returns a new subclass of a namedtuple which exposes indexing / slicing
    reads and writes applied to all contained objects, which must share
    indexing (__getitem__) behavior (e.g. numpy arrays or torch tensors).

    (Code follows pattern of collections.namedtuple.)

    >>> PointsCls = namedarraytuple('Points', ['x', 'y'])
    >>> p = PointsCls(np.array([0, 1]), y=np.array([10, 11]))
    >>> p
    Points(x=array([0, 1]), y=array([10, 11]))
    >>> p.x                         # fields accessible by name
    array([0, 1])
    >>> p[0]                        # get location across all fields
    Points(x=0, y=10)               # (location can be index or slice)
    >>> p.get(0)                    # regular tuple-indexing into field
    array([0, 1])
    >>> x, y = p                    # unpack like a regular tuple
    >>> x
    array([0, 1])
    >>> p[1] = 2                    # assign value to location of all fields
    >>> p
    Points(x=array([0, 2]), y=array([10, 2]))
    >>> p[1] = PointsCls(3, 30)     # assign to location field-by-field
    >>> p
    Points(x=array([0, 3]), y=array([10, 30]))
    >>> 'x' in p                    # check field name instead of object
    True
    """
    nt_typename = typename
    if classname_suffix:
        nt_typename += "_nt"  # Helpful to identify which style of tuple.
        typename += "_nat"

    try:  # For pickling, get location where this function was called.
        # NOTE: (pickling might not work for nested class definition.)
        module = sys._getframe(1).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
        module = None
    NtCls = namedtuple(nt_typename, field_names, module=module)

    def __getitem__(self, loc):
        try:
            return type(self)(*(None if s is None else s[loc] for s in self))
        except IndexError as e:
            for j, s in enumerate(self):
                if s is None:
                    continue
                try:
                    _ = s[loc]
                except IndexError:
                    raise Exception(f"Occured in {self.__class__} at field "
                        f"'{self._fields[j]}'.") from e

    __getitem__.__doc__ = (f"Return a new {typename} instance containing "
        "the selected index or slice from each field.")

    def __setitem__(self, loc, value):
        """
        If input value is the same named[array]tuple type, iterate through its
        fields and assign values into selected index or slice of corresponding
        field.  Else, assign whole of value to selected index or slice of
        all fields.  Ignore fields that are both None.
        """
        if not (isinstance(value, tuple) and  # Check for matching structure.
                getattr(value, "_fields", None) == self._fields):
            # Repeat value for each but respect any None.
            value = tuple(None if s is None else value for s in self)
        try:
            for j, (s, v) in enumerate(zip(self, value)):
                if s is not None or v is not None:
                    s[loc] = v
        except (ValueError, IndexError, TypeError) as e:
            raise Exception(f"Occured in {self.__class__} at field "
                f"'{self._fields[j]}'.") from e

    def __contains__(self, key):
        "Checks presence of field name (unlike tuple; like dict)."
        return key in self._fields

    def get(self, index):
        "Retrieve value as if indexing into regular tuple."
        return tuple.__getitem__(self, index)

    def items(self):
        "Iterate ordered (field_name, value) pairs (like OrderedDict)."
        for k, v in zip(self._fields, self):
            yield k, v

    for method in (__getitem__, __setitem__, get, items):
        method.__qualname__ = f'{typename}.{method.__name__}'

    arg_list = repr(NtCls._fields).replace("'", "")[1:-1]
    class_namespace = {
        '__doc__': f'{typename}({arg_list})',
        '__slots__': (),
        '__getitem__': __getitem__,
        '__setitem__': __setitem__,
        '__contains__': __contains__,
        'get': get,
        'items': items,
    }

    for index, name in enumerate(NtCls._fields):
        if name in RESERVED_NAMES:
            raise ValueError(f"Disallowed field name: {name}.")
        itemgetter_object = tuple_itemgetter(index)
        doc = f'Alias for field number {index}'
        class_namespace[name] = property(itemgetter_object, doc=doc)

    result = type(typename, (NtCls,), class_namespace)
    result.__module__ = NtCls.__module__

    if return_namedtuple_cls:
        return result, NtCls
    return result