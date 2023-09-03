import gym
import os
import random
from collections import deque
import numpy as np
from PIL import Image
import skvideo.io
import cv2
import tqdm

DMC_ENV = ['cheetah', 'walker', 'hopper', 'finger', 'quadruped', 'reacher', 'ball']
GYM_ROBOT_ENV = ['Fetch']

# -------------------------------- Make Environment from args ------------------------------------------

def make_env(args):
    if args.env.split('-')[0] in DMC_ENV:
        env = DeepMindControl(args, '-'.join(args.env.split('-')[:2]), args.seed, camera=args.camera_id)
        if args.env.split('-')[-1] == 'video':
            env = RandomVideoSource(env, args.video_dir, total_frames=1000, grayscale=True)
    elif args.env.split('-')[0] in GYM_ROBOT_ENV:
        env = GymRobotEnv(args, '-'.join(args.env.split('-')[:2]), args.seed, camera=args.camera_id)
    else:
        env = Claw(args, args.env, size=(64, 64))
    env = ActionRepeat(env, args.action_repeat)
    env = NormalizeActions(env)
    env = TimeLimit(env, args.time_limit / args.action_repeat)
    env = FrameStack(env, args.frame_stack) # TODO 参考FrameStack中具体问题
    #env = RewardObs(env)
    return env


# --------------------------------  Some Environments ------------------------------------------

class GymRobotEnv:
    def __init__(self, args, name, seed, size=(64, 64), camera=None):
        self.args = args
        domain, task = name.split('-', 1)
        self._env = gym.make(domain+task+'-v1')
        self._env.seed(seed)
        self._size = size
        self._camera = camera
        self._steps = 0

    @property
    def observation_space(self):
        spaces = {}
        spaces['image'] = gym.spaces.Box(
            0, 255, (3,) + self._size , dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        self._steps += 1
        _, reward, done, info = self._env.step(action)
        obs = dict()
        obs['image'] = self._env.render('rgb_array', width=self._size[1], height=self._size[0]).transpose(2, 0, 1).copy()
        return obs, reward, done, info
    
    def reset(self):
        self._steps = 0
        state = self._env.reset()
        obs = dict()
        obs['image'] = self._env.render('rgb_array', width=self._size[1], height=self._size[0]).transpose(2, 0, 1).copy()
        return obs

class DeepMindControl:

    def __init__(self, args, name, seed, camera=None):
        self.args = args
        domain, task = name.split('-', 1)
        if domain == 'cup':  # Only domain with multiple words.
          domain = 'ball_in_cup'
        if isinstance(domain, str):
          from dm_control import suite
          self._env = suite.load(domain, task, task_kwargs={'random':seed})
        else:
          assert task is None
          self._env = domain()
        self._size = (args.pre_transform_image_size, args.pre_transform_image_size)
        if camera is None:
          camera = dict(quadruped=2).get(domain, 0)
        self._camera = camera

    @property
    def observation_space(self):
        spaces = {}
        for key, value in self._env.observation_spec().items():
          spaces[key] = gym.spaces.Box(
              -np.inf, np.inf, value.shape, dtype=np.float32)
        spaces['image'] = gym.spaces.Box(
            0, 255, (3,) + self._size , dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    def step(self, action):
        self._steps += 1
        time_step = self._env.step(action)
        obs = dict(time_step.observation)
        obs['image'] = self.render().transpose(2, 0, 1).copy()
        reward = time_step.reward or 0
        done = time_step.last()
        info = {'discount': np.array(time_step.discount, np.float32)}
        return obs, reward, done, info

    def reset(self):
        self._steps = 0
        time_step = self._env.reset()
        obs = dict(time_step.observation)
        obs['image'] = self.render().transpose(2, 0, 1).copy()
        return obs

    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
          raise ValueError("Only render mode 'rgb_array' is supported.")
        # if self.args.change_camera_freq > 0:
        #     return self._env.physics.render(*self._size, camera_id=int(int(self._steps//self.args.change_camera_freq)%2))
        return self._env.physics.render(*self._size, camera_id=self._camera)

class Claw:
    def __init__(self, args, name, size=(128, 128)):
        self.args = args
        self._env = gym.make(name)
        self.size = size
        self._env.sim_scene.renderer.set_free_camera_settings(distance = 0.5,
                                                              azimuth = 180.0,
                                                              elevation = -30.0,
                                                              lookat = [0.0, 0.0, 0.2])
    
    @property
    def observation_space(self):
        spaces = {}
        spaces['image'] = gym.spaces.Box(
            0, 255, (3,) + self.size , dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return self._env.action_space

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._env, attr) 

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        img = self._env.render(mode='rgb_array', width = self.size[0], height = self.size[1])
        if img.shape[-1] == 3:
            img = img.transpose(2, 0, 1)
        obs = {'state':state, 'image':img}
        obs_dict = self._env.get_obs_dict()
        obs['proprio'] = obs_dict['proprio']
        return obs, reward, done, info

    def reset(self):
        state = self._env.reset()
        img = self._env.render(mode='rgb_array', width = self.size[0], height = self.size[1])
        if img.shape[-1] == 3:
            img = img.transpose(2, 0, 1)
        obs = {'state':state, 'image':img}
        obs_dict = self._env.get_obs_dict()
        obs['proprio'] = obs_dict['proprio']
        return obs
    
    def render(self, *args, **kwargs):
        if kwargs.get('mode', 'rgb_array') != 'rgb_array':
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self._env.render(mode='rgb_array', width = self.size[0], height = self.size[1])

# --------------------------------  Some Env Wrappers ------------------------------------------

class TimeLimit:

    def __init__(self, env, duration):
        self._env = env
        self._duration = duration
        self._step = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        assert self._step is not None, 'Must reset environment.'
        obs, reward, done, info = self._env.step(action)
        self._step += 1
        if self._step >= self._duration:
          done = True
          if 'discount' not in info:
            info['discount'] = np.array(1.0).astype(np.float32)
          self._step = None
        return obs, reward, done, info

    def reset(self):
        self._step = 0
        return self._env.reset()


class ActionRepeat:

    def __init__(self, env, amount):
        self._env = env
        self._amount = amount

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self._amount and not done:
          obs, reward, done, info = self._env.step(action)
          total_reward += reward
          current_step += 1
        return obs, total_reward, done, info


class NormalizeActions:

    def __init__(self, env):
        self._env = env
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low),
            np.isfinite(env.action_space.high))
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self._env.step(original)


class ObsDict:

    def __init__(self, env, key='obs'):
        self._env = env
        self._key = key

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        spaces = {self._key: self._env.observation_space}
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return self._env.action_space

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs = {self._key: np.array(obs)}
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs = {self._key: np.array(obs)}
        return obs


class OneHotAction:

    def __init__(self, env):
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def action_space(self):
        shape = (self._env.action_space.n,)
        space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
        space.sample = self._sample_action
        return space

    def step(self, action):
        index = np.argmax(action).astype(int)
        reference = np.zeros_like(action)
        reference[index] = 1
        if not np.allclose(reference, action):
          raise ValueError(f'Invalid one-hot action:\n{action}')
        return self._env.step(index)

    def reset(self):
        return self._env.reset()

    def _sample_action(self):
        actions = self._env.action_space.n
        index = self._random.randint(0, actions)
        reference = np.zeros(actions, dtype=np.float32)
        reference[index] = 1.0
        return reference


class RewardObs:

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def observation_space(self):
        spaces = self._env.observation_space.spaces
        assert 'reward' not in spaces
        spaces['reward'] = gym.spaces.Box(-np.inf, np.inf, dtype=np.float32)
        return gym.spaces.Dict(spaces)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        obs['reward'] = reward
        return obs, reward, done, info

    def reset(self):
        obs = self._env.reset()
        obs['reward'] = 0.0
        return obs


class ResizeImage:

    def __init__(self, env, size=(64, 64)):
        self._env = env
        self._size = size
        self._keys = [
            k for k, v in env.obs_space.items()
            if len(v.shape) > 1 and v.shape[:2] != size]
        print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
        if self._keys:
          from PIL import Image
          self._Image = Image

    def __getattr__(self, name):
        if name.startswith('__'):
          raise AttributeError(name)
        try:
          return getattr(self._env, name)
        except AttributeError:
          raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        for key in self._keys:
          shape = self._size + spaces[key].shape[2:]
          spaces[key] = gym.spaces.Box(0, 255, shape, np.uint8)
        return spaces

    def step(self, action):
        obs = self._env.step(action)
        for key in self._keys:
          obs[key] = self._resize(obs[key])
        return obs

    def reset(self):
        obs = self._env.reset()
        for key in self._keys:
          obs[key] = self._resize(obs[key])
        return obs

    def _resize(self, image):
        image = self._Image.fromarray(image)
        image = image.resize(self._size, self._Image.NEAREST)
        image = np.array(image)
        return image


class RenderImage:

    def __init__(self, env, key='image'):
        self._env = env
        self._key = key
        self._shape = self._env.render().shape

    def __getattr__(self, name):
        if name.startswith('__'):
          raise AttributeError(name)
        try:
          return getattr(self._env, name)
        except AttributeError:
          raise ValueError(name)

    @property
    def obs_space(self):
        spaces = self._env.obs_space
        spaces[self._key] = gym.spaces.Box(0, 255, self._shape, np.uint8)
        return spaces

    def step(self, action):
        obs = self._env.step(action)
        obs[self._key] = self._env.render('rgb_array')
        return obs

    def reset(self):
        obs = self._env.reset()
        obs[self._key] = self._env.render('rgb_array')
        return obs

class RandomVideoSource:
    def __init__(self, env, video_dir, total_frames=None, grayscale=False):
        """
        Args:
            filelist: a list of video files
        """
        self._env = env
        self.grayscale = grayscale
        self.total_frames = total_frames
        self.shape = self._env.render().shape
        self.filelist = [os.path.join(video_dir, file) for file in os.listdir(video_dir)]
        self.build_arr()
        self.current_idx = 0
        self.reset()

    def build_arr(self):
        if not self.total_frames:
            self.total_frames = 0
            self.arr = None
            random.shuffle(self.filelist)
            for fname in tqdm.tqdm(self.filelist, desc="Loading videos for natural", position=0):
                if self.grayscale: frames = skvideo.io.vread(fname, outputdict={"-pix_fmt": "gray"})
                else:              frames = skvideo.io.vread(fname)
                local_arr = np.zeros((frames.shape[0], self.shape[0], self.shape[1]) + ((3,) if not self.grayscale else (1,)))
                for i in tqdm.tqdm(range(frames.shape[0]), desc="video frames", position=1):
                    local_arr[i] = cv2.resize(frames[i], (self.shape[1], self.shape[0])) ## THIS IS NOT A BUG! cv2 uses (width, height)
                if self.arr is None:
                    self.arr = local_arr
                else:
                    self.arr = np.concatenate([self.arr, local_arr], 0)
                self.total_frames += local_arr.shape[0]
        else:
            self.arr = np.zeros((self.total_frames, self.shape[0], self.shape[1]) + ((3,) if not self.grayscale else (1,)))
            total_frame_i = 0
            file_i = 0
            with tqdm.tqdm(total=self.total_frames, desc="Loading videos for natural") as pbar:
                while total_frame_i < self.total_frames:
                    if file_i % len(self.filelist) == 0: random.shuffle(self.filelist)
                    file_i += 1
                    fname = self.filelist[file_i % len(self.filelist)]
                    if self.grayscale: frames = skvideo.io.vread(fname, outputdict={"-pix_fmt": "gray"})
                    else:              frames = skvideo.io.vread(fname)
                    for frame_i in range(frames.shape[0]):
                        if total_frame_i >= self.total_frames: break
                        if self.grayscale:
                            self.arr[total_frame_i] = cv2.resize(frames[frame_i], (self.shape[1], self.shape[0]))[..., None] ## THIS IS NOT A BUG! cv2 uses (width, height)
                        else:
                            self.arr[total_frame_i] = cv2.resize(frames[frame_i], (self.shape[1], self.shape[0])) 
                        pbar.update(1)
                        total_frame_i += 1


    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        img = self.get_obs(obs['image'])
        obs['image'] = img
        return obs, reward, done, info

    def reset(self):
        self._loc = np.random.randint(0, self.total_frames)
        obs = self._env.reset()
        img = self.get_obs(obs['image'])
        obs['image'] = img
        return obs

    def get_obs(self, img):
        img = img.transpose(1, 2, 0)
        mask = np.logical_and((img[:, :, 2] > img[:, :, 1]), (img[:, :, 2] > img[:, :, 0]))  # hardcoded for dmc
        bg = self.get_image()
        img[mask] = bg[mask]
        img = img.transpose(2, 0, 1).copy()
        # CHW to HWC for tensorflow
        return img

    def get_image(self):
        img = self.arr[self._loc % self.total_frames]
        self._loc += 1
        return img

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

class FrameStack: # TODO 加了FrameStack后，DeepMindControl的observation_space会变成Box(0, 1, (3, 64, 64), uint8)。但是DeepMindControl的observation_space是Dict，所以会报错
    def __init__(self, env, k):
        self._env = env
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space['image'].shape # origin: shp = env.observation_space.shape, NoneType Object
        spaces = self.observation_space
        spaces['image'] = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space['image'].dtype) # origin: dtype=env.observation_space.dtype
        self.observation_space = gym.spaces.Dict(spaces)
        # self._max_episode_steps = env._max_episode_steps # TODO AttributeError: 'DeepMindControl' object has no attribute '_max_episode_steps'. Not used

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        obs = self._env.reset()
        for _ in range(self._k):
            self._frames.append(obs['image']) # origin: self._frames.append(obs) # ValueError in self._get_obs np.concatenate: zero-dimensional arrays cannot be concatenated
        obs['image'] = self._get_obs()
        return obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self._frames.append(obs['image']) # origin: self._frames.append(obs) # ValueError in self._get_obs np.concatenate: zero-dimensional arrays cannot be concatenated
        obs['image'] = self._get_obs()
        return obs, reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)
