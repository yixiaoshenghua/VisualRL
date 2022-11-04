import torch
import numpy as np
import torch.nn as nn
import gym
import os
import sys
from collections import deque, namedtuple
import random
from PIL import Image, ImageEnhance

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
