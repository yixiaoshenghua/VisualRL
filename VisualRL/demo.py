# ------------------------------- train.py -------------------------------------

import os
import argparse
import torch
from agent import Agent
from VisualRL.envs import DMC, wrapper
from logger import Logger
from VisualRL.eval import evaluate

def get_args():
    return args

def make_env(args):
    env = DMC(args)
    env = wrapper(env)
    return env

def make_agent(args):
    agent = Agent(args)
    return agent

def make_log(args):
    L = Logger(args)
    return L

def main():
    args = get_args()
    # base settings
    os.mkdir()
    device = ...
    ...

    # base definition
    train_env = make_env(args)
    test_env = make_env(args)
    agent = make_agent(args, device)
    if args.load_agent:
        agent.load(args.save_path)
    L = make_log(args) # only main.py has the access to log something into file

    # training loop
    obs = train_env.reset()
    episode_return = 0
    for step in range(args.total_steps):
        
        for i in range(args.update_nums): 
            # because we need to update the model for many times during one env step, 
            # we should return the log and merge them every updating in main loop
            train_log_dict = agent.update()

        env_obs = torch.tensor(obs['image'], dtype=torch.float32).to(device)
        action = agent.sample_action(env_obs)
        env_action = action['action'].squeeze(0).detach().cpu().numpy()
        next_obs, reward, done, info = train_env.step(env_action)
        episode_return += reward
        L.log(..., step)


        agent.data_buffer.add(obs, action, next_obs, reward, done)
        obs = next_obs

        if done:
            obs = train_env.reset()
            episode_return = 0
            L.log(..., step)


        if (step+1) % args.eval_freq == 0:
            test_log_dict = evaluate(args, test_env, agent)
            L.log(..., step)



if __name__ == '__main__':
    main()


# -------------------------------- envs.py -------------------------------------

# different environments

# wrapper



# ------------------------------- agent.py -------------------------------------
# all functions should return dict
from replay_buffer import make_buffer
from models import ...

class DrQAgent(BaseAgent):
    def __init__(self, args, device) -> None:
        self.args = args
        self.data_buffer = make_buffer(args, device)
        self.device = device

    def _build_model(self) -> None:
        # sub model
        self.encoder = Encoder(
            obs_shape = self.args.image_size,
            detach = self.args.agent == 'drq', # example, or we can write as: self.args.detach, or: self.args.agent in ['drq', 'curl', 'rad'] 
        )
        self.actor = ...
        self.critic = ...
        # optimizer
        self.encoder_opt = ...
        return

    def sample_action(self, obs) -> dict:
        '''
        params:
            obs: torch.ndarray ()
        return:
            action: dict
                {
                    'action': ,
                    'log_prob': ,
                    'entropy: ,
                }

        '''
        return action
    
    def select_action(self, obs) -> dict:
        '''
        params:
            obs: torch.ndarray ()
        return:
            action: dict
                {
                    'action': ,
                    'log_prob': ,
                    'entropy: ,
                }

        '''
        return action
    
    def update_submodel1(self, data) -> dict:
        # forward
        
        # optimize

        # save into log_dict

        return log_dict
    
    def update(self, ) -> dict:
        data = self.data_buffer.sample()
        submodel1_log_dict = self.update_submodel1(data)
        submodel2_log_dict = self.update_submodel2(data)
        ...

        # merge log_dict
        log_dict = ...
        return log_dict

    def save(self, ) -> None:
        # save model and optimizer
        ...

    def load(self, ) -> None:
        # load model and optimizer
        ...

    def save_data(self, steps):
        for i in range(steps):
            # collect data and save in args.save_data_path
            ...

# -------------------------------- logger.py -------------------------------------

class Logger:
    def __init__(self, args) -> None:
        '''
        -| exp1
            -| videos/*.mp4
            -| figure/*.png
            -| tb/tf_event1...
            -| ckpt/*.pt
            -| args.json
            -| log.csv
        '''
        self.args = args
        self.json_file = ...
        self.csv_file = ...
        self.tb = ...
        self.video_dir = ...
        self.image_dir = ...

    def log_scalar(self, ):
        self.tb.add_scalar()
        self.csv_file.update()

    def log_video(self, ):
        ...

    def log_image(self, ):
        ...


# -------------------------------- replay_buffer.py -------------------------------------
import numpy as np
class ReplayBuffer:
    def __init__(self, args, device) -> None:
        self.args = args
        self.save = args.save_buffer
        self.device = device
        # data buffer
        self.obses = np.empty() # dict()
        self.actions = np.empty()
        self.next_obses = np.empty() # dict()
        self.rewards = np.empty()
        self.dones = np.empty()
        self.idx = 0
        self.capacity = args.replay_buffer_capacity
        self.full = False

    def sample(self) -> dict:
        '''
        return:
            data: dict
            {
                'obs': dict or torch.tensor,
                'action': dict or torch.tensor,
                'next_obs': dict or torch.tensor,
                'reward': torch.tensor,
                'done': torch.tensor,
            }
        
        '''
        data = {}
        # sample a batch of data

        # convert to torch tensor

        return data
    
    def add(self, obs, action, next_obs, reward, done) -> None:
        '''
        params:
            dict or numpy.ndarray
        
        '''
        size = reward.shape[0]
        self.obses[self.idx:self.idx+size] = obs
        ...
        
        if self.idx + size >= self.capacity:
            self.full = True
        self.idx = (self.idx + size) % self.capacity