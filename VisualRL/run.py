import numpy as npz
import torch
import argparse
import os
import math
import gym
import sys
import random
import time
import json
import yaml
import copy
import tqdm

import VisualRL.utils.util as util
from utils.replay_buffer import make_replay_buffer
from utils.util import eval_mode
from utils.logger import Logger
from utils.video import VideoRecorder
import envs as envs

from agent.model_free.sacae_agent import AgentSACAE
from agent.model_free.flare_agent import AgentFLARE
from agent.model_free.curl_agent import AgentCURL
from agent.model_free.rad_agent import AgentRad
from agent.model_free.baseline import BaselineAgent
from agent.model_free.deepmdp import DeepMDPAgent
from agent.model_free.dbc_agent import AgentDBC
from agent.model_free.drq_agent import AgentDrQ
from agent.model_free.dribo_agent import AgentDRIBO
from agent.model_based.dreamer_agent import AgentDreamer
from agent.model_based.tia_agent import AgentTIA

import numpy as np

from VisualRL.eval import make_eval

#TODO: Set the environment variable of OpenGL here
# os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_GL'] = 'glfw'
# os.environ['MUJOCO_GL'] = 'osmesa'


AGENTS = {
    "baseline": BaselineAgent,
    "curl": AgentCURL,
    "dbc": AgentDBC,
    "deepmdp": DeepMDPAgent,
    "dreamerv1": AgentDreamer,
    "dreamerv2": AgentDreamer,
    "dribo": AgentDRIBO,
    "drq": AgentDrQ,
    "flare": AgentFLARE,
    "rad": AgentRad,
    "sac_ae": AgentSACAE,
    "tia": AgentTIA
}

def get_args():
    parser = argparse.ArgumentParser(description='Reproduce of multiple Visual RL algorithms.')
    parser.add_argument('group', type=str, help='Agent name')
    
    agent_name = parser.parse_args().group.lower()
    
    global_config_path = 'VisualRL/config/global.yaml'
    agent_config_path = f'VisualRL/config/agents/{agent_name}.yaml'
    
    
    with open(global_config_path, 'r') as f:
        config = yaml.safe_load(f)
        args = argparse.Namespace(**config)
    
    with open(agent_config_path, 'r') as f:
        agent_config = yaml.safe_load(f)
        args.agent_config = agent_config
        
    args.agent = agent_name
    
    return args

def make_agent(agent_name, config, obs_shape, action_shape, action_range, device, restore_checkpoint, image_channel=3):
    if agent_name in AGENTS:       
        agent = AGENTS[agent_name](
            obs_shape=obs_shape,
            action_shape=action_shape, 
            action_range=action_range, 
            device=device, 
            restore_checkpoint=restore_checkpoint, 
            **config
        )
    else:
        assert f"Agent {agent_name} is not supported."
    return agent

def make_logdir(env, agent, exp_name, seed):
    logdir_root = os.path.join(os.getcwd(), 'logdir')
    ts = time.strftime("%m-%d-%H-%M-%S", time.gmtime())
    logdir = os.path.join(logdir_root, env, agent, f"{exp_name}-s-{seed}-{ts}")
    video_dir = os.path.join(logdir, 'video')
    model_dir = os.path.join(logdir, 'model')
    buffer_dir = os.path.join(logdir, 'buffer')
    
    os.makedirs(logdir_root, exist_ok=True)
    os.makedirs(logdir, exist_ok=False)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(buffer_dir, exist_ok=True)
    
    return logdir, video_dir, model_dir, buffer_dir

def make_log(logdir, save_tb):
    return Logger(logdir, save_tb)

def set_device(gpu):
    assert isinstance(gpu, int) and 0 <= gpu < torch.cuda.device_count(), f'Invalid GPU id: {gpu}'

    if torch.cuda.is_available() and gpu != -1:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')
        
    return device

def set_seed(seed):
    hashseed = os.getenv('PYTHONHASHSEED')
    if not hashseed:
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.execv(sys.executable, [sys.executable] + sys.argv)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


def save_args(args, logdir):
    with open(os.path.join(logdir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

def main():
    # get arguments
    args = get_args()

    # set seed
    if not args.seed:
        args.seed = np.random.randint(1,1000000)
    set_seed(args.seed)

    # set device
    device = set_device(args.gpu)

    # make directory
    logdir, video_dir, model_dir, buffer_dir = make_logdir(args.env, args.agent, args.exp_name, args.seed)
    print(logdir)
    print(video_dir)
    print(model_dir)
    print(buffer_dir)
    # assert 0

    # make logger
    video = VideoRecorder(video_dir if args.save_video else None)
    L = make_log(logdir, args.save_tb)

    # make train and eval envs
    train_env = envs.make_env(args)
    test_env = envs.make_env(args)
    # train_env.seed(args.seed)
    obs_shape = train_env.observation_space['image'].shape
    action_shape = train_env.action_space.shape
    action_range = [float(train_env.action_space.low.min()), float(train_env.action_space.high.max())]

    # make agent
    agent = make_agent(
        args.agent,
        args.agent_config,
        obs_shape,
        action_shape,
        action_range,
        device,
        args.restore_checkpoint
    )

    # save args
    save_args(args, logdir)

    evaluate = make_eval(args.agent)
    # -------------------------------------------------- training loop --------------------------------------------------
    # init settings
    obs = train_env.reset()
    agent.reset()
    episode, episode_length, episode_return, done = 0, 0, 0, True
    start_time = time.time()

    for step in range(0, args.total_steps, args.action_repeat):
        if done:
            # if args.decoder_type == 'inverse':
            #     for i in range(1, args.k):  # fill k_obs with 0s if episode is done
            #         replay_buffer.k_obses[replay_buffer.idx - i] = 0
            # log
            episode_log = {
                'train/episode_duration': time.time() - start_time,
                'train/episode': episode,
                'train/episode_length': episode_length,
                'train/episode_return': episode_return,
            }
            if step % args.log_interval == 0:
                print('****************************************')
                L.log_scalars(episode_log, step)
                print('****************************************')

            # evaluate agent periodically
            # if step % args.eval_freq == 0:
            #     L.log('eval/episode', episode, step)
            #     evaluate(test_env, agent, video, args.num_eval_episodes, L, step, args=args)

            # reset
            obs = train_env.reset()
            agent.reset()
            done = False
            episode_return = 0
            episode_length = 0
            episode += 1
            start_time = time.time()

        # run training update
        if step >= args.init_steps:
            if args.agent_type == 'model_based':
                if step % args.collect_steps == 0:
                    # Here num_updates should be a large num
                    num_updates = args.update_steps
                    for _ in tqdm.tqdm(range(num_updates), desc=str(step)):
                        agent_update_dict = agent.update()
                    L.log_scalars(agent_update_dict, step)
            else:
                # Here num_updates should be a small num
                if agent_name != 'sac_ae':
                    num_updates = args.update_steps
                else:
                    num_updates = args.init_steps if step == args.init_steps else args.update_steps
                for _ in range(num_updates):
                    loss_dict = agent.update(step)
                if step % args.log_interval == 0:
                    print('------------------------------------------')
                    print('step: ', step)
                    L.log_scalars(loss_dict, step)
                #TODO: Update L using loss_dict
                '''
                # Some code here
                '''

        # sample action for data collection
        if step < args.init_steps:
            action = train_env.action_space.sample()
        else:
            with util.eval_mode(agent):
                action = agent.sample_action(obs)   #FIXME: explore=True

        next_obs, reward, done, _ = train_env.step(action)

        # allow infinite bootstrap
        done_bool = 0 if episode_length + args.action_repeat == args.max_episode_length else float(done)
        episode_return += reward

        agent.data_buffer.add(obs, action, reward, next_obs, done_bool)
        
        # CURL doesn't need this
        # np.copyto(replay_buffer.k_obses[replay_buffer.idx - args.k], next_obs)

        obs = next_obs
        episode_length += args.action_repeat

        # save model and replay buffer
        if args.save_checkpoint and ((step+1) % args.save_checkpoint_interval == 0):
            agent.save(os.path.join(model_dir, 'checkpoint{}.pt'.format(step)))
        if args.save_data and ((step+1) % args.save_data_interval == 0):
            agent.save_data(os.path.join(buffer_dir, 'buffer{}.pt'.format(step))) # [TODO] not implemented yet


if __name__ == '__main__':
    main()
