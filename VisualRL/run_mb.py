import os
import random
import time
import argparse
import numpy as np
import json
import torch
import tqdm

from collections import OrderedDict
import envs

from utils.logger import Logger
from utils.video import VideoRecorder
import utils

from agent.model_based.dreamer_agent import AgentDreamer
from agent.model_based.tia_agent import AgentTIA

from eval import make_eval

os.environ['MUJOCO_GL'] = 'egl' # glfw, egl, osmesa

def get_args():
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--env', type=str, default='walker-walk', help='Control Suite environment')
    parser.add_argument('--agent', type=str, default='Dreamerv1', choices=['Dreamerv1', 'Dreamerv2'], help='choosing algorithm')
    parser.add_argument('--exp-name', type=str, default='None', help='name of experiment for logging')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help="GPU id used")

    # Restore parameters
    parser.add_argument('--restore-data', action='store_true', default=False, help='Restores data buffer')
    parser.add_argument('--restore-data-path', type=str, default='', help='Restore data buffer path')

    parser.add_argument('--restore-checkpoint', action='store_true', default=False, help='Restore model from checkpoint')
    parser.add_argument('--restore-checkpoint-path', type=str, default='', help='Restore checkpoint path')

    # Env parameters
    parser.add_argument('--camera-id', type=int, default=1, help='Camera id')
    parser.add_argument('--frame-stack', type=int, default=3, help='Number of frames to stack')
    parser.add_argument('--time-limit', type=int, default=1000, help='time limit') # Environment TimeLimit
    parser.add_argument('--action-repeat', type=int, default=2, help='Action repeat')

    # Data parameters
    parser.add_argument('--max-episode-length', type=int, default=1000, help='Max episode length')
    parser.add_argument('--buffer-size', type=int, default=1000000, help='Experience replay size')  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
    parser.add_argument('--batch-size', type=int, default=50, help='batch size')
    parser.add_argument('--train-seq-len', type=int, default=50, help='sequence length for training world model')

    # Models parameters
    parser.add_argument('--cnn-activation-function', type=str, default='relu', help='Model activation function for a convolution layer')
    parser.add_argument('--dense-activation-function', type=str, default='elu', help='Model activation function a dense layer')
    parser.add_argument('--obs-embed-size', type=int, default=1024, help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
    parser.add_argument('--num-units', type=int, default=400, help='num hidden units for reward/value/discount models')
    parser.add_argument('--hidden-size', type=int, default=200, help='GRU hidden size size')
    parser.add_argument('--deter-size', type=int, default=200, help='GRU deterministic belief size')
    parser.add_argument('--stoch-size', type=int, default=30, help='Stochastic State/latent size')
    parser.add_argument('--discrete', type=int, default=32, help='Discrete size')
    
    # Actor Exploration Parameters
    parser.add_argument('--actor-dist', type=str, default='tanh_normal', choices=['tanh_normal', 'trunc_normal'], help='The action distribution')
    parser.add_argument('--actor-grad', type=str, default='auto', choices=['dynamics', 'reinforce', 'both', 'auto'], help='The strategy of policy update')
    parser.add_argument('--actor-grad-mix', type=float, default=0.1, help='Actor update mixing rate')
    parser.add_argument('--actor-ent', type=float, default=1e-4, help='Action entropy scale')
    parser.add_argument('--action-noise', type=float, default=0.3, help='Action noise')
    parser.add_argument('--actor-min-std', type=float, default=1e-4, help='Action min std')
    parser.add_argument('--actor-init-std', type=float, default=5, help='Action init std')
    
    # Training parameters
    parser.add_argument('--total-steps', type=int, default=int(1e6), help='total number of training steps')
    parser.add_argument('--init-steps', type=int, default=5000, help='seed episodes')
    parser.add_argument('--update-steps', type=int, default=100, help='num of train update steps per iter')
    parser.add_argument('--num-reward-opt-iters', type=int, default=10, help='num of reward opt steps per iter; used for TIA')
    parser.add_argument('--collect-steps', type=int, default=1000, help='actor collect steps per 1 train iter')
    parser.add_argument('--imagine-horizon', type=int, default=15, help='Latent imagination horizon')
    parser.add_argument('--use-disc-model', action='store_true', default=False, help='whether to use discount model' )
    
    # Coeffecients and constants
    parser.add_argument('--free-nats', type=float, default=3, help='free nats')
    parser.add_argument('--discount', type=float, default=0.99, help='discount factor for actor critic')
    parser.add_argument('--td-lambda', type=float, default=0.95, help='discount rate to compute return')
    parser.add_argument('--kl-loss-coeff', type=float, default=1.0, help='weightage for kl_loss of model')
    parser.add_argument('--kl-alpha', type=float, default=0.8, help='kl balancing weight; used for Dreamerv2')
    parser.add_argument('--disc-loss-coeff', type=float, default=10.0, help='weightage of discount model')
    parser.add_argument('--disen-kl-loss-coeff', type=float, default=1.0, help='weightage of disentangled kl loss; used for TIA')
    parser.add_argument('--disen-rec-loss-coeff', type=float, default=1.0, help='weightage of disentangled reconstruction loss; used for TIA')
    parser.add_argument('--disen-neg-rew-loss-coeff', type=float, default=1.0, help='weightage of disentangled negative reward loss; used for TIA')
    
    # Optimizer Parameters
    parser.add_argument('--model-learning-rate', type=float, default=6e-4, help='World Model Learning rate') 
    parser.add_argument('--actor-learning-rate', type=float, default=8e-5, help='Actor Learning rate') 
    parser.add_argument('--value-learning-rate', type=float, default=8e-5, help='Value Model Learning rate')
    parser.add_argument('--disen-reward-learning-rate', type=float, default=8e-5, help='Disentangled Reward Model Learning rate; used for TIA')
    parser.add_argument('--adam-epsilon', type=float, default=1e-7, help='Adam optimizer epsilon value') 
    parser.add_argument('--grad-clip-norm', type=float, default=100.0, help='Gradient clipping norm')
    parser.add_argument('--slow-target', default=False, action='store_true', help='whether to use slow target value model')
    parser.add_argument('--slow-target-update', type=int, default=100, help='Slow target value model update interval')
    parser.add_argument('--slow-target-fraction', type=float, default=1.0, help='The fraction of EMA update')
    
    # Eval parameters
    parser.add_argument('--test', action='store_true', default=False, help='Test only')
    parser.add_argument('--test-interval', type=int, default=10000, help='Test interval (episodes)')
    parser.add_argument('--eval-freq', type=int, default=100, help='Test interval (episodes)')
    parser.add_argument('--test-episodes', type=int, default=10, help='Number of test episodes')
    
    # saving and checkpoint parameters
    parser.add_argument('--scalar-freq', type=int, default=1e3, help='scalar logging freq')
    parser.add_argument('--log-video-freq', type=int, default=-1, help='video logging frequency')
    parser.add_argument('--max-videos-to-save', type=int, default=2, help='max_videos for saving')

    parser.add_argument('--save-checkpoint', action='store_true', default=False, help='save model from checkpoint')
    parser.add_argument('--save-checkpoint-interval', type=int, default=100000, help='Checkpoint interval (steps)')
    parser.add_argument('--save-checkpoint-path', type=str, default='', help='save checkpoint path')

    parser.add_argument('--save-data', action='store_true', default=False, help='save data buffer')
    parser.add_argument('--save-data-interval', type=int, default=100000, help='data buffer interval (steps)')
    parser.add_argument('--save-data-path', type=str, default='', help='save data buffer path')
    
    parser.add_argument('--render', action='store_true', default=False, help='Render environment')

    args = parser.parse_args()
    
    return args

def make_agent(obs_shape, action_shape, args, device, action_range, image_channel=3):
    if args.agent == 'Dreamerv1' or args.agent == 'Dreamerv2':
        agent = AgentDreamer(args, obs_shape, action_shape, device, args.restore_checkpoint)
    elif args.agent == 'TIA':
        agent = AgentTIA(args, obs_shape, action_shape, device, args.restore_checkpoint)
    return agent

class TMP_Logger:
    def __init__(self):
        pass
    def log_scalars(self, dic, step):
        print(step, dic)
    def log_scalar(self, name, value, step):
        print(name, value, step)

def make_log(args, logdir):
    # return TMP_Logger()
    return Logger(args, logdir)

def make_logdir(args):
    logdir_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logdir/')
    os.makedirs(logdir_root, exist_ok=True)
    logdir = os.path.join(logdir_root, args.env, args.agent, args.exp_name + '_' + time.strftime("%d-%m-%Y-%H-%M-%S"))
    os.makedirs(logdir, exist_ok=False)
    return logdir

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def set_device(args):
    if torch.cuda.is_available() and args.gpu != -1:
        device = torch.device('cuda:{}'.format(args.gpu))
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')
    return device

def save_args(args, logdir):
    with open(os.path.join(logdir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

def main():
    
    # -------------------------------- global settings -------------------------------------
    args = get_args()

    # make logdir
    logdir = make_logdir(args)
    # set seed
    set_seed(args.seed)
    # set device
    device = set_device(args)

    # make train and eval env
    train_env = envs.make_env(args)
    test_env  = envs.make_env(args)
    obs_shape = train_env.observation_space['image'].shape
    action_shape = train_env.action_space.shape

    # make evaluation function
    # evaluate = make_eval(args.agent)

    # make agent
    agent = make_agent(obs_shape, action_shape, args, device, train_env.action_space.high)

    # make logger
    L = make_log(args, logdir) # [TODO] No module named 'caffe2' -> No module named 'tools.setup_helpers'
    
    # make video recorder
    # video = VideoRecorder(logdir if args.save_video else None)
    
    # save args
    save_args(args, logdir)

    # ---------------------------------- training loop ----------------------------------------
    # init set
    obs = train_env.reset()
    agent.reset()
    done = False
    episode = 0
    episode_return = 0
    episode_length = 0

    step = 0

    start_time = time.time()

    for step in range(0, args.total_steps, args.action_repeat):
        if done:
            # log
            episode_log = {
                'train/duration': time.time() - start_time,
                'train/episode': episode,
                'train/episode_length': episode_length,
                'train/episode_return': episode_return,
            }
            L.log_scalars(episode_log, step)

            # reset
            obs = train_env.reset()
            agent.reset()
            done = False
            episode += 1
            episode_return = 0
            episode_length = 0
            start_time = time.time()

        # update
        if step >= args.init_steps:
            if step % args.collect_steps == 0:
                # num_updates = args.init_steps*args.update_steps if step == args.init_steps else args.update_steps
                num_updates = args.update_steps
                for _ in tqdm.tqdm(range(num_updates), desc=str(step)):
                    agent_update_dict = agent.update()
                L.log_scalars(agent_update_dict, step)

        # get action
        if step < args.init_steps:
            action = train_env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs, explore=True)
        next_obs, reward, done, _ = train_env.step(action)
        episode_return += reward

        agent.data_buffer.add(obs, action, reward, next_obs, done, float(done))
        obs = next_obs
        episode_length += args.action_repeat


        # evaluate
        # if (step+1) % args.eval_freq == 0:
        #     # L.log_scalar('eval/episode', episode, step)
        #     evaluate(env, agent, video, args.num_eval_episodes, L, step, args=args) # [TODO] not implemented yet

        # save
        if args.save_checkpoint and ((step+1) % args.save_checkpoint_interval == 0):
            agent.save(os.path.join(logdir, 'checkpoint{}.pt'.format(step)))
        if args.save_data and ((step+1) % args.save_data_interval == 0):
            agent.save_data(os.path.join(logdir, 'buffer{}.pt'.format(step))) # [TODO] not implemented yet



    # elif args.evaluate:
    #     evaluate(test_env, agent, video, args.num_eval_episodes, L, step, args=args)

    # original implementation
    '''
    if args.train:
        initial_logs = OrderedDict()
        seed_episode_rews = agent.collect_random_episodes(train_env, args.seed_steps//args.action_repeat)
        global_step = agent.data_buffer.steps * args.action_repeat

        # without loss of generality intial rews for both train and eval are assumed same
        initial_logs.update({
            'train_avg_reward':np.mean(seed_episode_rews),
            'train_max_reward': np.max(seed_episode_rews),
            'train_min_reward': np.min(seed_episode_rews),
            'train_std_reward':np.std(seed_episode_rews),
            'eval_avg_reward': np.mean(seed_episode_rews),
            'eval_max_reward': np.max(seed_episode_rews),
            'eval_min_reward': np.min(seed_episode_rews),
            'eval_std_reward':np.std(seed_episode_rews),
            })

        L.log_scalars(initial_logs, step=0)
        L.flush()

        while global_step <= args.total_steps:

            print("##################################")
            print(f"At global step {global_step}")

            logs = OrderedDict()

            for _ in range(args.update_steps):
                model_loss, actor_loss, value_loss = agent.update()
    
            train_rews = agent.act_and_collect_data(train_env, args.collect_steps//args.action_repeat)

            # --------------------------------- test and log ------------------------------------------
            logs.update({
                'model_loss' : model_loss,
                'actor_loss': actor_loss,
                'value_loss': value_loss,
                'train_avg_reward':np.mean(train_rews),
                'train_max_reward': np.max(train_rews),
                'train_min_reward': np.min(train_rews),
                'train_std_reward':np.std(train_rews),
            })

            if global_step % args.test_interval == 0:
                episode_rews, video_images = agent.evaluate(test_env, args.test_episodes)

                logs.update({
                    'eval_avg_reward':np.mean(episode_rews),
                    'eval_max_reward': np.max(episode_rews),
                    'eval_min_reward': np.min(episode_rews),
                    'eval_std_reward':np.std(episode_rews),
                })
            
            L.log_scalars(logs, global_step)

            if global_step % args.log_video_freq ==0 and args.log_video_freq != -1 and len(video_images[0])!=0:
                L.log_videos(video_images, global_step, args.max_videos_to_save)

            if global_step % args.checkpoint_interval == 0:
                ckpt_dir = os.path.join(logdir, 'ckpts/')
                if not (os.path.exists(ckpt_dir)):
                    os.makedirs(ckpt_dir)
                agent.save(os.path.join(ckpt_dir,  f'{global_step}_ckpt.pt'))

            global_step = agent.data_buffer.steps * args.action_repeat
            agent.set_step(global_step)
            L.flush()

    elif args.evaluate:
        logs = OrderedDict()
        episode_rews, video_images = agent.evaluate(test_env, args.test_episodes, render=True)

        logs.update({
            'test_avg_reward': np.mean(episode_rews),
            'test_max_reward': np.max(episode_rews),
            'test_min_reward': np.min(episode_rews),
            'test_std_reward': np.std(episode_rews),
        })

        L.dump_scalars_to_pickle(logs, 0, log_title='test_scalars.pkl')
        L.log_videos(video_images, 0, max_videos_to_save=args.max_videos_to_save)
    '''

if __name__ == '__main__':
    main()
