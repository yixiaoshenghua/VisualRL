import os
import random
import time
import argparse
import numpy as np
import json
import torch

from collections import OrderedDict

import env_wrapper
from dreamer_copy import Dreamer
from bird import Bird
from utils import *

os.environ['MUJOCO_GL'] = 'egl'


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='walker_walk', help='Control Suite environment')
    parser.add_argument('--algo', type=str, default='Dreamerv1', choices=['Dreamerv1', 'Dreamerv2'], help='choosing algorithm')
    parser.add_argument('--exp-name', type=str, default='lr1e-3', help='name of experiment for logging')
    parser.add_argument('--train', action='store_true', help='trains the model')
    parser.add_argument('--evaluate', action='store_true', help='tests the model')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help="GPU id used")
    parser.add_argument('--video-dir', type=str, default='', help='The background videos directory')
    parser.add_argument('--camera-id', type=int, default=0, help='The camera id set for rendering observations')
    parser.add_argument('--change-camera-freq', type=int, default=-1, help='The interval to change the camera id')
    parser.add_argument('--combine-offline-datadir', default=None, type=str, help='The combined offline dataset')
    parser.add_argument('--bird', default=False, action='store_true', help='Bridging Imagination and Reality for Model-Based Deep Reinforcement Learning')
    # Data parameters
    parser.add_argument('--max-episode-length', type=int, default=1000, help='Max episode length')
    parser.add_argument('--buffer-size', type=int, default=1000000, help='Experience replay size')  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
    parser.add_argument('--time-limit', type=int, default=1000, help='time limit') # Environment TimeLimit
    # Models parameter
    parser.add_argument('--adversarial-train', default=False, action='store_true', help='Whether to train the model by GAN')
    parser.add_argument('--use-pretrained-encoder', default=False, action='store_true', help='Whether to use the pretrained encoder')
    parser.add_argument('--transform-action-size', default=0, type=int, help='The size of transformed action')
    parser.add_argument('--rssm-reverse', default=False, action='store_true', help='Whether using reversed observe rollout predition')
    parser.add_argument('--rssm-attention', default=False, action='store_true', help='Whether using future observation with attention machanism')
    parser.add_argument('--cnn-activation-function', type=str, default='elu', help='Model activation function for a convolution layer')
    parser.add_argument('--dense-activation-function', type=str, default='elu', help='Model activation function a dense layer')
    parser.add_argument('--obs-embed-size', type=int, default=1024, help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
    parser.add_argument('--num-units', type=int, default=400, help='num hidden units for reward/value/discount models')
    parser.add_argument('--rnn-type', type=str, default='GRU', help='The type of RNN')
    parser.add_argument('--hidden-size', type=int, default=200, help='GRU hidden size size')
    parser.add_argument('--deter-size', type=int, default=200, help='GRU deterministic belief size')
    parser.add_argument('--stoch-size', type=int, default=30, help='Stochastic State/latent size')
    parser.add_argument('--discrete', type=int, default=0, help='Discrete size')
    # Actor Exploration Parameters
    parser.add_argument('--exploration-decay', type=float, default=0.0, help='The decay weight of exploration')
    parser.add_argument('--exploration-min', type=float, default=0.0, help='The minimum exploration rate')
    parser.add_argument('--actor-dist', type=str, default='tanh_normal', choices=['tanh_normal', 'trunc_normal'], help='The action distribution')
    parser.add_argument('--actor-grad', type=str, default='auto', choices=['dynamics', 'reinforce', 'both', 'auto'], help='The strategy of policy update')
    parser.add_argument('--actor-grad-mix', type=float, default=0.1, help='Actor update mixing rate')
    parser.add_argument('--actor-ent', type=float, default=1e-4, help='Action entropy scale')
    parser.add_argument('--action-repeat', type=int, default=2, help='Action repeat')
    parser.add_argument('--action-noise', type=float, default=0.3, help='Action noise')
    parser.add_argument('--actor-min-std', type=float, default=0.1, help='Action min std')
    parser.add_argument('--actor-init-std', type=float, default=5, help='Action init std')
    # Training parameters
    parser.add_argument('--total-steps', type=int, default=1e6, help='total number of training steps')
    parser.add_argument('--seed-steps', type=int, default=5000, help='seed episodes')
    parser.add_argument('--update-steps', type=int, default=100, help='num of train update steps per iter')
    parser.add_argument('--collect-steps', type=int, default=1000, help='actor collect steps per 1 train iter')
    parser.add_argument('--batch-size', type=int, default=50, help='batch size')
    parser.add_argument('--train-seq-len', type=int, default=50, help='sequence length for training world model')
    parser.add_argument('--imagine-horizon', type=int, default=15, help='Latent imagination horizon')
    parser.add_argument('--use-disc-model', action='store_true', help='whether to use discount model' )
    # Coeffecients and constants
    parser.add_argument('--free-nats', type=float, default=1.0, help='free nats')
    parser.add_argument('--discount', type=float, default=0.99, help='discount factor for actor critic')
    parser.add_argument('--td-lambda', type=float, default=0.95, help='discount rate to compute return')
    parser.add_argument('--div-type', default='kl', type=str, choices=['kl', 'chi2'], help='The divergence type to measure prior and posterior distributions')
    parser.add_argument('--div-balancing', default=False, action='store_true', help='Whether to div balance')
    parser.add_argument('--div-loss-coeff', type=float, default=1.0, help='weightage for kl_loss of model')
    parser.add_argument('--div-alpha', type=float, default=0.8, help='kl balancing weight; used for Dreamerv2')
    parser.add_argument('--disc-loss-coeff', type=float, default=10.0, help='weightage of discount model')
    
    # Optimizer Parameters
    parser.add_argument('--model_learning-rate', type=float, default=3e-4, help='World Model Learning rate') 
    parser.add_argument('--actor_learning-rate', type=float, default=8e-5, help='Actor Learning rate') 
    parser.add_argument('--value_learning-rate', type=float, default=8e-5, help='Value Model Learning rate')
    parser.add_argument('--adam-epsilon', type=float, default=1e-5, help='Adam optimizer epsilon value') 
    parser.add_argument('--adam-weight-decay', type=float, default=1e-6, help='Adam optimizer weight decay')
    parser.add_argument('--grad-clip-norm', type=float, default=100.0, help='Gradient clipping norm')
    parser.add_argument('--slow-target', default=False, action='store_true', help='whether to use slow target value model')
    parser.add_argument('--slow-target-update', type=int, default=100, help='Slow target value model update interval')
    parser.add_argument('--slow-target-fraction', type=float, default=1.0, help='The fraction of EMA update')
    # Eval parameters
    parser.add_argument('--test', action='store_true', help='Test only')
    parser.add_argument('--test-interval', type=int, default=10000, help='Test interval (episodes)')
    parser.add_argument('--test-episodes', type=int, default=10, help='Number of test episodes')
    # saving and checkpoint parameters
    parser.add_argument('--scalar-freq', type=int, default=1e3, help='scalar logging freq')
    parser.add_argument('--log-video-freq', type=int, default=-1, help='video logging frequency')
    parser.add_argument('--max-videos-to-save', type=int, default=2, help='max_videos for saving')
    parser.add_argument('--checkpoint-interval', type=int, default=100000, help='Checkpoint interval (episodes)')
    parser.add_argument('--policy-checkpoint-path', type=str, default='', help='Load policy model checkpoint')
    parser.add_argument('--wm-checkpoint-path', type=str, default='', help='Load world model checkpoint')
    parser.add_argument('--restore', action='store_true', help='restores model from checkpoint')
    parser.add_argument('--experience-replay', type=str, default='', help='Load experience replay')
    parser.add_argument('--render', action='store_true', help='Render environment')


    args = parser.parse_args()
    
    return args


def main():
    
    # -------------------------------- global settings -------------------------------------
    args = get_args()
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logdir/')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = args.env + '/' + args.algo + '/' + args.exp_name + '_' + time.strftime("%Y-%m-%d-%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available() and args.gpu != -1:
        device = torch.device('cuda:{}'.format(args.gpu))
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')

    train_env = env_wrapper.make_env(args)
    test_env  = env_wrapper.make_env(args)
    obs_shape = train_env.observation_space['image'].shape
    action_size = train_env.action_space.shape[0]
    dreamer = Dreamer(args, obs_shape, action_size, device) if not args.bird else Bird(args, obs_shape, action_size, device)

    logger = Logger(logdir)
    with open(os.path.join(logdir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # ---------------------------------- start training ----------------------------------------
    if args.train:
        initial_logs = OrderedDict()
        seed_episode_rews = dreamer.collect_random_episodes(train_env, args.seed_steps//args.action_repeat)
        global_step = dreamer.data_buffer.steps * args.action_repeat

        # without loss of generality intial rews for both train and eval are assumed same
        initial_logs.update({
            'train/avg_reward':np.mean(seed_episode_rews),
            'train/max_reward': np.max(seed_episode_rews),
            'train/min_reward': np.min(seed_episode_rews),
            'train/std_reward':np.std(seed_episode_rews),
            'eval/avg_reward': np.mean(seed_episode_rews),
            'eval/max_reward': np.max(seed_episode_rews),
            'eval/min_reward': np.min(seed_episode_rews),
            'eval/std_reward':np.std(seed_episode_rews),
            })

        logger.log_scalars(initial_logs, step=0)
        logger.flush()

        while global_step <= args.total_steps:

            print("##################################")
            print(f"At global step {global_step}")

            logs = AvgDict()

            for _ in range(args.update_steps):
                train_log_dict = dreamer.update()
                logs.record(train_log_dict)
    
            train_rews = dreamer.act_and_collect_data(train_env, args.collect_steps//args.action_repeat)

            # --------------------------------- test and log ------------------------------------------
            logs.update({
                'train/avg_reward':np.mean(train_rews),
                'train/max_reward': np.max(train_rews),
                'train/min_reward': np.min(train_rews),
                'train/std_reward':np.std(train_rews),
            })

            if global_step % args.test_interval == 0:
                episode_rews, video_images, pred_videos = dreamer.evaluate(test_env, args.test_episodes)
                logger.log_videos(pred_videos, global_step, video_title='model', writer=True)

                logs.update({
                    'eval/avg_reward':np.mean(episode_rews),
                    'eval/max_reward': np.max(episode_rews),
                    'eval/min_reward': np.min(episode_rews),
                    'eval/std_reward':np.std(episode_rews),
                })
            
            logger.log_scalars(logs.dict, global_step)
            

            if global_step % args.log_video_freq == 0 and args.log_video_freq != -1 and len(video_images[0])!=0:
                logger.log_videos(video_images, global_step, args.max_videos_to_save, store=True)

            if global_step % args.checkpoint_interval == 0:
                ckpt_dir = os.path.join(logdir, 'ckpts/')
                if not (os.path.exists(ckpt_dir)):
                    os.makedirs(ckpt_dir)
                dreamer.save(os.path.join(ckpt_dir,  f'{global_step}_policy_ckpt.pt'))
                dreamer.world_model.save(os.path.join(ckpt_dir,  f'{global_step}_wm_ckpt.pt'))

            global_step = dreamer.data_buffer.steps * args.action_repeat
            dreamer.set_step(global_step)
            logger.flush()

    elif args.evaluate:
        logs = OrderedDict()
        episode_rews, video_images = dreamer.evaluate(test_env, args.test_episodes, render=True)

        logs.update({
            'test/avg_reward':np.mean(episode_rews),
            'test/max_reward': np.max(episode_rews),
            'test/min_reward': np.min(episode_rews),
            'test/std_reward':np.std(episode_rews),
        })

        logger.dump_scalars_to_pickle(logs, 0, log_title='test_scalars.pkl')
        logger.log_videos(video_images, 0, max_videos_to_save=args.max_videos_to_save)

if __name__ == '__main__':
    main()
