import os
import random
import time
import argparse
import numpy as np
import json
import torch

from collections import OrderedDict
import envs

from utils.logger import MBLogger
from utils.video import VideoRecorder
from utils.replay_buffer import MBReplayBuffer
import utils

from agent.model_based.dreamer_agent import AgentDreamer
from agent.model_based.tia_agent import AgentTIA

from eval import make_eval

os.environ['MUJOCO_GL'] = 'egl' # glfw, egl, osmesa

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='walker-walk', help='Control Suite environment')
    parser.add_argument('--agent', type=str, default='Dreamerv1', choices=['Dreamerv1', 'Dreamerv2'], help='choosing algorithm')
    parser.add_argument('--exp-name', type=str, default='lr1e-3', help='name of experiment for logging')
    parser.add_argument('--train', action='store_true', default=False, help='trains the model')
    parser.add_argument('--evaluate', action='store_true', default=False, help='tests the model')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help="GPU id used")
    # Data parameters
    parser.add_argument('--max-episode-length', type=int, default=1000, help='Max episode length')
    parser.add_argument('--buffer-size', type=int, default=1000000, help='Experience replay size')  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
    parser.add_argument('--time-limit', type=int, default=1000, help='time limit') # Environment TimeLimit
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
    parser.add_argument('--action-repeat', type=int, default=2, help='Action repeat')
    parser.add_argument('--action-noise', type=float, default=0.3, help='Action noise')
    parser.add_argument('--actor-min-std', type=float, default=1e-4, help='Action min std')
    parser.add_argument('--actor-init-std', type=float, default=5, help='Action init std')
    # Training parameters
    parser.add_argument('--total-steps', type=int, default=int(1e6), help='total number of training steps')
    parser.add_argument('--init-steps', type=int, default=5000, help='seed episodes')
    parser.add_argument('--update-steps', type=int, default=100, help='num of train update steps per iter')
    parser.add_argument('--collect-steps', type=int, default=1000, help='actor collect steps per 1 train iter')
    parser.add_argument('--batch-size', type=int, default=50, help='batch size')
    parser.add_argument('--train-seq-len', type=int, default=50, help='sequence length for training world model')
    parser.add_argument('--imagine-horizon', type=int, default=15, help='Latent imagination horizon')
    parser.add_argument('--use-disc-model', action='store_true', default=False, help='whether to use discount model' )
    # Coeffecients and constants
    parser.add_argument('--free-nats', type=float, default=3, help='free nats')
    parser.add_argument('--discount', type=float, default=0.99, help='discount factor for actor critic')
    parser.add_argument('--td-lambda', type=float, default=0.95, help='discount rate to compute return')
    parser.add_argument('--kl-loss-coeff', type=float, default=1.0, help='weightage for kl_loss of model')
    parser.add_argument('--kl-alpha', type=float, default=0.8, help='kl balancing weight; used for Dreamerv2')
    parser.add_argument('--disc-loss-coeff', type=float, default=10.0, help='weightage of discount model')
    
    # Optimizer Parameters
    parser.add_argument('--model_learning-rate', type=float, default=6e-4, help='World Model Learning rate') 
    parser.add_argument('--actor_learning-rate', type=float, default=8e-5, help='Actor Learning rate') 
    parser.add_argument('--value_learning-rate', type=float, default=8e-5, help='Value Model Learning rate')
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
    parser.add_argument('--checkpoint-interval', type=int, default=100000, help='Checkpoint interval (episodes)')
    parser.add_argument('--checkpoint-path', type=str, default='', help='Load model checkpoint')
    parser.add_argument('--restore', action='store_true', default=False, help='restores model from checkpoint')
    parser.add_argument('--experience-replay', type=str, default='', help='Load experience replay')
    parser.add_argument('--render', action='store_true', default=False, help='Render environment')
    parser.add_argument('--save-model', action='store_true', default=False, help='whether to save model')
    parser.add_argument('--save-buffer', action='store_true', default=False, help='whether to save buffer')


    args = parser.parse_args()
    
    return args

def make_log(args, logdir):
    return MBLogger(logdir)

def make_agent(obs_shape, action_shape, args, device, action_range, image_channel=3):
    if args.agent == 'Dreamerv1':
        agent = AgentDreamer(args, obs_shape, action_shape, device, args.restore)
    elif args.agent == 'TIA':
        agent = AgentTIA(args, obs_shape, action_shape, device, args.restore)
    return agent

def main():
    
    # -------------------------------- global settings -------------------------------------
    args = parse_args()

    # make logdir
    logdir_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logdir/')
    os.makedirs(logdir_root, exist_ok=True)
    logdir = os.path.join(logdir_root, args.env, args.agent, args.exp_name + '_' + time.strftime("%d-%m-%Y-%H-%M-%S"))
    os.makedirs(logdir, exist_ok=False)

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # set device
    if torch.cuda.is_available() and args.gpu != -1:
        device = torch.device('cuda:{}'.format(args.gpu))
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')

    # make train and eval env
    train_env = envs.make_env(args)
    test_env  = envs.make_env(args)
    obs_shape = train_env.observation_space['image'].shape
    action_shape = train_env.action_space.shape[0]

    # make evaluation function
    # evaluate = make_eval(args.agent)

    # make agent
    agent = make_agent(obs_shape, action_shape, args, device, train_env.action_space.high)
    # make replay buffer
    replay_buffer = MBReplayBuffer(args.buffer_size, obs_shape, action_shape, seq_len=args.train_seq_len, batch_size=args.batch_size)

    # make logger
    L = make_log(args, logdir) # [TODO] No module named 'caffe2' -> No module named 'tools.setup_helpers'
    # make video recorder
    # video = VideoRecorder(logdir if args.save_video else None)
    # save args
    with open(os.path.join(logdir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # ---------------------------------- start training ----------------------------------------
    if args.train:
        env = train_env
        obs = env.reset()
        agent.reset()
        done = False
        seed_episode_rews = [0.0]
        start_time = time.time()
        episode, episode_reward, done = 0, 0, False
        episode_step = 0
        for step in range(args.total_steps):
            if done:
                # log time
                # L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                # L.dump(step)

                # evaluate agent periodically
                if step % args.eval_freq == 0:
                    # L.log_scalar('eval/episode', episode, step)
                    # evaluate(env, agent, video, args.num_eval_episodes, L, step, args=args) # [TODO] not implemented yet
                    if args.save_model:
                        agent.save(os.path.join(logdir, 'models{}.pt'.format(step)))
                    if args.save_buffer:
                        replay_buffer.save('xxx') # [TODO] not implemented yet
                
                # L.log_scalar('train/episode_reward', episode_reward, step)

                obs = env.reset()
                agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                reward = 0

                # L.log_scalar('train/episode', episode, step)

            # sample action for data collection
            if step < args.init_steps:
                action = env.action_space.sample()
            else:
                with utils.eval_mode(agent):
                    action = agent.sample_action(obs)

            # run training update
            if step >= args.init_steps:
                num_updates = args.init_steps if step == args.init_steps else 1
                for _ in range(num_updates):
                    loss_dict = agent.update(replay_buffer)
                
                # L.update({
                #     **loss_dict,
                #     'train_avg_reward': np.mean(train_rews),
                #     'train_max_reward': np.max(train_rews),
                #     'train_min_reward': np.min(train_rews),
                #     'train_std_reward': np.std(train_rews),
                # })
            
            next_obs, reward, done, _ = env.step(action)

            episode_reward += reward
            replay_buffer.add(obs, action, reward, done)

            obs = next_obs
            episode_step += 1

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
