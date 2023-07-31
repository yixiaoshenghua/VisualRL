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
import copy

import utils.util as util
from utils.logger import Logger
from utils.video import VideoRecorder
from utils.replay_buffer import CPCReplayBuffer, ReplayBufferFLARE, AugReplayBuffer
import envs

from agent.model_free.sacae_agent import AgentSACAE
from agent.model_free.flare_agent import AgentFLARE
from agent.model_free.curl_agent import AgentCURL
from agent.model_free.rad_agent import AgentRad
from agent.model_free.baseline import BaselineAgent
from agent.model_free.deepmdp import DeepMDPAgent
from agent.model_free.dbc_agent import AgentDBC
from agent.model_free.drq_agent import AgentDrQ
from agent.model_free.dribo_agent import AgentDRIBO

import numpy as np

from eval import make_eval

#TODO: Set the environment variable of OpenGL here
os.environ['MUJOCO_GL'] = 'osmesa'


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    # parser.add_argument('--domain_name', default='cheetah')
    # parser.add_argument('--task_name', default='run')
    parser.add_argument('--env', type=str, default='walker-walk', help='Control Suite environment')
    parser.add_argument('--pre_transform_image_size', default=100, type=int)
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--camera_id', default=0, type=int)
    parser.add_argument('--resource_files', type=str)
    parser.add_argument('--eval_resource_files', type=str)
    parser.add_argument('--img_source', default=None, type=str, choices=['color', 'noise', 'images', 'video', 'none'])
    parser.add_argument('--total_frames', default=1000, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=1000000, type=int)
    # train
    parser.add_argument('--agent', default='sac_ae', type=str)
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--feature_dim', default=50, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--k', default=3, type=int, help='number of steps for inverse model')
    parser.add_argument('--bisim_coef', default=0.5, type=float, help='coefficient for bisim terms')
    parser.add_argument('--load_encoder', default=None, type=str)
    parser.add_argument('--mib_seq_len', default=32, type=int)
    parser.add_argument('--beta_start_value', default=1e-4, type=float)
    parser.add_argument('--beta_end_value', default=1e-3, type=float)
    # eval
    parser.add_argument('--eval_freq', default=10000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder/decoder
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--encoder_stride', default=1, type=int)
    parser.add_argument('--builtin_encoder', default=True, type=bool)
    parser.add_argument('--detach_encoder', default=False, action='store_true')
    parser.add_argument('--decoder_type', default='pixel', type=str)
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_latent_lambda', default=1e-6, type=float)
    parser.add_argument('--decoder_weight_lambda', default=1e-7, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--stochastic_dim', default=30, type=int)
    parser.add_argument('--deterministic_dim', default=200, type=int)
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--multi_view_skl', default=False, action='store_true')
    parser.add_argument('--kl_balance', default=False, action='store_true')
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)
    # data augs
    parser.add_argument('--data_augs', default='crop', type=str)
    parser.add_argument('--augment_target_same_rnd', default=False, action='store_true')
    parser.add_argument('--image_pad', default=4, type=int)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--transition_model_type', default='', type=str, choices=['', 'deterministic', 'probabilistic', 'ensemble'])
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--port', default=2000, type=int)
    # DistributeDataParallel + PyTorch launcher utility.
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--print_param_check', default=False, action='store_true')

    args = parser.parse_args()
    return args



def make_log(args):
    return

def make_agent(obs_shape, action_shape, args, device, action_range, image_channel=3):
    args.agent = args.agent.lower()
    if args.agent == 'sac_ae':
        return AgentSACAE(
            obs_shape=obs_shape,
            action_shape=action_shape,
            action_range=action_range,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_latent_lambda=args.decoder_latent_lambda,
            decoder_weight_lambda=args.decoder_weight_lambda,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            builtin_encoder=args.builtin_encoder,
        )
    elif args.agent == 'rad':
        return AgentRad(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            latent_dim=args.latent_dim,
            data_augs=args.data_augs,
            builtin_encoder=args.builtin_encoder,
        )
    elif args.agent == 'flare': # flare
        return AgentFLARE(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            latent_dim=args.latent_dim,
            data_augs=args.data_augs,
            rank=args.local_rank,
            print_param_check=args.print_param_check,
            action_range=action_range,
            image_channel=image_channel,
            builtin_encoder=args.builtin_encoder,
        )
    elif args.agent == 'baseline':
        agent = BaselineAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            encoder_stride=args.encoder_stride,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            builtin_encoder=args.builtin_encoder,
        )
    elif args.agent == 'dbc':
        agent = AgentDBC(
            obs_shape=obs_shape,
            action_shape=action_shape,
            action_range=action_range,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            bisim_coef=args.bisim_coef,
            builtin_encoder=args.builtin_encoder,
        )
    elif args.agent == 'deepmdp':
        agent = DeepMDPAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            encoder_stride=args.encoder_stride,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            builtin_encoder=args.builtin_encoder,
        )
    elif args.agent == 'drq':
        agent = AgentDrQ(
            obs_shape=obs_shape, 
            action_shape=action_shape, 
            action_range=action_range,
            device=device,
            encoder_feature_dim=args.encoder_feature_dim,
            hidden_dim=args.hidden_dim,
            #hidden_depth=args.hidden_depth,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            actor_update_freq=args.actor_update_freq,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            batch_size=args.batch_size,
            builtin_encoder=args.builtin_encoder,
        )
    elif args.agent == 'curl':
        agent = AgentCURL(
            obs_shape=obs_shape,
            action_shape=action_shape,
            action_range=action_range,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_tau=args.encoder_tau,
            encoder_lr=args.encoder_lr,
            cpc_update_freq=1,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            latent_dim=args.latent_dim,
            data_augs=args.data_augs,
            num_layers=args.num_layers,
            num_filters=args.num_filters
        )
    elif args.agent == 'dribo':
        agent = AgentDRIBO(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.feature_dim, # 50
            stochastic_size=args.stochastic_dim,
            deterministic_size=args.deterministic_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            multi_view_skl=args.multi_view_skl,
            mib_batch_size=args.batch_size,
            mib_seq_len=args.mib_seq_len,
            beta_start_value=args.beta_start_value,
            beta_end_value=args.beta_end_value,
            grad_clip=args.grad_clip,
            kl_balancing=args.kl_balance,
            builtin_encoder=args.builtin_encoder,
        )
    #TODO: Below are some model-based methods
    # elif args.agent == 'planet':
    #     agent = AgentPLANET()
    # elif args.agent == 'dreamer':
    #     agent = AgentDREAMER()
    # elif args.agent == 'tia':
    #     agent = AgentTIA(
    #         obs_shape=obs_shape,
    #         action_shape=action_shape,
    #         device=device,
    #         hidden_dim=args.hidden_dim,
    #         discount=args.discount,
    #         init_temperature=args.init_temperature,
    #         alpha_lr=args.alpha_lr,
    #         alpha_beta=args.alpha_beta,
    #         actor_lr=args.actor_lr,
    #         actor_beta=args.actor_beta,
    #         actor_log_std_min=args.actor_log_std_min,
    #         actor_log_std_max=args.actor_log_std_max,
    #         actor_update_freq=args.actor_update_freq,
    #         critic_lr=args.critic_lr,
    #         critic_beta=args.critic_beta,
    #         critic_tau=args.critic_tau,
    #         critic_target_update_freq=args.critic_target_update_freq,
    #         encoder_type=args.encoder_type,
    #         decoder_type=args.decoder_type,
    #         encoder_feature_dim=args.feature_dim, # 50
    #         stochastic_size=args.stochastic_dim,
    #         deterministic_size=args.deterministic_dim,
    #         encoder_lr=args.encoder_lr,
    #         encoder_tau=args.encoder_tau,
    #         num_layers=args.num_layers,
    #         num_filters=args.num_filters,
    #         num_units=args.num_units,
    #         grad_clip=args.grad_clip,
    #         disen_reward_lr=args.disen_reward_lr,
    #         reward_scale=args.reward_scale,
    #         reward_opt_num=args.reward_opt_num,
    #         free_nuts=args.free_nuts,
    #         kl_scale=args.kl_scale,
    #         disen_kl_scale=args.disen_kl_scale,
    #         disen_neg_rew_scale=args.disen_neg_rew_scale,
    #         disen_rec_scale=args.disen_rec_scale,
    #         disclam=args.disclam,
    #         batch_size=args.batch_size,
    #         seq_len=args.seq_len,
    #         builtin_encoder=args.builtin_encoder,
    #     )
    else:
        assert 'agent is not supported: %s' % args.agent

    return agent

def make_replay_buffer(args, env, obs_shape, pre_aug_obs_shape, device):
    if args.agent.lower() == 'curl':
        replay_buffer = CPCReplayBuffer(
            obs_shape=pre_aug_obs_shape,
            action_shape=env.action_space.shape,
            capacity=args.replay_buffer_capacity,
            batch_size=args.batch_size,
            device=device
        )
    elif args.agent.lower() == 'sac_ae' or args.agent.lower() == 'dbc':
        replay_buffer = CPCReplayBuffer(
            obs_shape=obs_shape,
            action_shape=env.action_space.shape,
            capacity=args.replay_buffer_capacity,
            batch_size=args.batch_size,
            device=device
        )
    elif args.agent.lower() == 'drq':
        replay_buffer = AugReplayBuffer(
            obs_shape=obs_shape,
            action_shape=env.action_space.shape,
            capacity=args.replay_buffer_capacity,
            image_pad=args.image_pad,
            device=device
        )
    return replay_buffer

def main():
    args = parse_args()
    if args.seed == -1: 
        args.__dict__["seed"] = np.random.randint(1,1000000)
    util.set_seed_everywhere(args.seed)
    evaluate = make_eval(args.agent)

    train_env = envs.make_env(args)
    test_env = envs.make_env(args)
    action_range = [float(train_env.action_space.low.min()), float(train_env.action_space.high.max())]

    # make directory
    ts = time.gmtime() 
    ts = time.strftime("%m-%d-%H-%M", ts)    
    env_name = args.domain_name + '-' + args.task_name
    exp_name = args.agent + '-' + env_name + '-s' + str(args.seed)  + '-' + ts + '-im' + str(args.image_size) +'-b'  \
    + str(args.batch_size)
    args.work_dir = args.work_dir + '/'  + exp_name

    util.make_dir(args.work_dir)
    video_dir = util.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = util.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = util.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # the dmc2gym wrapper standardizes actions
    assert train_env.action_space.low.min() >= -1
    assert test_env.action_space.high.max() <= 1

    #TODO: Added a reshape process
    obs_shape = (3*args.frame_stack, args.image_size, args.image_size)
    pre_aug_obs_shape = (3*args.frame_stack, args.pre_transform_image_size, args.pre_transform_image_size)
    
    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=train_env.action_space.shape,
        args=args,
        device=device, 
        action_range=action_range
    )

    replay_buffer = make_replay_buffer(args, train_env, obs_shape, pre_aug_obs_shape, device)

    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    for step in range(args.num_train_steps):
        if done:
            if args.decoder_type == 'inverse':
                for i in range(1, args.k):  # fill k_obs with 0s if episode is done
                    replay_buffer.k_obses[replay_buffer.idx - i] = 0
            if step > 0:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            # evaluate agent periodically
            if step % args.eval_freq == 0:
                L.log('eval/episode', episode, step)
                evaluate(test_env, agent, video, args.num_eval_episodes, L, step, args=args)
                if args.save_model:
                    agent.save(model_dir, step)
                if args.save_buffer:
                    replay_buffer.save(buffer_dir)

            L.log('train/episode_reward', episode_reward, step)

            obs = train_env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            reward = 0

            L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = train_env.action_space.sample()
        else:
            with util.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = args.init_steps if step == args.init_steps else 1
            for _ in range(num_updates):
                loss_dict = agent.update(replay_buffer, L, step)
            #TODO: Update L using loss_dict
            '''
            # Some code here
            '''

        next_obs, reward, done, _ = train_env.step(action)

        # allow infinit bootstrap
        done_bool = 0 if episode_step + 1 == train_env._max_episode_steps else float(
            done
        )
        episode_reward += reward

        #TODO: Some replay buffer needs parameters like this: (obs, action, reward, next_obs, *done,* done_bool)
        replay_buffer.add(obs, action, reward, next_obs, done, done_bool)
        
        # CURL doesn't need this
        # np.copyto(replay_buffer.k_obses[replay_buffer.idx - args.k], next_obs)

        obs = next_obs
        episode_step += 1


if __name__ == '__main__':
    main()
