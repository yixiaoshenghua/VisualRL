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

import utils.util as util
from utils.logger import Logger
from utils.video import VideoRecorder
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
from agent.model_based.dreamer_agent import AgentDreamer
from agent.model_based.tia_agent import AgentTIA

import numpy as np

from eval import make_eval

#TODO: Set the environment variable of OpenGL here
os.environ['MUJOCO_GL'] = 'egl'

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
    parser.add_argument('--config', default='./arguments/sac_ae.yaml', type=str, help='YAML file for configuration')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        args = argparse.Namespace(**config)
    return args

def make_agent(obs_shape, action_shape, args, device, action_range, image_channel=3):
    name = args.agent.lower()
    if name in AGENTS:
        agent = AGENTS[name](args, obs_shape, action_shape, device, args.restore_checkpoint)
    else:
        assert f"Agent {args.agent} is not supported."
    # if name == 'sac_ae':
    #     return AgentSACAE(
    #         args=args,
    #         obs_shape=obs_shape,
    #         action_shape=action_shape,
    #         device=device,
    #         init_temperature=args.init_temperature,
    #         alpha_lr=args.alpha_lr,
    #         alpha_beta=args.alpha_beta,
    #         encoder_lr=args.encoder_lr,
    #         decoder_type=args.decoder_type,
    #         decoder_lr=args.decoder_lr,
    #         decoder_update_freq=args.decoder_update_freq,
    #         decoder_latent_lambda=args.decoder_latent_lambda,
    #         decoder_weight_lambda=args.decoder_weight_lambda,
    #     )
    # elif name == 'rad':
    #     return AgentRad(
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
    #         encoder_feature_dim=args.encoder_feature_dim,
    #         encoder_lr=args.encoder_lr,
    #         encoder_tau=args.encoder_tau,
    #         num_layers=args.num_layers,
    #         num_filters=args.num_filters,
    #         log_interval=args.log_interval,
    #         detach_encoder=args.detach_encoder,
    #         latent_dim=args.latent_dim,
    #         data_augs=args.data_augs,
    #         builtin_encoder=args.builtin_encoder,
    #     )
    # elif name == 'flare': # flare
    #     return AgentFLARE(
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
    #         encoder_feature_dim=args.encoder_feature_dim,
    #         encoder_lr=args.encoder_lr,
    #         encoder_tau=args.encoder_tau,
    #         num_layers=args.num_layers,
    #         num_filters=args.num_filters,
    #         log_interval=args.log_interval,
    #         detach_encoder=args.detach_encoder,
    #         latent_dim=args.latent_dim,
    #         data_augs=args.data_augs,
    #         rank=args.local_rank,
    #         print_param_check=args.print_param_check,
    #         action_range=action_range,
    #         image_channel=image_channel,
    #         builtin_encoder=args.builtin_encoder,
    #     )
    # elif name == 'baseline':
    #     agent = BaselineAgent(
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
    #         encoder_feature_dim=args.encoder_feature_dim,
    #         encoder_lr=args.encoder_lr,
    #         encoder_tau=args.encoder_tau,
    #         encoder_stride=args.encoder_stride,
    #         decoder_type=args.decoder_type,
    #         decoder_lr=args.decoder_lr,
    #         decoder_update_freq=args.decoder_update_freq,
    #         decoder_weight_lambda=args.decoder_weight_lambda,
    #         transition_model_type=args.transition_model_type,
    #         num_layers=args.num_layers,
    #         num_filters=args.num_filters,
    #         builtin_encoder=args.builtin_encoder,
    #     )
    # elif name == 'dbc':
    #     agent = AgentDBC(
    #         obs_shape=obs_shape,
    #         action_shape=action_shape,
    #         action_range=action_range,
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
    #         encoder_feature_dim=args.encoder_feature_dim,
    #         encoder_lr=args.encoder_lr,
    #         encoder_tau=args.encoder_tau,
    #         decoder_lr=args.decoder_lr,
    #         decoder_update_freq=args.decoder_update_freq,
    #         decoder_weight_lambda=args.decoder_weight_lambda,
    #         transition_model_type=args.transition_model_type,
    #         num_layers=args.num_layers,
    #         num_filters=args.num_filters,
    #         bisim_coef=args.bisim_coef,
    #         builtin_encoder=args.builtin_encoder,
    #     )
    # elif name == 'deepmdp':
    #     agent = DeepMDPAgent(
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
    #         encoder_stride=args.encoder_stride,
    #         critic_lr=args.critic_lr,
    #         critic_beta=args.critic_beta,
    #         critic_tau=args.critic_tau,
    #         critic_target_update_freq=args.critic_target_update_freq,
    #         encoder_type=args.encoder_type,
    #         encoder_feature_dim=args.encoder_feature_dim,
    #         encoder_lr=args.encoder_lr,
    #         encoder_tau=args.encoder_tau,
    #         decoder_type=args.decoder_type,
    #         decoder_lr=args.decoder_lr,
    #         decoder_update_freq=args.decoder_update_freq,
    #         decoder_weight_lambda=args.decoder_weight_lambda,
    #         transition_model_type=args.transition_model_type,
    #         num_layers=args.num_layers,
    #         num_filters=args.num_filters,
    #         builtin_encoder=args.builtin_encoder,
    #     )
    # elif name == 'drq':
    #     agent = AgentDrQ(
    #         args=args,
    #         obs_shape=obs_shape, 
    #         action_shape=action_shape, 
    #         device=device,
    #         init_temperature=args.init_temperature,
    #         alpha_lr=args.alpha_lr,
    #         alpha_beta=args.alpha_beta,
    #         action_range=action_range
    #     )
    # elif name == 'curl':
    #     agent = AgentCURL(
    #         args=args,
    #         obs_shape=(3*args.frame_stack, args.image_size, args.image_size),
    #         action_shape=action_shape,
    #         device=device,
    #         init_temperature=args.init_temperature,
    #         alpha_lr=args.alpha_lr,
    #         alpha_beta=args.alpha_beta,
    #         encoder_lr=args.encoder_lr,
    #         cpc_update_freq=1,                      # This argument is not in the "args"
    #         log_interval=args.log_interval,
    #         detach_encoder=args.detach_encoder,
    #         curl_latent_dim=args.curl_latent_dim,
    #         data_augs=args.data_augs
    #     )
    # elif name == 'dribo':
    #     agent = AgentDRIBO(
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
    #         encoder_feature_dim=args.encoder_feature_dim, # 50
    #         stochastic_size=args.stoch_size,
    #         deterministic_size=args.deter_size,
    #         encoder_lr=args.encoder_lr,
    #         encoder_tau=args.encoder_tau,
    #         num_layers=args.num_layers,
    #         num_filters=args.num_filters,
    #         log_interval=args.log_interval,
    #         multi_view_skl=args.multi_view_skl,
    #         mib_batch_size=args.batch_size,
    #         mib_seq_len=args.mib_seq_len,
    #         beta_start_value=args.beta_start_value,
    #         beta_end_value=args.beta_end_value,
    #         grad_clip=args.grad_clip,
    #         kl_balancing=args.kl_balance,
    #         builtin_encoder=args.builtin_encoder,
    #     )
    # elif name == 'dreamerv1' or name == 'dreamerv2':
    #     agent = AgentDreamer(args, obs_shape, action_shape, device, args.restore_checkpoint)
    # elif name == 'tia':
    #     agent = AgentTIA(args, obs_shape, action_shape, device, args.restore_checkpoint)
    # else:
    #     assert 'agent is not supported: %s' % args.agent

    return agent, name

def make_logdir(args):
    logdir_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logdir/')
    os.makedirs(logdir_root, exist_ok=True)

    ts = time.strftime("%m-%d-%H-%M-%S", time.gmtime())
    logdir = os.path.join(logdir_root, args.env, args.agent, f"{args.exp_name}-s-{args.seed}-{ts}")
    os.makedirs(logdir, exist_ok=False)

    video_dir = os.mkdir(os.path.join(logdir, 'video'))
    model_dir = os.mkdir(os.path.join(logdir, 'model'))
    buffer_dir = os.mkdir(os.path.join(logdir, 'buffer'))
    return logdir, video_dir, model_dir, buffer_dir

def make_log(args, logdir):
    return Logger(args, logdir)

def set_device(args):
    if torch.cuda.is_available() and args.gpu != -1:
        device = torch.device('cuda:{}'.format(args.gpu))
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')
    return device

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(1,1000000)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_args(args, logdir):
    with open(os.path.join(logdir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

def main():
    # get arguments
    args = get_args()

    # set seed
    set_seed(args.seed)

    # set device
    device = set_device(args)

    # make directory
    logdir, video_dir, model_dir, buffer_dir = make_logdir(args)

    # make logger
    video = VideoRecorder(video_dir if args.save_video else None)
    L = make_log(args, logdir)

    # make train and eval envs
    train_env = envs.make_env(args)
    test_env = envs.make_env(args)
    obs_shape = train_env.observation_space['image'].shape
    action_shape = train_env.action_space.shape
    action_range = [float(train_env.action_space.low.min()), float(train_env.action_space.high.max())]

    # make agent
    agent, agent_name = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device, 
        action_range=action_range
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
