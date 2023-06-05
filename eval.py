import utils.util as util
import utils.data_augs as rad 

import torch
import time
import numpy as np

def make_eval(type):
    if 'rad' or 'curl' or 'drq' in type:
        type = 'rad'
    else:
        type = 'bisim'
    evals = {'bisim': evaluate_bisim, 'rad': evaluate_rad}
    return evals[type]

def evaluate_bisim(env, agent, video, num_episodes, L, step, device=None, embed_viz_dir=None, args=None):
    # embedding visualization
    obses = []
    values = []
    embeddings = []

    for i in range(num_episodes):
        obs = env.reset()
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        while not done:
            with util.eval_mode(agent):
                action = agent.select_action(obs)

            if embed_viz_dir:
                obses.append(obs)
                with torch.no_grad():
                    values.append(min(agent.critic(torch.Tensor(obs).to(device).unsqueeze(0), torch.Tensor(action).to(device).unsqueeze(0))).item())
                    embeddings.append(agent.critic.encoder(torch.Tensor(obs).unsqueeze(0).to(device)).cpu().detach().numpy())

            obs, reward, done, _ = env.step(action)
            video.record(env)
            episode_reward += reward

        video.save('%d.mp4' % step)
        L.log('eval/episode_reward', episode_reward, step)
    L.dump(step)

def evaluate_rad(env, agent, video, num_episodes, L, step, args):
    all_ep_rewards = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            #TODO: The following variables are not used!
            # save_name = args.image_dir + '/step_' + str(step) + '_eps_' + str(i) + '.pt'
            state_obs = []
            pixel_obs = []
            if 'pixel' in args.encoder_type:
                #TODO: When to add qpos?
                # obs, qpos = env.reset()
                obs = env.reset()
            else:
                obs =  env.reset()    

            video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            while not done:
                # center crop image
                if 'pixel' in args.encoder_type and 'crop' in args.data_augs:
                    obs = util.center_crop_image(obs,args.image_size)
                if 'pixel' in args.encoder_type and 'translate' in args.data_augs:
                    obs = util.center_crop_image(obs, args.pre_transform_image_size)
                    obs = np.expand_dims(obs,0)
                    obs = rad.center_translate(obs,args.image_size)
                    obs = np.squeeze(obs,0)
                if 'pixel' in args.encoder_type and 'window' in args.data_augs:
                    obs = np.expand_dims(obs,0)
                    obs = rad.center_window(obs,args.image_size)
                    obs = np.squeeze(obs,0)
                with util.eval_mode(agent):
                    if sample_stochastically:
                        if 'pixel' in args.encoder_type:
                            #TODO: When to normalize the obs?
                            # action = agent.sample_action(obs / 255.)
                            action = agent.sample_action(obs)
                        else:
                            action = agent.sample_action(obs)
                    else:
                        if 'pixel' in args.encoder_type:
                            #TODO: When to normalize the obs?
                            # action = agent.select_action(obs / 255.)
                            action = agent.sample_action(obs)
                        else:
                            action = agent.select_action(obs)
                obs, reward, done, _ = env.step(action)
                video.record(env)
                episode_reward += reward
       
            video.save('%d.mp4' % step)
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)
        
        L.log('eval/' + prefix + 'eval_time', time.time()-start_time , step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        std_ep_reward = np.std(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

        filename = args.work_dir + '/' + args.domain_name + '--'+args.task_name + '-' + args.data_augs + '--s' + str(args.seed) + '--eval_scores.npy'
        key = args.domain_name + '-' + args.task_name + '-' + args.data_augs
        try:
            log_data = np.load(filename,allow_pickle=True)
            log_data = log_data.item()
        except:
            log_data = {}
            
        if key not in log_data:
            log_data[key] = {}

        log_data[key][step] = {}
        log_data[key][step]['step'] = step 
        log_data[key][step]['mean_ep_reward'] = mean_ep_reward 
        log_data[key][step]['max_ep_reward'] = best_ep_reward 
        log_data[key][step]['std_ep_reward'] = std_ep_reward 
        log_data[key][step]['env_step'] = step * args.action_repeat

        np.save(filename,log_data)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)

