import argparse


def get_args():
    parser = argparse.ArgumentParser()
    '''
    # General parameters
    '''
    parser.add_argument('--exp_name', default='None', type=str, help='Name of experiment for logging')
    parser.add_argument('--seed', default=1, type=int, help='Random seed')
    parser.add_argument('--gpu', default=0, type=int, help="GPU id used")
    parser.add_argument('--agent', default='sac_ae', type=str, help='Choosing algorithm')
    parser.add_argument('--agent_type', default='model_free', type=str, choices=['model_free', 'model_based'], help='Type of algorithm')
    
    '''
    # Restore parameters
    '''
    parser.add_argument('--restore_data', default=False, action='store_true', help='Restores data buffer')
    parser.add_argument('--restore_data_path', default='', type=str, help='Restore data buffer path')

    parser.add_argument('--restore_checkpoint', default=False, action='store_true', help='Restore model from checkpoint')
    parser.add_argument('--restore_checkpoint-path', default='', type=str, help='Restore checkpoint path')
    
    '''
    # Environment parameters
    '''
    parser.add_argument('--env', default='walker-walk', type=str, help='Control Suite environment')
    parser.add_argument('--pre_transform_image_size', default=100, type=int)
    parser.add_argument('--image_size', default=84, type=int)
    parser.add_argument('--action_repeat', default=1, type=int, help='Action repeat')
    parser.add_argument('--frame_stack', default=3, type=int, help='Number of frames to stack')
    parser.add_argument('--camera_id', default=0, type=int, help='Camera id')
    parser.add_argument('--time_limit', default=1000, type=int, help='Environment TimeLimit')

    '''
    # Data parameters
    '''
    parser.add_argument('--buffer_size', default=1000000, type=int, help='Experience replay size') # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
    parser.add_argument('--max_episode_length', default=1000, type=int, help='Max episode length')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--train_seq_len', default=50, type=int, help='Sequence length for training world model')

    '''
    # Models parameters
    '''
    # Only for model-based methods.
    parser.add_argument('--cnn_activation_function', default='relu', type=str, help='Model activation function for a convolution layer')
    parser.add_argument('--dense_activation_function', default='elu', type=str, help='Model activation function a dense layer')
    parser.add_argument('--obs_embed_size', default=1024, type=int, help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
    parser.add_argument('--num_units', default=400, type=int, help='Num hidden units for reward/value/discount models')
    parser.add_argument('--hidden_size', default=200, type=int, help='GRU hidden size size')
    parser.add_argument('--deter_size', default=200, type=int, help='GRU deterministic belief size')
    parser.add_argument('--stoch_size', default=30, type=int, help='Stochastic State/latent size')
    parser.add_argument('--discrete', default=32, type=int, help='Discrete size')

    '''
    # Actor parameters
    '''
    # Model-free methods
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # Model-based methods
    parser.add_argument('--actor_dist', default='tanh_normal', type=str, choices=['tanh_normal', 'trunc_normal'], help='The action distribution')
    parser.add_argument('--actor_grad', default='auto', type=str, choices=['dynamics', 'reinforce', 'both', 'auto'], help='The strategy of policy update')
    parser.add_argument('--actor_grad_mix', default=0.1, type=float, help='Actor update mixing rate')
    parser.add_argument('--actor_ent', default=1e-4, type=float, help='Action entropy scale')
    parser.add_argument('--action_noise', default=0.3, type=float, help='Action noise')
    parser.add_argument('--actor_min_std', default=1e-4, type=float, help='Action min std')
    parser.add_argument('--actor_init_std', default=5, type=float, help='Action init std')

    '''
    # Critic parameters
    '''
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float, help='Q function EMA tau')
    parser.add_argument('--critic_target_update_freq', default=2, type=int)

    '''
    # Encoder/Decoder parameters
    '''
    parser.add_argument('--encoder_type', default='pixel', type=str)
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_tau', default=0.05, type=float, help='Encoder EMA tau')
    parser.add_argument('--encoder_stride', default=1, type=int)
    parser.add_argument('--builtin_encoder', default=True, type=bool)
    parser.add_argument('--detach_encoder', default=False, action='store_true')

    parser.add_argument('--decoder_type', default='pixel', type=str)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_latent_lambda', default=1e-6, type=float)
    parser.add_argument('--decoder_weight_lambda', default=1e-7, type=float)

    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--curl_latent_dim', default=128, type=int)

    #TODO: What are these?
    parser.add_argument('--multi_view_skl', default=False, action='store_true')
    parser.add_argument('--kl_balance', default=False, action='store_true')
    parser.add_argument('--load_encoder', default=None, type=str)
    parser.add_argument('--mib_seq_len', default=32, type=int)
    parser.add_argument('--beta_start_value', default=1e-4, type=float)
    parser.add_argument('--beta_end_value', default=1e-3, type=float)

    '''
    # Training parameters
    '''
    parser.add_argument('--init_steps', default=1000, type=int, help='Seed episodes')
    parser.add_argument('--total_steps', default=int(1e6), type=int, help='Total number of training steps')
    parser.add_argument('--update_steps', default=1, type=int, help='Num of train update steps per iter; notice the difference between model-free and model-based methods')
    parser.add_argument('--num_reward_opt_iters', default=10, type=int, help='Num of reward opt steps per iter; used only for TIA')
    parser.add_argument('--collect_steps', default=1000, type=int, help='Actor collect steps per 1 train iter')
    parser.add_argument('--imagine_horizon', default=15, type=int, help='Latent imagination horizon')
    parser.add_argument('--use_disc_model', default=False, action='store_true', help='Whether to use discount model' )

    '''
    # Coeffecients and constants
    '''
    parser.add_argument('--free_nats', default=3, type=float, help='Free nats')
    parser.add_argument('--discount', default=0.99, type=float, help='Discount factor for actor critic')
    parser.add_argument('--td_lambda', default=0.95, type=float, help='discount rate to compute return')
    parser.add_argument('--kl_loss_coeff', default=1.0, type=float, help='weightage for kl_loss of model')
    parser.add_argument('--kl_alpha', default=0.8, type=float, help='kl balancing weight; used for Dreamerv2')
    parser.add_argument('--disc_loss_coeff', default=10.0, type=float, help='weightage of discount model')
    parser.add_argument('--disen_kl_loss_coeff', default=1.0, type=float, help='weightage of disentangled kl loss; used for TIA')
    parser.add_argument('--disen_rec_loss_coeff', default=1.0, type=float, help='weightage of disentangled reconstruction loss; used for TIA')
    parser.add_argument('--disen_neg_rew_loss_coeff', default=1.0, type=float, help='weightage of disentangled negative reward loss; used for TIA')
    parser.add_argument('--k', default=3, type=int, help='number of steps for inverse model')
    parser.add_argument('--bisim_coef', default=0.5, type=float, help='coefficient for bisim terms')

    '''
    # Optimizer Parameters
    '''
    parser.add_argument('--actor_lr', default=1e-3, type=float, help='Actor Learning rate; used for model-free methods')
    parser.add_argument('--actor_learning_rate', default=8e-5, type=float, help='Actor Learning rate; used for model-based methods') 

    parser.add_argument('--critic_lr', default=1e-3, type=float, help='Critic Learning rate; used for model-free methods')
    parser.add_argument('--value_learning_rate', default=8e-5, type=float, help='Value Model Learning rate; used for model-based methods')

    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_lr', default=1e-3, type=float)

    parser.add_argument('--model_learning_rate', default=6e-4, type=float, help='World Model Learning rate') 
    parser.add_argument('--disen_reward_learning_rate', default=8e-5, type=float, help='Disentangled Reward Model Learning rate; used for TIA')
    parser.add_argument('--adam_epsilon', default=1e-7, type=float, help='Adam optimizer epsilon value') 
    parser.add_argument('--grad_clip_norm', default=100.0, type=float, help='Gradient clipping norm')
    parser.add_argument('--slow_target', default=False, action='store_true', help='whether to use slow target value model')
    parser.add_argument('--slow_target_update', default=100, type=int, help='Slow target value model update interval')
    parser.add_argument('--slow_target_fraction', default=1.0, type=float, help='The fraction of EMA update')

    '''
    # Eval parameters
    '''
    parser.add_argument('--eval_freq', default=10000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    
    '''
    # SAC parameters
    '''
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--alpha_beta', default=0.5, type=float)

    '''
    # Data augs
    '''
    parser.add_argument('--data_augs', default='crop', type=str)
    parser.add_argument('--augment_target_same_rnd', default=False, action='store_true')
    parser.add_argument('--image_pad', default=4, type=int)

    '''
    # Saving something parameters
    '''
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--log_video_freq', type=int, default=-1, help='video logging frequency')
    parser.add_argument('--max_videos_to_save', type=int, default=2, help='max_videos for saving')

    parser.add_argument('--save_checkpoint', action='store_true', default=False, help='save model from checkpoint')
    parser.add_argument('--save_checkpoint_interval', type=int, default=100000, help='Checkpoint interval (steps)')
    parser.add_argument('--save_checkpoint_path', type=str, default='', help='save checkpoint path')

    parser.add_argument('--save_data', default=False, action='store_true', help='save data buffer')
    parser.add_argument('--save_data_interval', type=int, default=100000, help='data buffer interval (steps)')
    parser.add_argument('--save_data_path', type=str, default='', help='save data buffer path')
    
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--render', default=False, action='store_true', help='Render environment')
    
    # misc
    #TODO: What are these?
    parser.add_argument('--transition_model_type', default='', type=str, choices=['', 'deterministic', 'probabilistic', 'ensemble'])
    parser.add_argument('--port', default=2000, type=int)
    # DistributeDataParallel + PyTorch launcher utility.
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--print_param_check', default=False, action='store_true')
    
    args = parser.parse_args()
    return args
