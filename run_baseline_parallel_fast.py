from os.path import exists
from pathlib import Path
import uuid
import glob
import random
from red_gym_env import RedGymEnv
from increment import GroundEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
# port the training onto tensorboard
import torch
from torch.utils.tensorboard import SummaryWriter


def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        # env = RedGymEnv(env_conf)
        env = GroundEnv(env_conf) # Train a GPT-designed reward agent instead
        env.reset(seed=(seed + rank)) # Parallel envs simulates slightly different games
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':


    ep_length = 2048 * 10
    sess_path = Path(f'logs/session_healer_01')
    # Branch from Trainer Session
    init_state_folder = 'logs/session_trainer/lack_of_health_states'
    init_states = glob.glob(init_state_folder+'/*.state')
    
    num_cpu = 4

    def get_env_config(init_state):
        env_config = {
            'headless': True, 'save_final_state': True, 'early_stop': False,
            'action_freq': 24, 'init_state': init_state, 'max_steps': ep_length, 
            'print_rewards': True, 'save_video': True, 'fast_video': True, 'session_path': sess_path,
            'gb_path': 'game/PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 
            'use_screen_explore': True, 'reward_scale': 4, 'extra_buttons': False,
            'explore_weight': 3 # 2.5
        }
        print(env_config)
        return env_config

    # For each CPU, sample a different initial state
    # env = SubprocVecEnv([make_env(i, get_env_config(random.choice(init_states))) for i in range(num_cpu)])
    env = SubprocVecEnv([make_env(i, get_env_config(init_states[0])) for i in range(num_cpu)])

    # Save a checkpoint every episode
    checkpoint_callback = CheckpointCallback(save_freq=ep_length, save_path=sess_path,
                                     name_prefix='poke')
    #env_checker.check_env(env)
    learn_steps = 40
    # put a checkpoint here you want to start from
    # Trainer Policy Checkpoint
    # file_name = '2increment_GPTreward_context_reward_iter2/poke_4587520_steps'
    
    # Complete Empty Policy Checkpoint
    file_name = 'logs/session_healer_01/poke_0_steps'

    if exists(file_name + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO('CnnPolicy', env, verbose=1, n_steps=ep_length // 8, batch_size=128, n_epochs=3, gamma=0.998)
    
    for i in range(learn_steps):
        model.learn(total_timesteps=(ep_length)*num_cpu*1000, callback=checkpoint_callback)

