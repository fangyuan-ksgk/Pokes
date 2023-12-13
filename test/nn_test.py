import sys
base_dir = '/Users/fangyuanyu/Implementation/Agent/Pokes'
sys.path.append(base_dir)

import os, glob
from increment import GroundEnv
from pathlib import Path

sess_path = Path(f'{base_dir}/logs/session_play')
ep_length = 2024*10
headless = True
init_state_folder = f'{base_dir}/logs/session_trainer/lack_of_health_states'
init_states = glob.glob(init_state_folder+'/*.state')
state_file = init_states[0]

env_config = {
                'headless': headless, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': state_file, 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': f'{base_dir}/game/PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 'extra_buttons': False,
            }

env = GroundEnv(env_config)

######################################

import torch.nn as nn

class CustomPolicy(nn.Module):
    def __init__(self, observation_space, action_space):
        super(CustomPolicy, self).__init__()
        # Define your custom network architecture
        # ...

    def forward(self, observation):
        # Process the observation through your network
        # ...
        return logits, value

# Create the model using your custom policy
from stable_baselines3 import A2C, PPO
model = PPO(CustomPolicy, env, verbose=1)