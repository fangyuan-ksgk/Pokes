
from pathlib import Path
import uuid
from red_gym_env import RedGymEnv
from increment import GroundEnv
import stable_baselines3
from stable_baselines3.common.utils import set_random_seed

def make_env(env_conf, seed=0):
    def _init():
        # env = RedGymEnv(env_conf)
        env = GroundEnv(env_conf)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    sess_path = Path(f'logs/session_play')
    ep_length = 2**23
    import glob
    init_state_folder = 'logs/session_trainer/lack_of_health_states'
    init_states = glob.glob(init_state_folder+'/*.state')
    state_file = init_states[0]

    env_config = {
                'headless': False, 'save_final_state': False, 'early_stop': False,
                'action_freq': 24, 'init_state': state_file, 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': 'game/PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 'extra_buttons': True
            }
    
    env = make_env(env_config)()
    
    obs, info = env.reset()
    while True:
        action = 7 # pass action
        obs, rewards, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            print('Done!')
            break
        env.render()



