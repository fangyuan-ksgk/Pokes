# Use Checkpoint file to train the 'healing mode' -- recover from damaged status

from pathlib import Path
import pandas as pd
import numpy as np
import mediapy as media
# from red_gym_env import RedGymEnv
from increment import GroundEnv
import glob, os
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.utils import set_random_seed
import collections
from collections import defaultdict
import cv2
import numpy as np

context_dict = {0: 'in_battle', 1: 'need_heal', 2: 'need_progress', 3: 'need_level_up'}

# Pad Image and Add Text to Paadded areas
def pad_img_with_text(img_pix, text='Hello', pad_height=20):
    pad_height = 20
    pad_img = np.ones((img_pix.shape[0]+pad_height, img_pix.shape[1], img_pix.shape[2]), dtype=np.uint8) * 255
    pad_img[pad_height:,:,:] = img_pix
    # Add text to the padded area
    pad_img = pad_img.astype(np.uint8).copy()
    pad_img = cv2.putText(pad_img, text, (0,10), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (255, 0, 0))
    return pad_img

def update_hist_rew(info, reward_history):
    for key, value in info['rewards'].items():
        reward_history[key].append(value)
    reward_history['context'].append(info['context'])
    reward_history['total_reward'].append(info['total_reward'])
    return reward_history

# Function to convert data x-coordinate to pixel x-coordinate
def get_x_pixel(x, current_index, x_window, width, x_scale):
    return int((x - (current_index - x_window)) * x_scale)

def draw_reward_dynamic_plot(reward_history, name, x_window, y_window, max_index=2048*10, width=82, height=82, target_size=(82, 82)):
    
    reward_history_data = reward_history[name]
    current_index = len(reward_history_data) - 1
    current_val = reward_history_data[-1]
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    # Scaling factors
    max_reward = current_val + y_window
    min_reward = current_val - y_window
    x_scale = width / (2 * x_window)
    y_scale = height / (max_reward - min_reward)


    # Function to convert data point to pixel location
    def get_pixel_from_data(x, y, x_origin, y_origin):
        x_pixel = int(x_scale * (x - x_origin))
        y_pixel = height - int(y_scale * (y - min_reward))
        return x_pixel, y_pixel

    # Draw the line plot
    x_origin = max(0, current_index - x_window)
    y_origin = current_val - y_window
    for i in range(len(reward_history_data) - 1):
        x1, y1 = get_pixel_from_data(i, reward_history_data[i], x_origin, y_origin)
        x2, y2 = get_pixel_from_data(i+1, reward_history_data[i+1], x_origin, y_origin)
        if x_origin <= i < x_origin + 2 * x_window:
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Mark the current point
    x_point, y_point = get_pixel_from_data(current_index, current_val, x_origin, y_origin)
    cv2.drawMarker(img, (x_point, y_point), (255, 0, 0), cv2.MARKER_CROSS, markerSize=3, thickness=3)
    
    # Define text properties
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 0.4
    font_color = (0, 255, 0)  # Black color
    thickness = 1
    line_type = 2

    # Define the text
    legend_text = f"{name}"

    # Position of the text (top-right corner)
    text_position = (width - int(800 / 1000 * width), int(150 / 1000 * height))  # Adjust the position as needed

    # Draw the text
    cv2.putText(img, legend_text, text_position, font, font_scale, font_color, thickness, line_type)
    
    

    # Drawing X ticks and labels
    num_ticks = 7
    for i in range(num_ticks):
        x_data = int((current_index - x_window) + i * (2 * x_window / (num_ticks - 1)))
        x_pixel = get_x_pixel(x_data, current_index, x_window, width, x_scale)
        
        if i==0 or i==num_ticks-1:
            continue
        # Draw tick line
        cv2.line(img, (x_pixel, height - int(50 / 1000 * height)), (x_pixel, height), (0, 0, 0), 1)

        # Draw label
        # label = str(x_data)
        # cv2.putText(img, label, (x_pixel - 15 / 1000 * width, height - 20 / 1000 * height), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 5)
    
    # return cv2.resize(img, target_size)
    return img


def visualize_reward_dynamic(img_game, reward_history):
    context = context_dict[reward_history['context'][-1]]
    total_reward = reward_history['total_reward'][-1]
    img_game = pad_img_with_text(img_game, text=f"C: {context} | R: {total_reward}")

    explore_reward = draw_reward_dynamic_plot(reward_history, 'explore', x_window=300, y_window=30)
    if context == 'need_heal':
        heal_reward = draw_reward_dynamic_plot(reward_history, 'heal', x_window=300, y_window=100)
        img_reward = np.concatenate((heal_reward, explore_reward), axis=0)
    else:
        battle_reward = draw_reward_dynamic_plot(reward_history, 'escape_battle', x_window=300, y_window=30)
        img_reward = np.concatenate((battle_reward, explore_reward), axis=0)
    
    img = np.concatenate((img_game, img_reward), axis=1)
    return img

def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = GroundEnv(env_conf)
        #env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


# prepare action list 
def prepare_action_list_from_ckpt(init_state, ckpt_path, sess_path, save_name='healer01'):

    ep_length = 2048 * 10
    env_config = {
                'headless': True, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': init_state, 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 'extra_buttons': False
            }
    
    env = make_env(0, env_config)() #SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])
    env.instance_id = save_name
    model = PPO.load(ckpt_path, env=env, custom_objects={'lr_schedule': 0, 'clip_range': 0})

    obs, info = env.reset()
    action_list = []
    while True:
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, terminated, truncated, info = env.step(action)
        action_list.append(action)
        if terminated or truncated:
            break
    return action_list


def save_video_on_reward_history_dict(d, folder_dir, record_history_length=10, shape=(144,160)):
    step = d['Step']
    vid_path = Path(f'{folder_dir}/record_{step}').with_suffix('.mp4')

    record_history_length = 10
    with media.VideoWriter(vid_path, shape, fps=60) as hist_frame_writer:
        for i in range(record_history_length + 10):
            i = min(i, record_history_length-1)
            frame = d['Frames'][i]
            hist_frame_writer.add_image(frame)
    hist_frame_writer.close()

# Record reward triggering frames
def check_triggered_reward_and_save(env, action_list, reward_name, sess_name, record_history_length=10, save_video=False):
    import tqdm
    trigger_hist = []
    info_queue = []
    frame_queue = []
    last_reward_value = 0.
    for i_step, action in tqdm.tqdm(enumerate(action_list), total=len(action_list), desc='Checking reward'):
        obs, rewards, term, trunc, info = env.step(action)
        info_queue.append(info)
        frame_queue.append(env.render(reduce_res=False, update_mem=False))
        if len(info_queue) > record_history_length:
            info_queue.pop(0)
            frame_queue.pop(0)

        rewards = env.info['rewards']
        if rewards[reward_name] > last_reward_value:
            last_reward_value = rewards[reward_name]
            print('Reward', reward_name,'triggered at step', i_step, 'with reward', rewards[reward_name], 'at context', ct_dict[info['context']])
            trigger_hist.append({'Step': i_step, 'Information': info_queue.copy(), 'Frames': frame_queue.copy()})
            
            if save_video:
                folder_dir = f'{sess_name}/{reward_name}_hist'
                save_video_on_reward_history_dict(trigger_hist[-1], folder_dir, record_history_length=record_history_length, shape=(144,160))
                print('Video Saved')

    return trigger_hist

# BootStrapping Healer to Warrior Mode
def save_state_for_lack_of_health_context(env, action_list, sess_name):
    import os
    folder_path = f'{sess_name}/lack_of_health_states'
    os.makedirs(folder_path, exist_ok=True)
    import tqdm
    interval = 100
    last_step = -interval - 1
    for i_step, action in tqdm.tqdm(enumerate(action_list), total=len(action_list), desc='Checking reward'):
        obs, rewards, term, trunc, info = env.step(action)
        # Not in battle & In battle Lack of Health situation
        in_battle_lack_of_health = env.info['context'] == 0 and (env.info['status']['hp_pokemon_battle'] / env.info['status']['max_hp_pokemon_battle'] < 0.3)
        out_battle_lack_of_health = env.info['context'] == 1
        if in_battle_lack_of_health or out_battle_lack_of_health:
            if abs(i_step - last_step) < interval:
                continue
            last_step = i_step
            name = f'lack_of_health_in_battle_{i_step}.state' if in_battle_lack_of_health else f'lack_of_health_out_battle_{i_step}.state'
            # Saving the state
            state_file_path = f'{folder_path}/{name}'
            with open(state_file_path, "wb") as f:
                env.pyboy.save_state(f)
            print('Saved State')
            # break


def port_recorded_action(state_file, sess_name, save_name, headless=True):
    agent_stat = f'{sess_name}/agent_stats_{save_name}.csv.gz'
    tdf = pd.read_csv(agent_stat, compression='gzip')
    tdf = tdf[tdf['map'] != 'map'] # remove unused
    action_arrays = np.array_split(tdf, np.array((tdf["step"].astype(int) == 0).sum()))
    action_list = [int(x) for x in list(action_arrays[0]["last_action"])]

    ep_length = len(action_list)
    env_config = {
                'headless': headless, 'save_final_state': True, 'early_stop': False,
                'action_freq': 24, 'init_state': state_file, 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': '../PokemonRed.gb', 'debug': False, 'sim_frame_dist': 2_000_000.0, 'extra_buttons': False
            }
    env = make_env(0, env_config)()
    return env, action_list


# For analysis on the recorded actions
sess_name = 'session_healer_01'
sess_path = Path(f'{sess_name}')

for step_num in [81920, 163840]
step_num = 1064960
ckpt_path = f'{sess_name}/poke_{str(step_num)}_steps'

init_state_folder = 'session_trainer/lack_of_health_states'
init_states = glob.glob(init_state_folder+'/*.state')
state_file = init_states[0]

save_name = 'ckpt_{step_num}_wtf'

# Save action using checkpoint file
# prepare_action_list_from_ckpt(state_file, ckpt_path, sess_path, save_name)

# Load recorded action & env
env, action_list = port_recorded_action(state_file, sess_name, save_name, headless=True)

# Run & Visualize Policy performance
reward_history = defaultdict(list)
os.makedirs(f'{sess_name}/inspection', exist_ok=True)
vid_path = f'{sess_name}/inspection/{save_name}_visualize.mp4'
policy_writer = media.VideoWriter(vid_path, (164, 242), fps=20)
policy_writer.__enter__()
obs, info = env.reset()
for action in action_list:
    obs, rewards, term, trunc, info = env.step(action)
    update_hist_rew(info, reward_history)
    img_game = env.render(reduce_res=False, update_mem=False)
    img_game = visualize_reward_dynamic(img_game, reward_history)
    # print('\n Shape: ', img_game.shape)
    policy_writer.add_image(img_game.astype(np.uint8))

policy_writer.close()
env.close()


# Expected behavior: heal reward never triggered, unless the pokemon levels up
# hist = check_triggered_reward_and_save(env, action_list, 'heal', sess_name, record_history_length=10, save_video=True)

# Store pyboy state for bootstrapping RL with compositionality

# save_state_for_lack_of_health_context(env, action_list, sess_name)

# Two ways ahead: 1. Separate policy for healing mode
