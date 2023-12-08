from red_gym_env import RedGymEnv

# Today's Idea
# The reward function is simply too bad, and the agent is not progressing here...
# LLM can do planning, I can build some decomposable tools, which can be trained with sub-reward functions
# designed by the LLM, since these sub-task is much more simple than play the entire game, I expect the result to be better

# I will ignore the CNN navigation for now ... (Generalize to other maps is a problem)
# Each Map_n can be decomposed into a set of sub-tasks, which can be trained with sub-reward functions
# Battle & Non-Battle should be dealt with separately
# Heal yourself, enage wild pokemon, explore map

class GroundEnv(RedGymEnv):
    def __init__(self, env_config):
        self.max_opponent_level = 5
        self.context_dict = {'in_battle': 0, 'need_heal': 1, 'need_progress': 2, 'need_level_up': 3}
        print('Context Dict initialized, ', self.context_dict)
        self.context = 3
        self.decompose_policy()
        self.done_count = 0
        super().__init__(env_config)

    def decompose_policy(self):
        # Overall Policy is to complete the Pokemon Game, for that, we split to SubPolicy for efficient RL
        self.sub_policy = {
            'escape-battle': 'Escape from wild pokemon battle',
            'fight-battle': 'Fight in all battle',
            'heal-pokemon': 'Heal pokemon in pokemon center',
            'find-wild-pokemon': 'Find wild pokemon in grass', # Levels of wild pokemon matters
            'explore-map': 'Explore the map', # Leaving pins in the map can be a good idea to design dense reward to re-visit these places
            'visit-map-position': 'Visit a specific location on map'
        }
        # Each of the sub-policy corresponds to a specifically designed reward function & terimination condition
        # The initial state to start training each sub-policy is also different

    def _determine_sub_policy(self):
        # Determine which sub-policy to use, given the current game state
        pass

    def put_position_pin(self):
        # Put a pin on the map, so that the agent can visit this location later
        # This can be used to design dense reward function
        pass

    
    # Add termination condition to insist on repetitive training within a specific scenario
    # For instance: Training 'escape-battle' sub-policy, we terminate episode when pokemon dies, or successfully escape from battle
    def check_if_done(self):
        # In healing mode, we terminate episode when pokemon dies, to repeat the training untill it figures out how to get healed
        done = super().check_if_done()
        terminate = False
        self.done_count += int(terminate)
        done = done or (self.done_count>1) # allow a term before terminate
        return done
    
    # Something funny going on with the healing reward
    def update_heal_reward(self):
        # differentiate between resurrections and healing
        cur_health = self.read_hp_fraction()
        if cur_health > self.last_health:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                # if heal_amount > 0.5:
                    # print(f'healed: {heal_amount}')
                    # self.save_screenshot('healing')
                if cur_health == 1.0: # Complete healing
                    self.total_healing_rew += 100.
                else:
                    self.total_healing_rew += 0.
                # self.total_healing_rew += heal_amount * 4
            else:
                self.died_count += 1
        
    # GPT4 should read the docs and design tools, strategy, and reward for the agents
    def check_status(self):
        # Status forms the basic ingredients for the reward function
        # LLM shall utilize ths status to translate a high-level 'train a beast' into low-level reward functions
        # which should then translate to low-level actions (decomposable ideally, so we avoid CNN completely from time to time)
        status = {}
        
        # Whether the player is in battle
        in_battle = self.read_m(0xD057)
        type_of_battle = self.read_m(0xD05A)
        wild_pokemon_battle = in_battle and type_of_battle == 0
        trainer_battle = in_battle and type_of_battle == 1

        if in_battle:
            # Pokemon Level & Status -- in battle
            hp_pokemon_battle = self.read_double_m(0xD015)
            max_hp_pokemon_battle = self.read_double_m(0xD023)
            attack_pokemon_battle = self.read_double_m(0xD025)
            defense_pokemon_battle = self.read_double_m(0xD027)

            # Opponent Pokemon Level & Status
            hp_opponent_battle = self.read_double_m(0xCFE6)
            max_hp_opponent_battle = self.read_double_m(0xCFF4)
            attack_opponent_battle = self.read_double_m(0xCFF6)
            defense_opponent_battle = self.read_double_m(0xCFF8)
            level_opponent_battle = self.read_m(0xCFE8)

            status['hp_pokemon_battle'] = hp_pokemon_battle
            status['max_hp_pokemon_battle'] = max_hp_pokemon_battle
            status['attack_pokemon_battle'] = attack_pokemon_battle
            status['defense_pokemon_battle'] = defense_pokemon_battle
            status['level_opponent_battle'] = level_opponent_battle
            status['hp_opponent_battle'] = hp_opponent_battle
            status['max_hp_opponent_battle'] = max_hp_opponent_battle
            status['attack_opponent_battle'] = attack_opponent_battle
            status['defense_opponent_battle'] = defense_opponent_battle

            # Update max opponent level
            self.max_opponent_level = max(self.max_opponent_level, level_opponent_battle)

        # Pokemon Level & Status -- not in battle
        pokemon_hp_fractions = self.read_hp_fractions()
        pokemon_levels = self.get_pokemon_levels()
        pokemon_types = self.read_party()

        # Map position
        x_pos = self.read_m(0xD362)
        y_pos = self.read_m(0xD361)
        map_n = self.read_m(0xD35E)

        # Healing status
        healing_reward = self.total_healing_rew

        # Death count
        death_count = self.died_count

        # Event reward
        max_event_reward = self.max_event_rew

        # Exploration reward
        exploration_reward = self.get_knn_reward()

        # General instruction: 
        # Ultimate goal is to train a beast pokemon, defeats all opponents, and reach the end of the game
        # Design a Reward Function based on the status above, and the reward function should be decomposable

        # Hint:
        # 1. Record the max opponent level, the higher opponent level implies further progression in the game
        # 2. Record the max pokemon level, the higher pokemon level implies more training
        # 3. Use master location to store dictionary with information about the map, whether wild pokemon is encounterd, whether npc fight is there, whether it is a pokemon center, etc.
        # 4. Train the pokemon to ensure it is stronger than opponents engaged in battle, proceed in map and train again. 

        # Store information inside status dictionary
        status['in_battle'] = in_battle
        status['wild_pokemon_battle'] = wild_pokemon_battle
        status['trainer_battle'] = trainer_battle
        status['pokemon_hp_fractions'] = pokemon_hp_fractions
        status['pokemon_levels'] = pokemon_levels
        status['pokemon_types'] = pokemon_types
        status['x_pos'] = x_pos
        status['y_pos'] = y_pos
        status['map_n'] = map_n
        status['max_opponent_level'] = self.max_opponent_level
        status['healing_reward'] = healing_reward
        status['death_count'] = death_count
        # print('\n line 135: death count ', status['death_count'])
        status['max_event_reward'] = max_event_reward
        status['exploration_reward'] = exploration_reward
        return status


    # Each Context shall be processed by a different sub-policy model
    def get_context(self, status):
        exist_pokemon = [i for i in range(6) if status['pokemon_levels'][i] > 0]
        # If in battle
        if status['in_battle']:
            return self.context_dict['in_battle']
        # If health is below a certain threshold where Pokemon needs healing
        elif any(hp_frac < 0.3 for i, hp_frac in enumerate(status['pokemon_hp_fractions']) if i in exist_pokemon):
            # print('\n -- Context: Pokemon needs healing here')
            # print('Pokemon no. ', i, ' needs healing', ' at level ', status['pokemon_levels'][i])
            return self.context_dict['need_heal']
        # If the player Pokemon's level is significantly lower than the opponent's level, indicating need for training
        elif max(status['pokemon_levels']) < self.max_opponent_level * 2:
            # print('Maximal opponent level is ', self.max_opponent_level, ' and max pokemon level is ', max(status['pokemon_levels']))
            return self.context_dict['need_level_up']
        # Otherwise, assuming need for progression
        else:
            # print('Maximal opponent level is ', self.max_opponent_level, ' and max pokemon level is ', max(status['pokemon_levels']))
            return self.context_dict['need_progress']
        

    # LLM -- Revised Reward Function
    def get_game_state_reward(self, print_stats=False):
        # fetch game status
        status = self.check_status()  

        # Calculate context
        self.context = self.get_context(status)

        # Initialize rewards
        rewards = {
            'event': status['max_event_reward'], 
            'level': max(status['pokemon_levels']) * 10,
            'heal': status['healing_reward'],
            'battle_power_advantage': 0,
            'battle_damage': 0 if self.info=={} else self.info['rewards']['battle_damage'],
            'battle_loss': 0 if self.info=={} else self.info['rewards']['battle_loss'],
            'efficiency': 0,
            'existence': 0 if self.info=={} else self.info['rewards']['existence'],
            'type_advantage': 0,
            'death_penalty': -100 * status['death_count'],
            'explore': status['exploration_reward'],
            'survival': 10 if not status['death_count'] else 0,
            'escape_battle': 0 if self.info=={} else self.info['rewards']['escape_battle']
        }

        # Penalize existence: Bugged to reward existence here??
        rewards['existence'] -= 0.1

        # Experience reward for battling -- stronger than opponent, but not too strong to harvest experience
        if self.context == self.context_dict['in_battle']:
            opponent_strength = status['attack_opponent_battle'] + status['defense_opponent_battle']
            pokemon_strength = status['attack_pokemon_battle'] + status['defense_pokemon_battle']
            lack_of_health = (status['hp_pokemon_battle'] / status['max_hp_pokemon_battle'] < 0.3)
            # Always nice to have advantage over opponent, but too much advantage is not good for gaining experience
            has_advantage = (1.0 < (pokemon_strength / opponent_strength) <= 3.0) and not lack_of_health
            has_disadvantage = ((pokemon_strength / opponent_strength) < 0.6) or lack_of_health
            if has_advantage:
                rewards['battle_power_advantage'] = 2.0 
            elif has_disadvantage:
                rewards['battle_power_advantage'] = -2.0

            # Battle Success - prefer more hp advantage over opponent, the more the better
            # Reward for Winning and Hitting the bitches
            if 'status' in self.info and self.info['status']['in_battle']:
                # Count pokemon damage, the more the better, incremental reward, WarMonger attribute right here
                hp_damage = self.info['status']['hp_opponent_battle'] - status['hp_opponent_battle']
                hp_loss = (self.info['status']['hp_pokemon_battle'] - status['hp_pokemon_battle'])
                # Damage matters only when not lacking in health
                rewards['battle_damage'] += hp_damage * int(not lack_of_health)
                # HP advantage -- run away when in disadvantage
                rewards['battle_loss'] -= hp_loss * int(lack_of_health) * 10.0                
                


        if self.context == self.context_dict['need_progress']:
            # Emphasize Exploration
            rewards['explore'] *= 10.0
            rewards['event']  *= 10.0

        if self.context == self.context_dict['need_heal']:
            # Emphasize Healing
            rewards['heal'] *= 1000.0 # pokemon center visitation, specifically
            if rewards['heal'] > 0:
                # healing spot saved to txt file
                x_pos, y_pos, map_n = status['x_pos'], status['y_pos'], status['map_n']
                self.save_screenshot(f'healing_{x_pos}_{y_pos}_{map_n}')
                # relieve the existence penalty
                rewards['existence'] *= 0.0
                
            rewards['explore'] *= 5.0 # explore when agent can not figure out how to achieve the goal, in this case, healing
            
            # Escaping battle also helps healing
            was_in_wild_battle = 'status' in self.info and self.info['status']['wild_pokemon_battle']
            if was_in_wild_battle:
                print('Was in wild pokemon battle ! Exit from battle without dying!')
                rewards['escape_battle'] += 10.0

        if self.context == self.context_dict['need_level_up']:
            # Emphasize Leveling
            rewards['level'] *= 10.0
            # Emphasize Enaging Battle with Advantage
            rewards['battle_power_advantage'] *= 10.0
            rewards['battle_damage'] *= 10.0
            rewards['battle_loss'] *= 1.0

        # When Context window shifts, scale the total reward to the same level
        # This helps enables the context-change in rewarding structure
        if 'context' in self.info and self.context != self.info['context']:
            curr_total_reward = sum([val for _, val in rewards.items()])
            prev_total_reward = self.info['total_reward']
            scale_factor = prev_total_reward / curr_total_reward
            # Scale each reward by the scaling factor
            rewards = {key: val * scale_factor for key, val in rewards.items()}
        
        # Store current information
        total_reward = sum([val for _, val in rewards.items()])
        self.info = {'status': status, 'context': self.context, 'rewards': rewards, 'total_reward': total_reward}
        return rewards
    
    


    

