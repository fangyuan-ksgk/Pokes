from red_gym_env import RedGymEnv
from utils.pheromon import mapAC

# Today's Idea
# The reward function is simply too bad, and the agent is not progressing here...
# LLM can do planning, I can build some decomposable tools, which can be trained with sub-reward functions
# designed by the LLM, since these sub-task is much more simple than play the entire game, I expect the result to be better

# Ant-search algorithm leaves dense reward in the map, which can be used to design dense reward function

# I will ignore the CNN navigation for now ... (Generalize to other maps is a problem)
# Each Map_n can be decomposed into a set of sub-tasks, which can be trained with sub-reward functions
# Battle & Non-Battle should be dealt with separately
# Heal yourself, enage wild pokemon, explore map

 # General instruction: 
# Ultimate goal is to train a beast pokemon, defeats all opponents, and reach the end of the game
# Design a Reward Function based on the status above, and the reward function should be decomposable

# Hint:
# 1. Record the max opponent level, the higher opponent level implies further progression in the game
# 2. Record the max pokemon level, the higher pokemon level implies more training
# 3. Use master location to store dictionary with information about the map, whether wild pokemon is encounterd, whether npc fight is there, whether it is a pokemon center, etc.
# 4. Train the pokemon to ensure it is stronger than opponents engaged in battle, proceed in map and train again. 

class GroundEnv(RedGymEnv):
    def __init__(self, env_config):
        self.max_opponent_level = 5
        self.done_count = 0
        # Each of the sub-policy corresponds to a specifically designed reward function & terimination condition
        # The initial state to start training each sub-policy is also different
        # Overall Policy is to complete the Pokemon Game, for that, we split to SubPolicy for efficient RL
        self.sub_policy_info = {
            'escape-battle': 'Escape from wild pokemon battle',
            'fight-battle': 'Fight in all battle, inorder to win & level up',
            'heal-pokemon': 'Heal pokemon in pokemon center, in order to recover from wounds',
            'find-wild-pokemon': 'Find wild pokemon in grass, in order to level up', # Levels of wild pokemon matters
            'explore-map': 'Explore the map', # Leaving pins in the map can be a good idea to design dense reward to re-visit these places
            'visit-map-position': 'Visit a specific location on map'
        }
        
        # Ant Colony algorithm for Path Finding
        self.mapAC = self._init_map_ac()
        super().__init__(env_config)
        self.sub_policy = self._determine_sub_policy()
        
    def _init_map_ac(self, pheromone_half_life=400, initial_discount=0.5, excite_half_life=40):
        # Initialize Ant Colony algorithm for path finding
        return mapAC(pheromone_half_life, initial_discount, excite_half_life)
        
    # Built specificly for lack of health scenarios
    def _determine_sub_policy(self, status={}):
        # Determine which sub-policy to use, given the current game state
        if status == {}:
            status = self._get_current_status()
        
        # Health - Strength - Progress (Three key ingredients)
        lack_of_health = status['pokemon_hp_fraction'] < 0.3
        lack_of_strength = max(2 * status['max_opponent_level'], status['max_opponent_level']+10) > max(status['pokemon_levels'])
        lack_of_progress = not lack_of_health and not lack_of_strength
        
        in_battle = status['in_battle']
        wild_pokemon_battle = status['wild_pokemon_battle']
        
        # Health
        if lack_of_health:
            if in_battle:
                if wild_pokemon_battle:
                    return 'escape-battle'
                else:
                    return 'fight-battle'
            else:
                return 'heal-pokemon'
        # Strength
        if lack_of_strength:
            if in_battle:
                return 'fight-battle'
            else:
                return 'find-wild-pokemon'
        # Progression
        if lack_of_progress:
            if in_battle:
                return 'fight-battle'
            else:
                return 'explore-map'
        

    def put_position_pin(self):
        # Put a pin on the map, so that the agent can visit this location later
        # This can be used to design dense reward function
        pass
    
    def _get_terminate_condition(self):
        # Terminate condition for each sub-policy -- Assumes begging with battle state
        die = 'status' in self.info and self.info['status']['die'] == 1
        fully_healed = 'status' in self.info and self.info['status']['pokemon_hp_fraction'] == 1 and not die
        out_of_battle = 'status' in self.info and not self.info['status']['in_battle'] and not die
        terminate = die or out_of_battle
        if terminate:
            print('-- Die --', die, '-- Out of Battle --', out_of_battle)
        return terminate

    # Add termination condition to insist on repetitive training within a specific scenario
    def check_if_done(self):
        # In healing mode, we terminate episode when pokemon dies, to repeat the training untill it figures out how to get healed
        done = super().check_if_done()
        terminate = self._get_terminate_condition()
        self.done_count += int(terminate)
        done = done or (self.done_count>3) # allow a term before terminate
        return done
    
    def _router(self):
        # Router to determine which sub-policy to use
        pass
    
        
    # This will be the status of the Game used for designing reward & termination condition and sub-policy
    def _get_current_status(self):
        # Status forms the basic ingredients for the reward function
        # LLM shall utilize ths status to translate a high-level 'train a beast' into low-level reward functions
        # which should then translate to low-level actions (decomposable ideally, so we avoid CNN completely from time to time)
        status = {}
        prev_status = self.info['status'] if 'status' in self.info else {}
        
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

        # Pokemon Level & Status -- not in battle | pokemon with no levels are not pokemon!
        pokemon_hp_fractions = self.read_hp_fractions()
        pokemon_hp_fraction = self.read_hp_fraction()
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

        # Temporal Event 
        die = 'death_count' in prev_status and death_count > prev_status['death_count']
        escape_battle_alive = 'wild_pokemon_battle' in prev_status and prev_status['wild_pokemon_battle'] and not wild_pokemon_battle and not die
        engage_wild_battle = 'wild_pokemon_battle' in prev_status and not prev_status['wild_pokemon_battle'] and wild_pokemon_battle and not die
        engage_trainer_battle = 'trainer_battle' in prev_status and not prev_status['trainer_battle'] and trainer_battle and not die
        heal_pokemon = 'pokemon_hp_fraction' in prev_status and prev_status['pokemon_hp_fraction'] < 1 and pokemon_hp_fraction == 1 and not die
        status['die'] = die 
        status['escape_wild_battle_alive'] = escape_battle_alive
        status['engage_wild_battle'] = engage_wild_battle
        status['engage_trainer_battle'] = engage_trainer_battle
        status['heal_pokemon'] = heal_pokemon
        

        # Store information inside status dictionary
        status['in_battle'] = in_battle
        status['wild_pokemon_battle'] = wild_pokemon_battle
        status['trainer_battle'] = trainer_battle
        status['pokemon_hp_fractions'] = pokemon_hp_fractions
        status['pokemon_hp_fraction'] = pokemon_hp_fraction
        status['pokemon_levels'] = pokemon_levels
        status['pokemon_types'] = pokemon_types
        status['x_pos'] = x_pos
        status['y_pos'] = y_pos
        status['map_n'] = map_n
        status['max_opponent_level'] = self.max_opponent_level
        status['healing_amount'] = self.heal_amount
        status['death_count'] = death_count
        status['max_event_reward'] = max_event_reward
        status['exploration_reward'] = exploration_reward
        return status
        

    # LLM -- Revised Reward Function
    def get_game_state_reward(self, print_stats=False):
        # fetch game status
        status = self._get_current_status() 
        prev_status = self.info['status'] if 'status' in self.info else {}
        

        # Initialize rewards: Add extra rewards if needed (LLM)
        rewards = {
            'event': status['max_event_reward'], 
            'level': max(status['pokemon_levels']) * 10,
            'heal': 0 if self.info=={} else self.info['rewards']['heal'],
            'battle_power_advantage': 0,
            'battle_damage': 0 if self.info=={} else self.info['rewards']['battle_damage'],
            'battle_loss': 0 if self.info=={} else self.info['rewards']['battle_loss'],
            'efficiency': 0,
            'existence': 0 if self.info=={} else self.info['rewards']['existence'] - 0.1, # urgency
            'type_advantage': 0,
            'death_penalty': -100 * status['death_count'],
            'explore': status['exploration_reward'],
            'survival': 10 if not status['death_count'] else 0,
            'escape_battle': 0 if self.info=={} else self.info['rewards']['escape_battle'],
            'pheromone': 0 if self.info=={} else self.info['rewards']['pheromone']
        }

        # Penalize existence: 
        rewards['existence'] -= 0.1

        # (LLM) Calculate rewards for each sub-policy


        # Escape Battle -- Reward for escaping battle
        if status['escape_wild_battle_alive']:
            rewards['escape_battle'] += 10
            # print('----Escaped Alive!')
        
        # Full-Health Recover -- After Escape Battle, Visit Pokemon Center
        pokemon_center_visit_reward = 0
        if status['heal_pokemon']:
            pokemon_center_visit_reward = 100
            rewards['heal'] += pokemon_center_visit_reward
        

        # During Map Exploration, we want to visit the pokemon center to heal
        if not status['in_battle']:
            self.mapAC.update(status['x_pos'], status['y_pos'], status['map_n'], pokemon_center_visit_reward)
            pheromone_addictive_reward = self.mapAC.get_pheromone(status['x_pos'], status['y_pos'], status['map_n'])
            rewards['pheromone'] += pheromone_addictive_reward
            
            
        
        # Store current information
        total_reward = sum([val for _, val in rewards.items()])
        self.sub_policy = self._determine_sub_policy(status)
        self.info = {'status': status, 'rewards': rewards, 'total_reward': total_reward, 'sub_policy': self.sub_policy}
        return rewards
    
    


    

