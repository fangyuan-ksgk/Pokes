from red_gym_env import RedGymEnv

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
        self.sub_policy = self._determine_sub_policy()
        # Ant Colony algorithm for Path Finding
        self.mapAC = self._init_map_ac()
        super().__init__(env_config)
        
    def _init_map_ac(self, evap_rate=0.8):
        # Initialize Ant Colony algorithm for path finding
        import collections
        from collections import defaultdict
        class mapAC:
            def __init__(self, evap_rate):
                self.evap_rate = evap_rate
                self.time = -1
                self.pheromone_map = defaultdict(float)
            
            def excite(self, x, y, map, reward):
                self.pheromone_map[(x, y, map)] += reward
                
            def time_elapse(self):
                self.time += 1
                
            def get_pheromone(self, x, y, map):
                self.pheromone_map[(x, y, map)] *= (self.evap_rate**(max(self.time, 0)))
                return self.pheromone_map[(x, y, map)]
                    
        return mapAC(evap_rate)
        
    def _determine_sub_policy(self):
        # Determine which sub-policy to use, given the current game state
        pass

    def put_position_pin(self):
        # Put a pin on the map, so that the agent can visit this location later
        # This can be used to design dense reward function
        pass
    
    # Add termination condition to insist on repetitive training within a specific scenario
    def check_if_done(self):
        # In healing mode, we terminate episode when pokemon dies, to repeat the training untill it figures out how to get healed
        done = super().check_if_done()
        terminate = False
        self.done_count += int(terminate)
        done = done or (self.done_count>1) # allow a term before terminate
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
        status['max_event_reward'] = max_event_reward
        status['exploration_reward'] = exploration_reward
        return status
        

    # LLM -- Revised Reward Function
    def get_game_state_reward(self, print_stats=False):
        # fetch game status
        status = self._get_current_status()  

        # Initialize rewards: Add extra rewards if needed (LLM)
        rewards = {
            'event': status['max_event_reward'], 
            'level': max(status['pokemon_levels']) * 10,
            'heal': status['healing_reward'],
            'battle_power_advantage': 0,
            'battle_damage': 0 if self.info=={} else self.info['rewards']['battle_damage'],
            'battle_loss': 0 if self.info=={} else self.info['rewards']['battle_loss'],
            'efficiency': 0,
            'existence': 0 if self.info=={} else self.info['rewards']['existence'] - 0.1, # urgency
            'type_advantage': 0,
            'death_penalty': -100 * status['death_count'],
            'explore': status['exploration_reward'],
            'survival': 10 if not status['death_count'] else 0,
            'escape_battle': 0 if self.info=={} else self.info['rewards']['escape_battle']
        }

        # Penalize existence: 
        rewards['existence'] -= 0.1

        # (LLM) Calculate rewards for each sub-policy
        
        # Escape Battle -- Reward for escaping battle
        if not status['wild_pokemon_battle'] and self.info != {} and self.info['status']['wild_pokemon_battle'] > 0:
            rewards['escape_battle'] += 10
        
        # Full-Health Recover -- After Escape Battle, Visit Pokemon Center
        if status['healing_reward'] > 0 and status['pokemon_hp_fractions'][0] == 1:
            pokemon_center_visit_reward = 1000
            self.mapAC.excite(status['x_pos'], status['y_pos'], status['map_n'], pokemon_center_visit_reward)
            rewards['heal'] += pokemon_center_visit_reward
        
        # During Map Exploration, we want to visit the pokemon center to heal
        if not status['in_battle']:
            pheromone_addictive_reward = self.mapAC.get_pheromone(status['x_pos'], status['y_pos'], status['map_n'])
            rewards['heal'] += pheromone_addictive_reward
            
        self.mapAC.time_elapse()
            
        
        # Store current information
        total_reward = sum([val for _, val in rewards.items()])
        self.info = {'status': status, 'rewards': rewards, 'total_reward': total_reward}
        return rewards
    
    


    

