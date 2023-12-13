import collections
from collections import defaultdict

def calculate_decay_rate(half_life):
    return 0.5**(1/half_life)

class mapAC:
    def __init__(self, evap_half_life=400, initial_discount=0.1, excite_half_life=40):
        self.evap_rate = calculate_decay_rate(evap_half_life)
        self.initial_discount = initial_discount
        self.excite_discount = calculate_decay_rate(excite_half_life)
        self.time = -1
        self.excite_signals = list()
        self.pheromone_map = defaultdict(float)
    
    def _excite(self, x, y, map, reward):
        self.excite_signals.append(reward * self.initial_discount)
        
    def _time_elapse(self):
        self.time += 1
        self.excite_signals = [signal * self.excite_discount for signal in self.excite_signals]
        for (x,y,map) in self.pheromone_map:
            self.pheromone_map[(x, y, map)] = self.pheromone_map[(x, y, map)] * self.evap_rate

    def _update_pheromone(self, x, y, map):
        self.pheromone_map[(x,y,map)] += sum([pheromone for pheromone in self.excite_signals])

    def update(self, x, y, map, reward_gain = 0):
        # Agent updates its own excitement signal
        if reward_gain > 0:
            self._excite(x, y, map, reward_gain)
        # Agent release pheromon to the map
        self._update_pheromone(x, y, map)
        # Map's pheromon evaporates over time && Agent's excitement signal decays over time
        self._time_elapse()
        
    def get_pheromone(self, x, y, map):
        return self.pheromone_map[(x, y, map)]
    