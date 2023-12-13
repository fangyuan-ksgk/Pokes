# Ant algorithm needs some testing
import numpy as np
import cv2

# Test on my ACmap algorithm -- under random walks, what is the pheromone map looks like?
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
    

# utils
def update_test_map_with_ac(test_map, ac, scale_factor):
    for x,y,_ in ac.pheromone_map:
        color_pheromone = ac.get_pheromone(x,y,0)
        pos= x * scale_factor, y * scale_factor
        # print(f'Pos ({x}, {y}) original color {test_map[pos[0], pos[1]]} new color {color_pheromone + test_map[pos[0], pos[1]]}')
        test_map[pos[0]:pos[0]+scale_factor, pos[1]:pos[1]+scale_factor] = color_pheromone
    test_map = np.clip(test_map, 0, 255)
    return test_map

# Test
scale_factor = 50
width, height = 21, 21
test_map = np.zeros((width*scale_factor,height*scale_factor), dtype=float)
excite_pos = (10*scale_factor,10*scale_factor)
excite_pos = excite_pos[0]%(width*scale_factor), excite_pos[1]%(height*scale_factor)

# Exponential Decay is too fast here ... 
pheromone_half_life = 400 # In 1000 steps, the pheromone will decay to 50% 半衰期
excite_half_life = 40 # In 1000 steps, the excitement will decay to 50% 半衰期
initial_discount = 0.5
ac = mapAC(pheromone_half_life, initial_discount, excite_half_life)

pos = excite_pos

while True:
    pos = pos[0]%(width*scale_factor), pos[1]%(height*scale_factor)
    pos_ac = pos[0]//scale_factor, pos[1]//scale_factor
    ac.update(pos_ac[0], pos_ac[1], 0, 255 if pos==excite_pos else 0)
    test_map = update_test_map_with_ac(test_map, ac, scale_factor)

    direction = np.random.randint(4)
    if direction == 0:
        pos = (pos[0], pos[1]+1*scale_factor)
    elif direction == 1:
        pos = (pos[0], pos[1]-1*scale_factor)
    elif direction == 2:
        pos = (pos[0]-1*scale_factor, pos[1])
    elif direction == 3:
        pos = (pos[0]+1*scale_factor, pos[1])
    else:
        raise Exception("Invalid direction")

    test_map_with_circle = test_map.copy()
    cv2.circle(test_map_with_circle, (pos[1]+scale_factor//2, pos[0]+scale_factor//2), 5, 255, -1)
    cv2.imshow("Pheromone Trace of Ants (Food at MapCenter)", test_map_with_circle.astype(np.uint8))
    cv2.waitKey(2)
cv2.destroyAllWindows()