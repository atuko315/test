# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:35:54 2023

@author: uguis
"""

import sys
import base64
import inspect
import os
import io
from submission_sample import SimpleAgent, dotdict, MCTS

from tqdm import tqdm
from parl.utils import logger
from connectx_try import encode_weight, load_buffer, System, extract
from connect4_game import Connect4Game
from pathlib import Path
from connectx_try import load_data, getCurrentPlayer
from random import uniform

sample_s_path = '/home/student/PARL/benchmark/torch/AlphaZero/best_200.pth.tar'
sample_b_path = '/home/student/PARL/benchmark/torch/AlphaZero/saved_model/checkpoint_1.pth.tar'
encoded_weights = encode_weight(sample_b_path)
print("encoded")
buffer = load_buffer(sample_b_path)
print("buffer")
game = Connect4Game()


    
strong_timelimit = 0.005
weak_timelimit = 0
strong_puct = 0
weak_puct = 0
sample_system = System(game, sample_s_path, sample_b_path, turn=1, strong_timelimit=strong_timelimit,
                    weak_timelimit=weak_timelimit, strong_puct=strong_puct, weak_puct=weak_puct)
cycle = 200
#cycle = 1
results = []
print("config")
print(strong_timelimit, weak_timelimit, strong_puct, weak_puct, cycle)
for i in range(cycle):
    turn = 1 if i % 2 == 0 else -1
    result = sample_system.playGameWithPolicy(None, ["v2u"], turn, verbose=False, reach=True) * turn
    results.append(result)


print(results.count(1), results.count(-1))
print(results)