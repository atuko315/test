# -*- coding: utf-8 -*-
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
#そもそもの分布　どの時点でどの局面が多いのか

sample_s_path = '/home/student/PARL/benchmark/torch/AlphaZero/best_200.pth.tar'
sample_b_path = '/home/student/PARL/benchmark/torch/AlphaZero/saved_model/checkpoint_1.pth.tar'
encoded_weights = encode_weight(sample_b_path)
buffer = load_buffer(sample_b_path)

game = Connect4Game()


    
strong_timelimit = 5
weak_timelimit = 0.5
strong_puct = 1
weak_puct = 0.1
sample_system = System(game, sample_s_path, sample_b_path, turn=1, strong_timelimit=strong_timelimit,
                    weak_timelimit=weak_timelimit, strong_puct=strong_puct, weak_puct=weak_puct)
labels = ["important", "trivial"]
lengths = ["short", "middle", "long"]
pattern = [[0, 0], [0, 1], [1, 0], [1, 1]]
for i in range(len(pattern)):
    sizes = []
    for l in lengths:
        
        p = os.path.join('./label', labels[pattern[i][0]])
        p = os.path.join(p, labels[pattern[i][1]])
        p = os.path.join(p, l)
        
        #print(type(p))
        paths = sorted(Path(p).glob('*.board'))
        size = len(paths)
        sizes.append(size)
    
    print(f"{i+1}, {sizes[0]}, {sizes[1]}, {sizes[2]}")