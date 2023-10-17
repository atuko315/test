import sys
import base64
import inspect
import os
import io
from submission_sample import SimpleAgent, dotdict, MCTS
import numpy as np
from tqdm import tqdm
from parl.utils import logger
from connectx_try import encode_weight, load_buffer, System, extract
from connect4_game import Connect4Game
from pathlib import Path
from connectx_try import load_data, getCurrentPlayer, getStep, store_data
from feature import DatasetManager
from random import uniform
import random
from time import sleep
from collections import defaultdict
import collections

sample_s_path = '/home/student/PARL/benchmark/torch/AlphaZero/best_200.pth.tar'
sample_b_path = '/home/student/PARL/benchmark/torch/AlphaZero/saved_model/checkpoint_1.pth.tar'

game = Connect4Game()
strong_timellimit = 5
weak_timelimit = 0.5
strong_puct = 1
weak_puct = 0.1
sample_system = System(game, sample_s_path, sample_b_path, turn=1, strong_timelimit=strong_timellimit,
                        weak_timelimit=weak_timelimit, strong_puct=strong_puct, weak_puct=weak_puct)

dataset = DatasetManager(game, [])
paths = sorted(Path('./offdata').glob('*.history'))
finish = defaultdict(lambda: 0)
check_mate = defaultdict(lambda: 0)

finish = [(34, 124), (36, 81), (32, 57), (38, 46), (30, 30), (22, 22), (40, 20), (18, 19), (24, 18), (16, 16), (28, 14), (26, 11), (10, 9), (14, 8), (20, 8), (12, 6), (42, 5), (8, 3)]
finish = dict(finish)
finish = dict(sorted(dict(finish).items()))
print(finish)

values = np.array(list(finish.values()))
total = np.sum(values)
print(total)
print(values)
c_array = np.percentile(values, q=[0, 25, 50, 75, 100])
mean = np.mean(values)
print(c_array, mean)


'''
for p in paths:
    
    contents = load_data(p)
    if len(contents) % 2 == 0:
        continue
    finish[len(contents)-1] += 1
    tmp = contents[len(contents)-2]
    fboard, sNsa, bNsa, sv, bv, sVs, bVs = tmp
    
    valid = game.getValidMoves(fboard, getCurrentPlayer(fboard))
    valid = [i  for i in range(len(valid)) if valid[i]]
    reach = []
    for a in valid:
        vboard, number = sample_system.add_stone(fboard.copy(), getCurrentPlayer(fboard), a, number=True)
        vf = sample_system.detectFatalStone(vboard, per_group=True)
        if vf:
            check_mate[number] += 1

finish = dict(finish)
finish_sorted = sorted(dict(finish).items(), reverse=True, key=lambda x : x[1])
print(finish_sorted)
print(check_mate)

visual = [0 if i not in collections.Counter(check_mate).keys() else collections.Counter(check_mate)[i]
                        for i in range(42)]
visual = np.array(visual).reshape(6, 7)
print(np.sum(visual, axis=0))
print(np.sum(visual, axis=1))
print(visual)
'''
            


