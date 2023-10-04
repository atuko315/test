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

#factor = 31

sample_s_path = '/home/student/PARL/benchmark/torch/AlphaZero/best_200.pth.tar'
sample_b_path = '/home/student/PARL/benchmark/torch/AlphaZero/saved_model/checkpoint_1.pth.tar'

game = Connect4Game()
strong_timellimit = 5
weak_timelimit = 0.5
strong_puct = 1
weak_puct = 0.1
sample_system = System(game, sample_s_path, sample_b_path, turn=1, strong_timelimit=strong_timellimit,
                        weak_timelimit=weak_timelimit, strong_puct=strong_puct, weak_puct=weak_puct)

paths1 = sorted(Path('./label/important/important/long').glob('*.board'))
paths2 = sorted(Path('./label/important/trivial/long').glob('*.board'))
paths3 = sorted(Path('./label/trivial/important/long').glob('*.board'))
paths4 = sorted(Path('./label/trivial/trivial/long').glob('*.board'))
dataset1 = DatasetManager(game, paths1)
dataset4 = DatasetManager(game, paths4)
paths = [paths1, paths2, paths3, paths4]

baseline = 3
analist = 1

print(len(paths1), len(paths2), len(paths3), len(paths4))
path = sorted(Path('./label/important/important/long').glob('*.board'))[-2]
imp, fboard, branch, fpath, importance = load_data(path)
print(fboard)
reach = dataset1.detect_actual_reach(path, sample_system)
boards = dataset1.collect_promising(fboard, path, sample_system, analist, step=5, baseline=3)
#print(boards)
#print(dataset1.check_convergence(boards, reach, fpath, getStep(fboard), sample_system, analist))
print(dataset1.hot_states_one_way(fboard, path, sample_system, analist, step=5, toend=True))

'''
for  i in range(len(paths)):
    dataset = DatasetManager(game, paths)
    result = []
    total_size = len(paths[i])
    print(f"total size: {total_size}")
'''
