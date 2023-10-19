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

print(f"original: {len(paths1)}, {len(paths2)}, {len(paths3)}, {len(paths4)}")
dataset1 = DatasetManager(game, paths1)
dataset2 = DatasetManager(game, paths2)
dataset3 = DatasetManager(game, paths3)
dataset4 = DatasetManager(game, paths4)
dataset1.make_board_set()
dataset2.make_board_set()
dataset3.make_board_set()
dataset4.make_board_set()

print("pattern, 1, 2, 3, 4")
board1 = np.array(
[[ 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 0, 1, 1, 1, 1],
 [ 0, 0, 0,-1,-1,-1,-1],
 [ 0,-1, 1, 1, 1,-1, 1],
 [ 0,-1, 1, 1,-1, 1,-1]], dtype=np.int32)

pattern_path_set = dataset3.make_pattern_path_set(dataset3.pattern_set[0])
path = pattern_path_set[1]
content = load_data(path)
imp, board, branch, fpath, importance = content
dataset = DatasetManager(game, pattern_path_set)
#traj = dataset1.collect_pattern_vector(dataset.pattern_set[0], sample_system, 1)
print(dataset.hot_states_two_ways(board, path, sample_system, analist=1, step=3, baseline=6, promising=2, mode="focus"))

'''
for i in range(4):
    paths = paths[i]
    dataset = DatasetManager(game, paths)
    dataset.make_board_set()
    pattern_path_set = dataset.make_pattern_path_set(dataset.pattern_set[0])
    #一旦その時点からベクトルをあつめる
    pattern_dataset = DatasetManager(game, pattern_path_set)
'''
