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
datasets = [dataset1, dataset2, dataset3, dataset4]

print("pattern, 1, 2, 3, 4")
board = np.array(
[[ 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 0,-1, 1, 0,-1],
 [ 0,-1,-1, 1, 1, 0, 1],
 [ 0,-1, 1,-1, 0, 0, 1]], dtype=np.int32)
#print(dataset1.pattern_set[2])
#contain_indices, pure_indices = dataset1.match_pattern(board, dataset1.pattern_set[2])
#print(contain_indices) #ななめいけてない？？？6,7, 10, 11も 
#for dataset in datasets:
#    pattern_path_set = dataset.make_pattern_path_set(dataset.pattern_set[0])

analist = -1
step = 1

for i in range(4):
        dataset = datasets[i]
        pattern_path_set = dataset.make_pattern_path_set(dataset.pattern_set[0])
        size = len(pattern_path_set)
        traj_set, vec_set, dist_set = dataset.collect_pattern_vector(dataset.pattern_set[0], sample_system, analist, step=step)
        traj_size = len(traj_set)
        vector = None
        distance = None
        print(traj_set)
        print(vec_set)
        print(dist_set)
        if vec_set:
            vector = np.mean(np.array(sum(vec_set, [])))
        if dist_set:
            distance = np.mean(np.array(sum(dist_set, [])))
        metric = abs(vector)*distance
        print(f"{i+1} {abs(vector)} {distance} {metric} {size} {traj_size}")

