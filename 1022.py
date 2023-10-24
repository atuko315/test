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


paths1 = sorted(Path('./label/important').glob('*.board'))
paths2 = sorted(Path('./label/trivial').glob('*.board'))
paths1.extend(paths2)
paths = paths1
print(f"original: {len(paths)}")
dataset = DatasetManager(game, paths)
dataset.make_board_set()

#　方向は-1, 0, 1
# 距離は0〜５
# tmp_trajごとに一番多いベクトルと距離を取り出す
analists = [-1, 1]
steps = [1, 3, 5]
print("analist, j, step, vector, distance, metric, size, traj_size, path_size")
for step in steps:
    for analist in analists: 
        for j in range(len(dataset.path_set)-1):
        
            pattern_path_set = dataset.make_pattern_path_set(j)
            pattern_dataset = DatasetManager(game, pattern_path_set)
            size = len(pattern_path_set)
            traj_set, vec_set, dist_set = pattern_dataset.collect_pattern_vector_origin(sample_system, j, analist, step=step)
            traj_size = len(traj_set)
            path_size = len(pattern_dataset.fpath_set)
            vector = None
            distance = None
            metoric = None
            if vec_set:
                vector = abs(np.mean(np.array(vec_set)))
            if dist_set:
                distance = np.mean(np.array(dist_set))
            if vector and distance:
                metric = abs(vector)*distance
            print(f"{analist}, {j}, {step}, {vector}, {distance}, {metric}, {size}, {traj_size}, {path_size}")



