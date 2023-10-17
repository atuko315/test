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


paths1 = sorted(Path('./label/important/important/middle').glob('*.board'))
paths2 = sorted(Path('./label/important/trivial/middle').glob('*.board'))
paths3 = sorted(Path('./label/trivial/important/middle').glob('*.board'))
paths4 = sorted(Path('./label/trivial/trivial/middle').glob('*.board'))

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
for i in range(3):
    size1 = len(dataset1.make_pattern_set(dataset1.pattern_set[-(i+8)])) / len(paths1)
    size2 = len(dataset2.make_pattern_set(dataset1.pattern_set[-(i+8)])) / len(paths2)
    size3 = len(dataset3.make_pattern_set(dataset1.pattern_set[-(i+8)])) / len(paths3)
    size4 = len(dataset4.make_pattern_set(dataset1.pattern_set[-(i+8)])) / len(paths4)
    print(f"{i+1}, {size1}, {size2}, {size3}, {size4}")
    
