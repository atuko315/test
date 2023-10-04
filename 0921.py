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
print("start")

# 縦３パターンのチェック　とりあえず0.5、0.1で　dataからリーチを引っ張る
'''
strong_timelimit = uniform(3, 5)
weak_timelimit = uniform(0, 2)
strong_puct = uniform(0.8, 1)
weak_puct = uniform(0, 0.5)
'''

sample_s_path = '/home/student/PARL/benchmark/torch/AlphaZero/best_200.pth.tar'
sample_b_path = '/home/student/PARL/benchmark/torch/AlphaZero/saved_model/checkpoint_1.pth.tar'

game = Connect4Game()
strong_timellimit = 5
weak_timelimit = 0.5
strong_puct = 1
weak_puct = 0.1
sample_system = System(game, sample_s_path, sample_b_path, turn=1, strong_timelimit=strong_timellimit,
                        weak_timelimit=weak_timelimit, strong_puct=strong_puct, weak_puct=weak_puct)
size = len(sorted(Path('./label/trivial').glob('*.board')))
paths = sorted(Path('./label/trivial').glob('*.board'))

dataset = DatasetManager(game, paths)
dataset.make_board_set()

board1 = np.array(
[[ 1, 1, 0, 0, 0, 0, 0],
 [ 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 0,-1,-1, 0, 0],
 [ 0, 0, 0,-1, 1, 0, 0],
 [ 0,-1,-1,-1, 1, 0, 0],
 [ 0, 1,-1, 1, 1, 1,-1]], dtype=np.int32)
print(dataset.match_pattern(board1, dataset.pattern_set[9]))
print(dataset.pattern_set[9])



pattern_paths = dataset.make_pattern_path_set(dataset.pattern_set[9])
sample_dataset = DatasetManager(game, pattern_paths)

#steps = [1, 3, 5, 7, 9]


analist = -1




uacount = 0
count = 0

for p in sample_dataset.path_set:
    height, width = game.getBoardSize()
    #imp, board, branch, fpath, importance = load_data(p)
    importance, board, brance, fpath = load_data(p)
    c, p = sample_dataset.match_pattern(board, dataset.pattern_set[9])
    observe_u = []
    bflag = 0
    aflag = 0
    for s in c:
        h = int(s/width) -1
        w = s % width
        print(board)
       
        if h > 0:
            if board[h-1][w] == 0:
                observe_u.append(s)
                if bflag == 0:
                    bflag = 1
                    
                    print("bflag")
    
    if bflag == 0:
        continue
    else:
        count += 1
    
    fboard, category = sample_system.detectHotState( board, analist, fpath, getStep(board), limit=1)
    #
    
    
    for s in observe_u:
        h = int(s/width) - 1
        w = s % width
        if fboard[h-1][w] == 0:
            
            aflag = 1
            print(count, uacount)
            break
    
    if aflag == 0:
        print("aflag")
        uacount += 1

    print("-----------------")
    print(fboard)
    #print("---------------------------")
    
    #if len(c1) == 0 and len(c2) == 0:
    #    count += 1
print(count, uacount)


