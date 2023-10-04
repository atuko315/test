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



sample_s_path = '/home/student/PARL/benchmark/torch/AlphaZero/best_200.pth.tar'
sample_b_path = '/home/student/PARL/benchmark/torch/AlphaZero/saved_model/checkpoint_1.pth.tar'

game = Connect4Game()
strong_timellimit = 5
weak_timelimit = 0.5
strong_puct = 1
weak_puct = 0.1
sample_system = System(game, sample_s_path, sample_b_path, turn=1, strong_timelimit=strong_timellimit,
                        weak_timelimit=weak_timelimit, strong_puct=strong_puct, weak_puct=weak_puct)
size1 = len(sorted(Path('./label/important/important/short').glob('*.board')))
size2 = len(sorted(Path('./label/trivial/trivial/short').glob('*.board')))
print(size1, size2)
paths1 = sorted(Path('./label/important/important/short').glob('*.board'))[-size1: ]
paths2 = sorted(Path('./label/trivial/trivial/short').glob('*.board'))[-size2: ]
dataset1 = DatasetManager(game, paths1)
dataset2 = DatasetManager(game, paths2)
dataset1.make_board_set()
dataset2.make_board_set()

pattern_paths0 = dataset1.make_pattern_path_set(dataset1.pattern_set[1])


#まず片側

sample_dataset = DatasetManager(game, pattern_paths0)
#steps = [1, 3, 5, 7, 9]
steps = [1, 3, 5]
#steps = [1]
analist = [1, -1]
count = 0
board1 = np.array(
[[ 1, 1, 0, 0, 0, 0, 0],
 [ 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 0,-1, 0, 0, 0],
 [ 0, 0, 0,-1, 0, 0, 0],
 [ 0,-1, 0, -1, 1, 0, 0],
 [ 0, 1,-1, 0, 1, 1,-1]], dtype=np.int32)
print(dataset1.match_pattern(board1, dataset1.pattern_set[1]))

for step in steps:
    lbcount = 0
    lacount = 0
    rbcount = 0
    racount = 0
    count = 0
    for p in sample_dataset.path_set:
        height, width = game.getBoardSize()
        imp, board, branch, fpath, importance = load_data(p)
        c, p = sample_dataset.match_pattern(board, dataset1.pattern_set[1])
        observe_l = []
        observe_r = []
        for s in c:
            h = int(s/width) - 1
            w = s % width
            if w > 0:
                if board[h][w-1] == 0:
                    observe_l.append(s)
                    lbcount += 1
                    #print("lb")
            if w < width -2:
                if board[h][w+2] == 0:
                    observe_r.append(s)
                    rbcount += 1
                    #print("rb")
           


        #print(board)
        fboard, category = sample_system.detectHotState( board, -1, fpath, getStep(board), limit=step)
        #print(fboard)
    
        for s in observe_l:
            
            h = int(s/width) - 1
            w = s % width
            print(s, h, w)
            if fboard[h][w-1] == 0:
                lacount += 1
                #print("la")
        
        for s in observe_r:
            h = int(s/width) - 1
            w = s % width
            if fboard[h][w+2] == 0:
                racount += 1
                #print("ra")
        
        
        #print(fboard)
        #print("---------------------------")
        
    print(step, lbcount, lacount, rbcount, racount)


