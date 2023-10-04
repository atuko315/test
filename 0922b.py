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
#print("start")

#上空いてるかをもう一度

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
paths1 = sorted(Path('./label/important/important/short').glob('*.board'))
paths2 = sorted(Path('./label/important/trivial/short').glob('*.board'))
paths3 = sorted(Path('./label/trivial/important/short').glob('*.board'))
paths4 = sorted(Path('./label/trivial/trivial/short').glob('*.board'))
dataset1 = DatasetManager(game, paths1)
dataset2 = DatasetManager(game, paths2)
dataset3 = DatasetManager(game, paths3)
dataset4 = DatasetManager(game, paths4)

dataset1.make_board_set()
dataset2.make_board_set()
dataset3.make_board_set()
dataset4.make_board_set()

datasets = [dataset1, dataset2, dataset3, dataset4]
board1 = np.array(
[[ 1, 1, 0, 0, 0, 0, 0],
 [ 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 0,-1, 0, 0, 0],
 [ 0, 0, 0,-1, 0, 0, 0],
 [ 0,-1, 0,-1, 1, 0, 0],
 [ 0, 1,-1, 0, 1, 1,-1]], dtype=np.int32)
print(dataset1.match_pattern(board1, dataset1.pattern_set[1]))
analists = [1, -1]

for analist in analists:
    for i in range(len(datasets)):
        #print(f"dataset{i+1}")
        pattern_paths0 = datasets[i].make_pattern_path_set(dataset1.pattern_set[1])
        sample_dataset = DatasetManager(game, pattern_paths0)
    
        #steps = [1, 3, 5, 7, 9]
        steps = [1, 3, 5]
    
        
        
        
        
        for step in steps:
            ubcount = 0
            lbcount = 0
            lacount = 0
            rbcount = 0
            racount = 0
            uacount = 0
            count = 0
            for p in sample_dataset.path_set:
                height, width = game.getBoardSize()
                imp, board, branch, fpath, importance = load_data(p)
                c, p = sample_dataset.match_pattern(board, dataset1.pattern_set[1])
                observe_l = []
                observe_r = []
                observe_ul = []
                observe_ur = []
                for s in c:
                    h = int(s/width) - 1
                    w = s % width
                
                    if h > 0:
                        if board[h-1][w] == 0:
                            observe_ul.append(s)
                            ubcount += 1
                            #print("ulb")
                        if board[h-1][w+1] == 0:
                            observe_ur.append(s)
                            ubcount += 1
                            #print("urb")
                            
                    
                    if w > 0:
                        if board[h][w-1] == 0:
                            if h == height-1:
                                observe_l.append(s)
                                lbcount += 1
                            elif board[h+1][w-1] != 0:
                                observe_l.append(s)
                                lbcount += 1
                            #print("lb")

                    if w < width -2:
                        if board[h][w+2] == 0:
                            if h == height-1:
                                observe_r.append(s)
                                rbcount += 1
                            elif board[h+1][w+2] != 0:
                                observe_r.append(s)
                                rbcount += 1
                            #print("rb")
                

                #print(board)
                fboard, category = sample_system.detectHotState( board, analist, fpath, getStep(board), limit=step)
                #print(fboard)
            
                for s in observe_l:
            
                    h = int(s/width) - 1
                    w = s % width
                    
                    if fboard[h][w-1] == 0:
                        lacount += 1
                        #print("la")
        
                for s in observe_r:
                    h = int(s/width) - 1
                    w = s % width
                    if fboard[h][w+2] == 0:
                        racount += 1
                        #print("ra")
                
                for s in observe_ul:
                    h = int(s/width) - 1
                    w = s % width
                    if fboard[h-1][w] == 0:
                        uacount += 1
                        #print("ula")
                
                for s in observe_ur:
                    h = int(s/width) - 1
                    w = s % width
                    if fboard[h-1][w+1] == 0:
                        uacount+= 1
                        #print("ura")

                #print("-----------------")
                #print(fboard)
                #print("---------------------------")
                
                #if len(c1) == 0 and len(c2) == 0:
                #    count += 1
            print(f"{i+1}, {analist}, {step}, {ubcount}, {uacount}, {lbcount+rbcount}, {lacount+racount}")


