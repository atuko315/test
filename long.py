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
size1 = len(sorted(Path('./label/important/important/short').glob('*.board')))
size4 = len(sorted(Path('./label/trivial/trivial/short').glob('*.board')))
paths1 = sorted(Path('./label/important/important/short').glob('*.board'))[-size1: ]
paths4 = sorted(Path('./label/trivial/trivial/short').glob('*.board'))[-size4: ]
dataset1 = DatasetManager(game, paths1)
dataset4 = DatasetManager(game, paths4)
paths = sorted(Path('./offdata').glob('*.history'))
#steps = [1, 3, 5]
steps = [9]

total_size = len(paths)
print(f"total size: {total_size}")
result = []
for s in steps:
    print(f"step= {s}")
    ave_rate = 0
    ave_frate = 0
    ave_fdrate = 0
    pcount = 0
    for p in paths:
        if pcount % 100 == 0:
            print(f"pcount={pcount}")
        pcount += 1
        h = load_data(p)
        tmp = h[len(h)-2]
        fboard, sNsa, bNsa, sv, bv, sVs, bVs = tmp
        #print(fboard)
        #reach = sample_system.detectFatalStone(fboard, reach=True, per_group=True)
        valid = game.getValidMoves(fboard, getCurrentPlayer(fboard))
        valid = [i  for i in range(len(valid)) if valid[i]]
        reach = []
        for a in valid:
            vboard = sample_system.add_stone(fboard.copy(), getCurrentPlayer(fboard), a)
            vf = sample_system.detectFatalStone(vboard, per_group=True)
            if vf:
                reach.extend(vf)
        
        #print("reach")
        #print(reach)
        bboards = dataset1.generate_before_board(fboard, 1, sample_system, p, getStep(fboard), vstep=s)
        size = len(bboards)
        fdcount = 0
        fcount = 0
        count = 0
        #fatal_group = {}
        for b in bboards:
            hot = sample_system.detectHotState(b, 1, p, getStep(b), toend=True)
            if hot[1] != None:
                
                end = game.getGameEnded(hot[0], getCurrentPlayer(hot[0]))
                if end:
                    #print(hot[0])
                    count += 1
                    fatal = sample_system.detectFatalStone(hot[0], per_group=True)
                    #print("fatal")
                    #print(fatal)
                    fu = np.unique(fatal.copy()).tolist() if fatal else [-1]
                    ru = np.unique(reach.copy()).tolist() if reach else [-2]
                    if len(set(ru)) > 0:
                        fdcount += (len(set(fu) & set(ru)) / len(set(ru)))
                    #print(fu, ru, len(set(fu) & set(ru)))
                    if fatal:
                        for g in fatal:
                            for i in range(len(reach)):
                                r = reach[i]
                                if set(r).issubset(set(g)):
                                    fcount += 1
                        '''
                        gs = str(g)
                        #print(fatal_group.keys())
                        if gs not in fatal_group.keys():
                            fatal_group[gs] = 1
                        else:
                            fatal_group[gs] += 1
                        '''
                        

        rate = count / size
        frate = fcount / count if count > 0 else 0
        fdrate = fdcount / count if count > 0 else 0

        ave_rate += rate
        ave_frate += frate
        ave_fdrate += fdrate
        
    
    ave_rate /= total_size
    ave_frate /= total_size
    ave_fdrate /= total_size #ここなんとか
    result.append([s, ave_rate, ave_frate, ave_fdrate])

for r in result:
    print(f"{r[0]},{r[1]},{r[2]},{r[3]}")

        
        

#print(sample_system.detectFatalStone(board1, reach=True))


'''
print(sub_dataset1.match_pattern(board1, sub_dataset1.pattern_set[0]))

hotStates1 = sub_dataset4.collect_hot_states(sample_system, 1, limit=5)

hotStates_1 = sub_dataset4.collect_hot_states(sample_system, -1, limit=5)
print(hotStates1)
print(hotStates_1)
'''


'''
imp, board, branch, fpath, importance = load_data(path)
h = load_data(fpath)
tmp = h[len(h)-2] # 注目する部分

board, sNsa, bNsa, sv, bv, sVs, bVs = tmp
pattern_paths1 = dataset1.make_pattern_path_set(dataset1.pattern_set[0])
pattern_paths4 = dataset4.make_pattern_path_set(dataset1.pattern_set[0])

sub_dataset1 = DatasetManager(game, pattern_paths1)
sub_dataset4 = DatasetManager(game, pattern_paths4)
board1 = np.array(
[[ 0, 0, 0,-1, 0, 0, 0],
 [ 0, 0, 0,-1, 0, 0, 0],
 [ 1, 0, 1,-1, 1, 0, 0],
 [ 1, 0, 1, 1, 1, 0, 0],
 [ 1, 0, 1, 1,-1, 0, 0],
 [ -1,0,-1,-1, 1, 0, 1]], dtype=np.int32) 
print(board)
bboards = dataset1.generate_before_board(board, 1, sample_system, fpath, getStep(board), vstep=2)
size = len(bboards)
count = 0
fatal_group = {}
for b in bboards:
    hot = sample_system.detectHotState(b, 1, fpath, getStep(b), toend=True)
    if hot[1] != None:
        
        end = game.getGameEnded(hot[0], getCurrentPlayer(hot[0]))
        if end:
            print(hot[0])
            count += 1
            fatal = sample_system.detectFatalStone(hot[0], per_group=True)
            print(fatal)
            for g in fatal:
                gs = str(g)
                #print(fatal_group.keys())
                if gs not in fatal_group.keys():
                    fatal_group[gs] = 1
                else:
                    fatal_group[gs] += 1



rate = count / size
print(f"end rate: {rate}")
print("fatal_group")
print(fatal_group)

#print(sample_system.detectFatalStone(board1, reach=True))
'''

'''
print(sub_dataset1.match_pattern(board1, sub_dataset1.pattern_set[0]))

hotStates1 = sub_dataset4.collect_hot_states(sample_system, 1, limit=5)

hotStates_1 = sub_dataset4.collect_hot_states(sample_system, -1, limit=5)
print(hotStates1)
print(hotStates_1)
'''
