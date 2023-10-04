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
compare = defaultdict(lambda: [])
factor = 31
print(factor)
def labels_before_after(olabel, dataset, system, pattern, threshold=0.6):
    size = len(dataset.path_set)
    #print(size)
    data = []
    
    for i in range(size):
        tmp = {}
       
        
        h = load_data(dataset.path_set[i])
        _, board, branch, fpath, importance = h
        step = getStep(board)
        #simp = system.getImportance(board, 1, fpath, step)
        #wimp = system.getImportance(board, -1, fpath, step)
        #print(f"original: simp:{simp}, wimp:{wimp}")
        
       
        number, _ = dataset.match_pattern(board, pattern)
        #print(number)
        if len(number) > 0 and factor in number:
            #print(board)
            #print(i)
            tmp["label"] = olabel
            tmp["board"] = board
            print(olabel)
            for j in number:
                
                compare[j].append(tmp)
            
            #print(board)
            print("detect")
            #verbose = np.random.choice([0, 1], p=[0.8, 0.2])
            '''
            verbose = True
            alter_board, simp, wimp = dataset.generate_alternative_board(dataset.path_set[i], system, 'alter', change=1, verbose=verbose)
            print("alter board")
            print(alter_board)
            alter_number, _ = dataset.match_pattern(alter_board, pattern)
            #simp = system.getImportance(alter_board, 1, fpath, step)
            #wimp = system.getImportance(alter_board, -1, fpath, step)
            if simp > threshold:
                if wimp > threshold:
                    label = 1
                else:
                    label = 2
            else:
                if wimp > threshold:
                    label = 3
                else:
                    label = 4
            print(f"label: {olabel}->{label}, margin: {abs(olabel-label)}")

            tmp = (olabel, label, board, alter_board, alter_number)
            data.append(tmp)
            if abs(label - olabel) == 3:
                compare.append(tmp)
            '''
    
    #return data, compare
    return None, compare
        



sample_s_path = '/home/student/PARL/benchmark/torch/AlphaZero/best_200.pth.tar'
sample_b_path = '/home/student/PARL/benchmark/torch/AlphaZero/saved_model/checkpoint_1.pth.tar'

game = Connect4Game()
strong_timellimit = 5
weak_timelimit = 0.5
strong_puct = 1
weak_puct = 0.1
sample_system = System(game, sample_s_path, sample_b_path, turn=1, strong_timelimit=strong_timellimit,
                        weak_timelimit=weak_timelimit, strong_puct=strong_puct, weak_puct=weak_puct)
size2 = len(sorted(Path('./label/important/trivial/short').glob('*.board')))
size3 = len(sorted(Path('./label/trivial/important/short').glob('*.board')))
print(size2, size3)
paths1 = sorted(Path('./label/important/important/long').glob('*.board'))
paths2 = sorted(Path('./label/important/trivial/long').glob('*.board'))
paths3 = sorted(Path('./label/trivial/important/long').glob('*.board'))
paths4 = sorted(Path('./label/trivial/trivial/long').glob('*.board'))
dataset1 = DatasetManager(game, paths1)
dataset2 = DatasetManager(game, paths2)
dataset3 = DatasetManager(game, paths3)
dataset4 = DatasetManager(game, paths4)
dataset1.make_board_set()
dataset2.make_board_set()
dataset3.make_board_set()
dataset4.make_board_set()

#pattern_paths1 = dataset1.make_pattern_path_set(dataset1.pattern_set[0])
#pattern_paths4 = dataset4.make_pattern_path_set(dataset1.pattern_set[0])

#sub_dataset1 = DatasetManager(game, pattern_paths1)
#sub_dataset4 = DatasetManager(game, pattern_paths4)
board1 = np.array(
[[ 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 1, 0, 0, 0, 0],
 [ 1, 0,-1,-1, 0, 0, 0],
 [ 1, 0,-1, 1,-1, 0, 1]], dtype=np.int32)

#dataset1.collect_hot_states(sample_system, 1) 
#print("start 4")
#dataset4.collect_hot_states(sample_system, 1) 
#print(sub_dataset1.match_pattern(board1, sub_dataset1.pattern_set[0]))
print(size3)
for p in dataset2.pattern_set:
    contain_count, pure_count = dataset3.accumulate_pattern(p)
    print(contain_count, pure_count)
    

'''
for p in paths1:
    
    imp, board, branch, fpath, importance = load_data(p)
    sHot = sample_system.detectHotState(board, 1, fpath, getStep(board), limit=5)
    wHot = sample_system.detectHotState(board, -1, fpath, getStep(board), limit=5)
    print(sHot)
    print(wHot)
    print("-------------------------")
'''
#hotStates1 = sub_dataset4.collect_hot_states(sample_system, 1, limit=5)

#hotStates_1 = sub_dataset4.collect_hot_states(sample_system, -1, limit=5)
#print(hotStates1)
#print(hotStates_1)
#print("start 4")
#hotStates4 = sub_dataset4.collect_hot_states(sample_system, 1, limit=5)
#print(hotStates4)