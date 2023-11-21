import sys
import base64
import inspect
import os
import io
from submission_sample import SimpleAgent, dotdict, MCTS

from tqdm import tqdm
from parl.utils import logger
from connectx_try import encode_weight, load_buffer, System, extract
from connect4_game import Connect4Game
from pathlib import Path
from connectx_try import load_data, getCurrentPlayer, getStep, store_data
from random import uniform
from time import sleep
from feature import DatasetManager

sample_s_path = '/home/student/PARL/benchmark/torch/AlphaZero/best_200.pth.tar'
sample_b_path = '/home/student/PARL/benchmark/torch/AlphaZero/checkpoint_1.pth.tar'

game = Connect4Game()
strong_timellimit = 5
weak_timelimit = 0.5
strong_puct = 1
weak_puct = 0.1
sample_system = System(game, sample_s_path, sample_b_path, turn=1, strong_timelimit=strong_timellimit,
                        weak_timelimit=weak_timelimit, strong_puct=strong_puct, weak_puct=weak_puct)

paths = sorted(Path('./offdata').glob('*.history'))
phase = ["short", "middle", "long"]
imp_label = ["important", "trivial"]
labels = [[0, 0], [0, 1], [1, 0], [1, 1]]
pattern_number = 21
#n0.004 と0.18
dataset = DatasetManager(game, [])
for name in dataset.pattern_name:
    print(name, end=", ")
print()
for l in labels:
    ex_l = []
    
    path = os.path.join("label", imp_label [l[0]], imp_label [l[1]])
    buffer = [0 for i in range(pattern_number)]
    for ph in phase:
        path = os.path.join("label", imp_label [l[0]], imp_label [l[1]])
        tmp = []
        path = os.path.join(path, ph) 
        path_set = sorted(Path(path).glob('*.board'))
        dataset = DatasetManager(game, path_set)
        print(path, end=", ")
        for i in range(pattern_number):
            pattern_path_set = dataset.make_pattern_path_set(i)
            print(len(pattern_path_set), end=", ")
        print()  

# extreme [[[1short], [1middle], [1long]], [[2short], ...], [[3short], ...], [[4short], ...]]         
                
                 

            

