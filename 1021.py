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
from scipy.stats import norm, entropy

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
analists = [1, -1]
match = 0


for analist in analists: 
    entropies = []
    for path in paths:
        content = load_data(path)
        if len(content) < 5:
            importance, board, brance, fpath = content
        else:
            imp, board, branch, fpath, importance = content
        
        values = sample_system.getAllPastValues(fpath, getStep(board), board, analist)
        
        counts = sample_system.getPastCount(fpath, getStep(board), board, analist)
        if values[-1] == counts[-1]:
            match += 1
        values = (np.argsort(np.array(values)) + 0.01) / 21.6
        counts = (np.argsort(np.array(counts)) + 0.01) / 21.6
        
       
        e= entropy(values, counts)
        
        entropies.append(e)
    
    match /= len(paths)
    print(analist, np.percentile(entropies, [0, 25, 50, 75, 100]), np.mean(entropies), match)
