import sys
import math
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
from sotsuron.feature import DatasetManager
from random import uniform
import random
from time import sleep
from collections import defaultdict
from statistics import quantiles

sample_s_path = '/home/student/PARL/benchmark/torch/AlphaZero/best_200.pth.tar'
sample_b_path = '/home/student/PARL/benchmark/torch/AlphaZero/checkpoint_1.pth.tar'

game = Connect4Game()
strong_timellimit = 5
weak_timelimit = 1
strong_puct = 1
weak_puct = 0.25
sample_system = System(game, sample_s_path, sample_b_path, turn=1, strong_timelimit=strong_timellimit,
                        weak_timelimit=weak_timelimit, strong_puct=strong_puct, weak_puct=weak_puct)

#paths1 = sorted(Path('./label/important/important/long').glob('*.board'))
#paths2 = sorted(Path('./label/important/trivial/long').glob('*.board'))
#paths3 = sorted(Path('./label/trivial/important/long').glob('*.board'))
#paths4 = sorted(Path('./label/trivial/trivial/long').glob('*.board'))
#
#paths = [paths1, paths2, paths3, paths4]
analist = 1
paths = sorted(Path('./poffdata').glob('*.history'))[:100]
#print(len(paths1), len(paths2), len(paths3), len(paths4))
print(len(paths))




sdata = []
wdata = []
dthreshold = 0.004
uthreshold = 0.18
count = [0, 0, 0, 0]
#　1、2、3、４とそのエージェントにとって重要かどうかは別途考えるべき そもそも二つの　相関を求める。

i=0
for p in paths:
    print(f"{i}/{len(paths)}")
    h = load_data(p)
    for j in range(len(h)-1):
        board = h[j][0]
        #print(board)
        #imp, board, branch, fpath, importance = load_data(p)
        new_simp = sample_system.getMyImportance(board, 1, p, getStep(board))
        new_wimp = sample_system.getMyImportance(board, -1, p, getStep(board))
        sdata.append(new_simp)
        wdata.append(new_wimp)
        
        #print(new_simp, new_wimp)
        if new_simp <= dthreshold:
            if new_wimp <= dthreshold:
                count[4-1] += 1
            elif new_wimp >= uthreshold:
                count[3-1] += 1
        
        elif new_simp >= uthreshold:
            if new_wimp <= dthreshold:
                count[2-1] += 1
            elif new_wimp >= uthreshold:
                count[1-1] += 1
    

    #sdata.append(new_simp)
    #wdata.append(new_wimp)


print(count)
coef = np.corrcoef(sdata, wdata)
print(coef)
print(sdata)
print(wdata)

sq1 = math.ceil(np.percentile(sdata, 25))
sq2 = math.ceil(np.percentile(sdata, 50))
sq3 = math.ceil(np.percentile(sdata, 75))
print(sq1, sq2, sq3, np.mean(sdata))

wq1 = math.ceil(np.percentile(wdata, 25))
wq2 = math.ceil(np.percentile(wdata, 50))
wq3 = math.ceil(np.percentile(wdata, 75))

print(wq1, wq2, wq3, np.mean(wdata))



