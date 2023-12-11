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
from sotsuron.feature import DatasetManager
from random import uniform
import random
from time import sleep
from collections import defaultdict

#factor = 31

sample_s_path = '/home/student/PARL/benchmark/torch/AlphaZero/best_200.pth.tar'
sample_b_path = '/home/student/PARL/benchmark/torch/AlphaZero/checkpoint_1.pth.tar'

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

paths = [paths1, paths2, paths3, paths4]


print(len(paths1), len(paths2), len(paths3), len(paths4))

#boards = dataset1.collect_promising(fboard, path, sample_system, analist, step=4, baseline=3)
#print(boards)
#print(dataset1.check_convergence(boards, reach, fpath, getStep(fboard), sample_system, analist))


analist = 1
step = 2
baseline = 2
promising = 4
print(" 1と２")
print(f"analist: {analist}, step: {step}, baseline: {baseline}, promising: {promising}")
for  i in range(len(paths)):
    
    #次の一手がどっちの手番かで分けるべき
    #つまり相手の力量を打ちながら計る必要がある？？？？
    dataset = DatasetManager(game, paths[i])
    #ave_brate, ave_bfrate, ave_bfdrate, ave_srate, ave_sfrate, ave_sfdrate, size = dataset.collect_two_ways(sample_system, analist, step=step, baseline=baseline, promising=promising)
    bfcount, bfdcount, sfcount, sfdcount, size = dataset.collect_two_ways_cache(sample_system, analist, step=step, baseline=baseline, promising=promising, mode="focus")
    print(f"result: {i+1} {bfcount} {bfdcount} {sfcount} {sfdcount} {size}")
    #print(f"{i+1} {ave_brate} {ave_bfrate} {ave_bfdrate} {ave_srate} {ave_sfrate} {ave_sfdrate} {size}")

