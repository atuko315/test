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
baseline = 2
step = 4
tail = 3
print("手番修正版")
print(f"analist: {analist} baseline:{baseline} step:{step} tail:{tail}")
for  i in range(len(paths)):
    if i != 3:
        continue
    #次の一手がどっちの手番かで分けるべき
    #つまり相手の力量を打ちながら計る必要がある？？？？
    dataset = DatasetManager(game, paths[i])
    for p in paths[i]:
        content = load_data(p)
        if len(content) < 5:
            importance, board, brance, fpath = content
        else:
            imp, board, branch, fpath, importance = content
        
        h = load_data(fpath)
        tmp = h[len(h)-2]
        fboard, sNsa, bNsa, sv, bv, sVs, bVs = tmp
        
        valid = game.getValidMoves(fboard, getCurrentPlayer(fboard))
        valid = [i  for i in range(len(valid)) if valid[i]]
        reach = []
        for a in valid:
            vboard = dataset.add_stone(fboard.copy(), getCurrentPlayer(fboard), a)
            vf = sample_system.detectFatalStone(vboard, per_group=True)
            if vf:
                reach.extend(vf)
        memory = h[getStep(board)]

        #ra = dataset.hot_result(board, p, sample_system, analist, mode="focus")
        ra = dataset.hot_states_one_way_cache(board, sample_system, analist, memory, reach, step, baseline, mode="traj")
        rb = dataset.hot_trajs_cache( board, memory, reach, sample_system, analist, baseline, step, tail=tail)
        print("--------")
        


    #print(f"{i+1} {ave_brate} {ave_bfrate} {ave_bfdrate} {ave_srate} {ave_sfrate} {ave_sfdrate} {size}")

