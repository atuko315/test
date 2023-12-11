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

paths = sorted(Path('./data').glob('*.history'))
print(len(paths))


analist = 1
step = 2
baseline = 6
promising = 4
start = 10
end = 22
print(" 1と6")
print(f"analist: {analist}, step: {step}, baseline: {baseline}, promising: {promising}, start: {start}, end: {end}")
dataset = DatasetManager(game, [])

ave_bfrate = 0
ave_bfdrate = 0
ave_sfrate = 0
ave_sfdrate = 0
size = 0
for p in paths:
    

    #if getStep(board) < 15:
    #    continue
    #if abs(analist) == 1:
    #    if getCurrentPlayer(board) != analist:
    #        continue

    h = load_data(p)
    tmp = h[len(h)-1]
    #print(tmp[3], tmp[5], tmp[4], tmp[6])
    strong_timellimit = tmp[3]
    weak_timelimit = tmp[5]
    strong_puct = tmp[4]
    weak_puct = tmp[6]

    sample_system = System(game, sample_s_path, sample_b_path, turn=1, strong_timelimit=strong_timellimit,
                        weak_timelimit=weak_timelimit, strong_puct=strong_puct, weak_puct=weak_puct)

    
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

    
    if size%100 == 0:
        print(f"{size}/{len(paths)}")
    if len(h) <= start:
        continue
    tmp_end = min(end, len(h)-2)
    for i in range(start, tmp_end+1):
        tmp = h[i]
        board, sNsa, bNsa, sv, bv, sVs, bVs = tmp
        memory = h[i]
        size += 1
        
        '''
        bfcount, bfdcount = dataset.hot_result_cache(board, memory, reach, sample_system, analist, mode="focus")
        ave_bfrate += bfcount
        ave_bfdrate += bfdcount

        '''
        answer = dataset.hot_states_two_ways_cache(board, memory, reach, sample_system, analist, step=step, baseline=baseline, promising=promising, mode="focus")
        bfcount, bfdcount, sfcount, sfdcount = answer
        ave_bfrate += bfcount
        ave_bfdrate += bfdcount
        ave_sfrate += sfcount
        ave_sfdrate += sfdcount
        
        
        

ave_bfrate /= size
ave_bfdrate /= size
ave_sfrate /= size
ave_sfdrate /= size
print(f"result: {i+1} {ave_bfrate} {ave_bfdrate} {ave_sfrate} {ave_sfdrate} {size}")
#print(f"{i+1} {ave_brate} {ave_bfrate} {ave_bfdrate} {ave_srate} {ave_sfrate} {ave_sfdrate} {size}")

