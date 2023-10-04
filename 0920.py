# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:35:54 2023

@author: uguis
"""

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
from connectx_try import load_data, getCurrentPlayer
from random import uniform

sample_s_path = '/home/student/PARL/benchmark/torch/AlphaZero/best_200.pth.tar'
sample_b_path = '/home/student/PARL/benchmark/torch/AlphaZero/saved_model/checkpoint_1.pth.tar'
encoded_weights = encode_weight(sample_b_path)
print("encoded")
buffer = load_buffer(sample_b_path)
print("buffer")
game = Connect4Game()


    
strong_timelimit = uniform(3, 5)
weak_timelimit = uniform(0, 2)
strong_puct = uniform(0.8, 1)
weak_puct = uniform(0, 0.5)
sample_system = System(game, sample_s_path, sample_b_path, turn=1, strong_timelimit=strong_timelimit,
                    weak_timelimit=weak_timelimit, strong_puct=strong_puct, weak_puct=weak_puct)
#sample_system.playGame()
path = sorted(Path('./offdata').glob('*.history'))[-1]
#path = Path("/home/student/PARL/benchmark/torch/AlphaZero/data/20230526180235.history")
h = load_data(path)
print(len(h) - 4)
#sample_system.train_offline(path)
#print(h[5][0])
#print(getCurrentPlayer(h[5][0]))

#print(h)
#sample_system.simulate(path, 30, onlyS=True)
#analysis = sample_system.analyzeAll(path, interactive=False)
#for info in analysis:
    #print(info, analysis[info])
#print("input the number you want to see")
#a = int(input())
#print(h[a][0])
#print(sample_system.getImportance(h[a][0], 1, path, a))
#print(sample_system.getImportance(h[a][0], 1))
datas = [path]
print(sample_system.highlights(datas, 1))
print(h[1][0])
print(sample_system.getPastValue(path, 1, h[1][0], 1))
print(sample_system.getPastValue(path, 1, h[1][0], 1))
#sample_system.analyzeOne(path,a, h[a][0], onlyS=True)
#r = sample_system.observeTraverse(path, a, h[a][0], onlyS=True,infinite=1True)
#print(extract(sample_system.saliency_map(h[a][0], 1), 5))
#print(extract(sample_system.saliency_map(h[a][0], 1, mode="value"), 5))
#print(extract(sample_system.fatal_map(path, a, h[a][0]), 5))
'''
commons = []
for i in range(8, len(h)-1):
    saliency_points = set(extract(sample_system.saliency_map(h[i][0], 1), 5))
    fatal_points = set(extract(sample_system.fatal_map(path, i, h[i][0]), 5))
    common = len(saliency_points.intersection(fatal_points))
    print(common/5)
    commons.append(common/5)
print(commons)
'''
#print(sample_system.saliency_map(h[a][0], 1, mode="value"))
#print(sample_system.fatal_map(path, a, h[a][0]))

#sample_system.tewari(path, a, h[a][0], 1)

#sample_system.tewari(path, a, h[a][0], 1, search=True)

'''
for i in range(len(h)):
    board, sNsa, bNsa, sv, bv, sVs, bVs = h[i]
    print(board)
    curPlayer = getCurrentPlayer(board[0])
    print("curPlayer", curPlayer)
    canonicalBoard = game.getCanonicalForm(board, curPlayer)
    print(sample_system.getPastActionProb(path, i, canonicalBoard, 1))
    print(sample_system.getPastActionProb(path, i, canonicalBoard, -1))
'''