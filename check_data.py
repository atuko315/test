# -*- coding: utf-8 -*-
"""
Created on Mon May 15 17:36:45 2023

@author: uguis
"""
from pathlib import Path
from connectx_try import load_data
import sys
import base64
import inspect
import os
import io
from submission_sample import SimpleAgent, dotdict, MCTS

from tqdm import tqdm
from parl.utils import logger
from connectx_try import encode_weight, load_buffer, System
from connect4_game import Connect4Game
from pathlib import Path
from connectx_try import load_data, getCurrentPlayer
sample_s_path = '/home/student/PARL/benchmark/torch/AlphaZero/best_200.pth.tar'
sample_b_path = '/home/student/PARL/benchmark/torch/AlphaZero/best_200.pth.tar'
encoded_weights = encode_weight(sample_b_path)
print("encoded")
buffer = load_buffer(sample_b_path)
print("buffer")
game = Connect4Game()

sample_system = System(game, sample_s_path, sample_b_path)


path = sorted(Path('./data').glob('*.history'))[-1]
#path = Path("/home/student/PARL/benchmark/torch/AlphaZero/data/20230526180235.history")
h = load_data(path)

#curPlayer = getCurrentPlayer(board)
#canonicalBoard= game.getCanonicalForm(board, curPlayer)
#print(sample_system.s_mcts.nn_agent.predict(canonicalBoard))
#sample_system.getPastActionProb(path, 0, canonicalBoard, 1)
#sample_system.getPastActionProb(path, 5, canonicalBoard, 1)
#sample_system.getPastActionProb(path, 10, canonicalBoard, 1)
print("student turn: ", h[len(h)-1])
for i in range(len(h)-1):
    print(h[i][0], h[i][3], h[i][4], i)
    print("-------------")