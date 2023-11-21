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
sample_b_path = '/home/student/PARL/benchmark/torch/AlphaZero/checkpoint_1.pth.tar'
encoded_weights = encode_weight(sample_b_path)
buffer = load_buffer(sample_b_path)
game = Connect4Game()


cycle = 2000
for _ in range(cycle):

    strong_timelimit = uniform(3, 5)
    weak_timelimit = uniform(0, 2)
    strong_puct = uniform(0.8, 1)
    weak_puct = uniform(0, 0.5)
    sample_system = System(game, sample_s_path, sample_b_path, turn=1, strong_timelimit=strong_timelimit,
                        weak_timelimit=weak_timelimit, strong_puct=strong_puct, weak_puct=weak_puct)
    sample_system.playGame()

