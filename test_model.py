# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:05:02 2023

@author: uguis
"""
import sys
import base64
import inspect
import os
import io
import numpy as np
from submission_sample import SimpleAgent, dotdict, MCTS
from tqdm import tqdm
from parl.utils import logger
from connect4_game import Connect4Game
import time
from datetime import datetime
import pickle
from connectx_try import load_buffer
game = Connect4Game()

# superior and basic is the path of the two models
#prepare superior model
mcts_args = dotdict({'nmMCTSSims': 1, 'cpuct': 1.0})
agent = SimpleAgent(game)
buffer = load_buffer('/home/student/PARL/benchmark/torch/AlphaZero/best_200.pth.tar')
agent.load_checkpoint(buffer)
mcts = MCTS(game, agent, mcts_args)


board = np.array([[0,  0, 0,  0, 0,  0,  0],
 [ 0,  0,  0,  0, 0, 0,  0],
 [0,  0, 0,  0,  0, -1,  0],
 [ 0, 0,  0,  0,  1, -1,  0],
 [0,  0, 0,  0, 1,  -1,  0],
 [ 0, 0, 0,  0, 1,  1,  0]])
print(agent.predict(board))