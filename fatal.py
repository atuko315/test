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
import numpy as np

sample_s_path = '/home/student/PARL/benchmark/torch/AlphaZero/best_200.pth.tar'
sample_b_path = '/home/student/PARL/benchmark/torch/AlphaZero/saved_model/checkpoint_1.pth.tar'
encoded_weights = encode_weight(sample_b_path)
print("encoded")
buffer = load_buffer(sample_b_path)
print("buffer")
game = Connect4Game()

sample_system = System(game, sample_s_path, sample_b_path, turn=1)
#sample_system.playGame()
path = sorted(Path('./data').glob('*.history'))[-1]
h = load_data(path)
pboard = np.array(
[[ 0, 0, 0, 0, 0, 0, 0],
 [ 0, 1,-1, -1,-1, 0, 0],
 [ 0, 1,-1, 1, 1, 0, 0],
 [ 0, 1,-1, 1, 1, 0, 0],
 [ 0, 1, 0,-1, 1,-1, 0],
 [ 0, 1, 0,-1, 1, 1, 0]])

cboard = np.array(
[[ 0, 0, 0, 0, 0, 0, 0],
 [ 0, 1,-1, 1,-1, 0, 0],
 [ 0, 1,-1, 1, 1, 0, 0],
 [ 0, 1,-1, 1, 1, 0, 0],
 [ 0, 1, 0,-1, 1, 1, 0],
 [ 0, 1, 0,-1, 1,-1, 0]])
print(pboard)

#print(sample_system.saliency_map(pboard, 1))

#print(sample_system.removeStone(pboard, 40))
#print(sample_system.getHorizontalEnemy(board, 0, check=True))
#print(sample_system.getVerticalEnemy(board, 0, check=True))
#print(sample_system.getDistance(pboard, cboard, simple=True))
#print(sample_system.getDistance(pboard, cboard, simple=False))
#print(sample_system.remove_stone(pboard.copy(), 1, 1))
#print(sample_system.remove_random_stone(pboard,1,ex=[3]))

#print(sample_system.visualizeFatalStone(board))