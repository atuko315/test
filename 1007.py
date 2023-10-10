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
from feature import DatasetManager
from random import uniform
import random
from time import sleep
from collections import defaultdict
from statistics import quantiles

sample_s_path = '/home/student/PARL/benchmark/torch/AlphaZero/best_200.pth.tar'
sample_b_path = '/home/student/PARL/benchmark/torch/AlphaZero/saved_model/checkpoint_1.pth.tar'

game = Connect4Game()
strong_timellimit = 5
weak_timelimit = 0.5
strong_puct = 1
weak_puct = 0.1
sample_system = System(game, sample_s_path, sample_b_path, turn=1, strong_timelimit=strong_timellimit,
                        weak_timelimit=weak_timelimit, strong_puct=strong_puct, weak_puct=weak_puct)

paths1 = sorted(Path('./label/important/important/short').glob('*.board'))
paths2 = sorted(Path('./label/important/trivial/short').glob('*.board'))
paths3 = sorted(Path('./label/trivial/important/short').glob('*.board'))
paths4 = sorted(Path('./label/trivial/trivial/short').glob('*.board'))

path = sorted(Path('./label/important/important/long').glob('*.board'))[-1]
imp, board, branch, fpath, importance = load_data(path)
print(sample_system.getImportantAction(board, 1, fpath, getStep(board), 5))
