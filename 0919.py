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
sample_b_path = '/home/student/PARL/benchmark/torch/AlphaZero/saved_model/checkpoint_1.pth.tar'

game = Connect4Game()
strong_timellimit = 5
weak_timelimit = 0.5
strong_puct = 1
weak_puct = 0.1
sample_system = System(game, sample_s_path, sample_b_path, turn=1, strong_timelimit=strong_timellimit,
                        weak_timelimit=weak_timelimit, strong_puct=strong_puct, weak_puct=weak_puct)
size1 = len(sorted(Path('./label/important/important/long').glob('*.board')))
size4 = len(sorted(Path('./label/trivial/trivial/long').glob('*.board')))
paths1 = sorted(Path('./label/important/important/long').glob('*.board'))[-size1: ]
paths2 = sorted(Path('./label/important/trivial/long').glob('*.board'))
paths3 = sorted(Path('./label/trivial/important/long').glob('*.board'))
paths4 = sorted(Path('./label/trivial/trivial/long').glob('*.board'))[-size4: ]
dataset1 = DatasetManager(game, paths1)
dataset4 = DatasetManager(game, paths4)
paths = sorted(Path('./offdata').glob('*.history'))
#例えば最後の三手から次の言ってをブレさせてもみんな同じ部分にたどり着けるのか？
#一個先は上位３つ
steps = [8, 10]
#steps = [2]
baseline = 3
analist = 1
#print(len(paths2), len(paths3))
print(len(paths1), len(paths2), len(paths3), len(paths4))

result = []
total_size = len(paths4)
print(f"total size: {total_size}")
for s in steps:
    print(f"step= {s}")
    ave_rate = 0
    ave_frate = 0
    ave_fdrate = 0
    pcount = 0
    for p in paths4:
        if pcount % 100 == 0:
            print(f"pcount={pcount}")
        pcount += 1
        imp, board, branch, fpath, importance = load_data(p)
        h = load_data(fpath)
        if len(h) - 2 - s < 0:
            continue
        tmp = h[len(h)-2]
        fboard, sNsa, bNsa, sv, bv, sVs, bVs = tmp
        #print(fboard)
        #reach = sample_system.detectFatalStone(fboard, reach=True, per_group=True)
        valid = game.getValidMoves(fboard, getCurrentPlayer(fboard))
        valid = [i  for i in range(len(valid)) if valid[i]]
        reach = []
        for a in valid:
            vboard = sample_system.add_stone(fboard.copy(), getCurrentPlayer(fboard), a)
            vf = sample_system.detectFatalStone(vboard, per_group=True)
            if vf:
                reach.extend(vf)
        
        #print("reach")
        #print(reach)
        #あくまでも良さを図る指標であり、確率の形をするひつようはない　ｆｄを４で悪法にする
        sboard, sNsa, bNsa, sv, bv, sVs, bVs = h[len(h) - 2 - s]
        valid = game.getValidMoves(sboard, getCurrentPlayer(sboard))
        valid = [i  for i in range(len(valid)) if valid[i]]
        l = len(valid) if len(valid) < baseline else baseline
        counts = sample_system.getPastCount(fpath, getStep(sboard), sboard, analist)
        counts = np.argsort(np.array(counts))
       
        counts = [c for c in counts if c in valid]
       
        counts = counts[-l:]
        '''
        print("sboard")
        print(sboard)
        print("fboard")
        print(fboard)
        print("---------------")
        '''
        bboards = []
        for c in counts:
            bboards.append(sample_system.add_stone(sboard.copy(), getCurrentPlayer(sboard), c))
        
        size = len(bboards)
        fdcount = 0
        fcount = 0
        count = 0
        #fatal_group = {}
        for b in bboards:
            #print("before")
            #print(b)
            hot = sample_system.detectHotState(b, analist, fpath, getStep(sboard), toend=True)
            #print("after")
            #print(hot)
            if hot[1] != None:
                
                end = game.getGameEnded(hot[0], getCurrentPlayer(hot[0]))
                if end:
                    #print(hot[0])
                    count += 1
                    fatal = sample_system.detectFatalStone(hot[0], per_group=True)
                    #print("fatal")
                    #print(fatal)
                    fu = np.unique(fatal.copy()).tolist() if fatal else [-1]
                    ru = np.unique(reach.copy()).tolist() if reach else [-2]
                    if len(set(ru)) > 0:
                        fdcount += (len(set(fu) & set(ru))) / 4
                    #print(fu, ru, len(set(fu) & set(ru)))
                    if fatal:
                        for g in fatal:
                            for i in range(len(reach)):
                                r = reach[i]
                                if set(r).issubset(set(g)):
                                    fcount += 1
                        '''
                        gs = str(g)
                        #print(fatal_group.keys())
                        if gs not in fatal_group.keys():
                            fatal_group[gs] = 1
                        else:
                            fatal_group[gs] += 1
                        '''
                        

        rate = count / size
        frate = fcount / count if count > 0 else 0
        fdrate = fdcount / count if count > 0 else 0

        ave_rate += rate
        ave_frate += frate
        ave_fdrate += fdrate
        
    
    ave_rate /= total_size
    ave_frate /= total_size
    ave_fdrate /= total_size #ここなんとか
    result.append([s, ave_rate, ave_frate, ave_fdrate])

for r in result:
    print(f"{r[0]},{r[1]},{r[2]},{r[3]}")

#最後だけのバリエーションは同一とみなす
        

