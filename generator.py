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
from connectx_try import load_data, getCurrentPlayer, getStep, store_data
from random import uniform
from time import sleep

class DatasetGenerator(object):
    def __init__(self, strong_range, weak_range, strong_puct_range, weak_puct_range, s_path, b_path):
        self.s_path = s_path
        self.b_path = b_path
        self.supp, self.slow = strong_range
        self.wupp, self.wlow = weak_range
        self.spupp, self.splow = strong_puct_range
        self.wpupp, self.wplow = weak_puct_range
        self.game = Connect4Game()
    
    def setOfflineSystem(self, s_path, b_path, strong_timelimit, weak_timelimit, strong_puct, weak_puct):
        self.offSystem = System(self.game, s_path, b_path, turn=1, strong_timelimit=strong_timelimit,
                        weak_timelimit=weak_timelimit, strong_puct=strong_puct, weak_puct=weak_puct)
    
    def generateData(self, number):
        for i in range(number):
            strong_timelimit = uniform(self.slow, self.supp)
            weak_timelimit = uniform(self.wlow, self.wupp)
            strong_puct = uniform(self.splow, self.spupp)
            weak_puct = uniform(self.wplow, self.wpupp)
            system = System(self.game, self.s_path, self.b_path, turn=1, strong_timelimit=strong_timelimit,
                        weak_timelimit=weak_timelimit, strong_puct=strong_puct, weak_puct=weak_puct)
            system.playGame()
    
    def translateOffLine(self, number):
        '''
        generateData()と同時に用いる場合はnumberの値を同じにする
        '''
        for i in range(number):
            path = sorted(Path('./offdata').glob('*.history'))[-number-1]
            self.offSystem.train_offline(path)
    
    def labelFour(self, number, threshold, dir, short=10, middle=20):
        '''
        強い方で二つにわける
        その後弱い方で二つにおける
        '''
        datas = sorted(Path('./offdata').glob('*.history'))[-number-1:]
        self.offSystem.highlights(datas, 1, threshold=threshold) #ディレクトリを書き換えて
        name = ["trivial", "important"]
        for s in name:
            dirname = dir + "/label/" + s
            size = len(len(sorted(Path(dirname).glob('*.board'))))
            for i in range(size):
                path = sorted(Path(dirname).glob('*.board'))[-i-1]
                h = load_data(path)
                importance, board, branch, fpath= h
                step = getStep(board)
                imp = self.offSystem.getImportance(board, -1, fpath, step)
                new_h = (imp, board, branch, fpath, importance) 
                dir_path = "" # 格納先のパス
                if imp > threshold:
                    dir_path = dirname + "important"
                    
                else:
                    dir_path = dirname + "trivial"
                store_data(new_h, dir_path)

                if step < short:
                    store_data(dir_path+"/short")
                elif step < middle:
                    store_data(dir_path+"/middle")
                else:
                    store_data(dir_path+"/long")



        


        