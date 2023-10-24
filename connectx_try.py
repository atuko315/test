# -*- coding: utf-8 -*-
"""
Created on Sun May 14 23:22:16 2023

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
from time import sleep
from datetime import datetime
import pickle
from collections import defaultdict
import collections
from random import choice
import bisect
import torch
import torch.nn as nn
import math



#mcts_args = dotdict({'numMCTSSims': 800, 'cpuct': 1.0})
#mcts_weak_args = dotdict({'numMCTSSims': 0, 'cpuct': 0.1})
#weak_timelimit = 1
# arg一応この最終版でいいかな？？
def write_data(history, offline=False, p=False):
  now = datetime.now()
  os.makedirs('./data/', exist_ok=True)
  path = './data/{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(
      now.year, now.month, now.day, now.hour, now.minute, now.second)
  if offline == True:
      os.makedirs('./offdata/', exist_ok=True)
      path = './offdata/{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(
          now.year, now.month, now.day, now.hour, now.minute, now.second)
  if p == True:
      os.makedirs('./pdata/', exist_ok=True)
      path = './pdata/{:04}{:02}{:02}{:02}{:02}{:02}.history'.format(
          now.year, now.month, now.day, now.hour, now.minute, now.second)
  with open(path, mode='wb') as f:
    pickle.dump(history, f)

def store_data(data, dirname):
    path = './'+dirname
    now = datetime.now()
    path += '/{:04}{:02}{:02}{:02}{:02}{:02}.board'.format(
        now.year, now.month, now.day, now.hour, now.minute, now.second)
    with open(path, mode='wb') as f:
      pickle.dump(data, f)
def load_data(path):
    with path.open(mode='rb') as f:
        return pickle.load(f)

def encode_weight(model_path):
    with open(model_path, 'rb') as f:
        raw_bytes = f.read()
        encoded_weights = base64.encodebytes(raw_bytes)
    
    return encoded_weights

def load_buffer(model_path):
    encoded_weights = encode_weight(model_path)
    decoded = base64.b64decode(encoded_weights)
    buffer = io.BytesIO(decoded)
    return buffer

def getCurrentPlayer(board):
    sample = board.flatten()
    #print(sample)
    f_number = np.count_nonzero(sample == 1)
    b_number = np.count_nonzero(sample == -1)
    #print(s_number, b_number)
    
    if f_number > b_number:
        return -1
    else:
        return 1
def getStep(board):
    sample = board.flatten()
    return np.count_nonzero(sample != 0)



def yesOrNo():
    answer = input()
    while answer != "y" and answer != "n":
        answer = input()
    return answer

def detectSequence(data, threshold):
    depict = [0] * len(data)
    i = 0
    while i < len(data):
        if i + threshold - 1 > len(data) - 1:
            break
        if data[i] != data[i+threshold-1]:
            
            i += 1
            continue
        else:
            
            count = 0
            for j in range(i, i+threshold):
                if data[i] != data[j]:
                    i = j - 1
                    break
                count += 1
            if count == threshold:
                depict[i+1] = -1
                i += threshold - 1
                if i == len(data) - 1:
                   
                    break
                while True:
                    if data[i] == data[i+1]:
                        count += 1
                        i += 1
                        if i == len(data) - 1:
                            break
                    else:
                        break
                if i < len(data) - 1:
                    depict[i+1] = 1
                i -= 1
        i += 1
        
    return depict
    point = []
    i = 0
    while i < len(data):
        if i + threshold - 1 > len(data) - 1:
            break
        if data[i] != data[i+threshold-1]:
            i += 1
            continue
        else:
            count = 0
            for j in range(i, i+threshold):
                if data[i] != data[j]:
                    i = j - 1
                    break
                count += 1
                print(j, count)
            if count == threshold:
                i += threshold - 1
                if i == len(data) - 1:
                   
                    break
                while True:
                    print(i)
                    if data[i] == data[i+1]:
                        count += 1
                        i += 1
                        if i == len(data) - 1:
                            break
                    else:
                        break
                if i < len(data) - 1:
                    point.append(i+1)
                i -= 1
        i += 1
        
    return point

def extract(data, n, reverse=False):
    data = np.array(data)
    print(data.flatten())
    print(np.argsort(data.flatten()))
    index_list = np.argsort(data.flatten())
    if reverse == True:
        return index_list[: n]
    else:
        return index_list[-n:]
    
def saliency(pp, cp):
    #元、取り除いた後
    #tensorにして
    #vは片方ーに
    pp = torch.from_numpy(pp.astype(np.float32)).clone()
    cp = torch.from_numpy(cp.astype(np.float32)).clone()
    
    #loss = nn.CrossEntropyLoss()
    loss = nn.MSELoss()
    saliency = loss(pp, cp)
    return abs(saliency.to('cpu').detach().numpy().copy().tolist())

def is_playable(board, n):
    height, width = len(board), len(board[0])
    i = int(n / height)
    j = n % height
    if i == height - 1:
        return True
    
    if board[i+1][j] != 0:
        return True
    
    return False



     
class System(object):
    def __init__(self, game, superior, basic, turn=1, strong_timelimit= 5, weak_timelimit=1,
                 strong_puct=1, weak_puct=0.1):
        self.strong_timelimit = strong_timelimit
        self.weak_timelimit = weak_timelimit
        self.strong_puct = strong_puct
        self.weak_puct = weak_puct
        self.mcts_args = dotdict({'numMCTSSims': 800, 'cpuct': strong_puct})
        self.mcts_weak_args = dotdict({'numMCTSSims': 800, 'cpuct': weak_puct})
        self.game = game
        if turn != None:
            self.turn = turn
        # superior and basic is the path of the two models
        #prepare superior model
        self.superior = SimpleAgent(self.game)
        s_buffer = load_buffer(superior)
        self.s_link = superior
        self.superior.load_checkpoint(s_buffer)
        self.s_mcts = MCTS(game, self.superior, self.mcts_args)
        
        self.b_link = basic
        self.basic = SimpleAgent(self.game)
        b_buffer = load_buffer(basic)
        self.basic.load_checkpoint(b_buffer)
        self.b_mcts = MCTS(game, self.basic, self.mcts_weak_args)
        
        self.setting = (
            lambda x: np.argmax(self.s_mcts.getActionProb(x, temp=0)),
            lambda x: np.argmax(self.b_mcts.getActionProb(x, temp=0)),
            self.game)
        
        self.data = [] #v of tearcher, v of student,
        # n of tearcher at that point, n of student at that point
    def saliency_map(self, board, analist, mode="policy",path=-1, step=-1):
        #countにするのは真ん中に不利なので今回policyはニューロだけに
        height, width = self.game.getBoardSize()
        boards = [self.removeStone(board.copy(), i) for i in range(height*width)]
        if analist == 1:
            agent = self.s_mcts
        else:
            agent = self.b_mcts
        
        #元が前
        if mode == "value":
            #手番の問題で裏返し
            saliencies = saliencies = [saliency(
                -agent.nn_agent.predict(self.game.getCanonicalForm(board, getCurrentPlayer(board)))[1],
                agent.nn_agent.predict(self.game.getCanonicalForm(boards[i], getCurrentPlayer(boards[i])))[1]
                ) for i in range(height*width)]   
        else:
            saliencies = [saliency(
                agent.nn_agent.predict(self.game.getCanonicalForm(board, getCurrentPlayer(board)))[0],
                agent.nn_agent.predict(self.game.getCanonicalForm(boards[i], getCurrentPlayer(boards[i])))[0]
                ) for i in range(height*width)]   
        saliencies = np.array(saliencies).reshape(height, width)
        
        return saliencies
        
        #if type(path) == int and step == -1:
    
    def getImportantAction(self, board, analist, path, step, number):
        player = getCurrentPlayer(board)
        valids = self.game.getValidMoves(board, player)
        #とりあえず最下位を渡す
        next_values = []

        for a in range(self.game.getActionSize()):
            if not valids[a]:
                next_values.append(-1)
                continue

            next_board, _ = self.game.getNextState(board.copy(), player, a)

            next_value = -self.getPastValueNoModification(path, step, next_board, analist)
            next_values.append(next_value)
        
        ranking = np.argsort(next_values)
        ranking = ranking[::-1]
        
        valids = [i  for i in range(len(valids)) if valids[i]]
        if len(valids) < number+1:
            number = len(valids) - 1 # とりあえず最下位を渡す
        ranking = [i for i in ranking if i in valids]
       
        return ranking[number]
            


    def getImportance(self, board, analist, path=-1, step=-1, baseline=1):
        player = getCurrentPlayer(board)
        past_mode = (type(path) != int and (step) != -1)
        valids = self.game.getValidMoves(board, player)
        next_values = []
        if not past_mode:
            agent = self.s_mcts if analist == 1 else self.b_mcts

        for a in range(self.game.getActionSize()):
            if not valids[a]:
                continue
            next_board, _ = self.game.getNextState(board.copy(), player, a)
            #print(type(next_board))
            if past_mode:
                next_value = -self.getPastValueNoModification(path, step, next_board, analist)
            else:
                next_value = -agent.search(self.game.getCanonicalForm(next_board, -player))
            next_values.append(next_value)
        
        next_values.sort(reverse=True)
        print(next_values[:2])
        if len(next_values) < baseline:
            return abs(next_values[0] - next_values[-1])
        else:
            return abs(next_values[0] - next_values[baseline])


    
    def getMyImportance(self, board, analist, path=-1, step=-1):
        #　一番上から第3四分位数までの分散
        player = getCurrentPlayer(board)
        past_mode = (type(path) != int and (step) != -1)
        valids = self.game.getValidMoves(board, player)
        next_values = []
        if not past_mode:
            agent = self.s_mcts if analist == 1 else self.b_mcts

        for a in range(self.game.getActionSize()):
            if not valids[a]:
                continue
            next_board, _ = self.game.getNextState(board.copy(), player, a)
            #print(type(next_board))
            if past_mode:
                next_value = -self.getPastValueNoModification(path, step, next_board, analist)
            else:
                next_value = -agent.search(self.game.getCanonicalForm(next_board, -player))
            next_values.append(next_value)
        
        next_values.sort(reverse=True)
        q3 = math.ceil(np.percentile([i for i in range(len(next_values))], 75))

        return np.var(next_values[:q3])

    
    def highlights(self, datas, dirname, analist, context_length=1, minimum_gap=0, 
                   budget=5, threshold=0.6):
        summary = [] # importance の昇順
        #mctsは後で考える
        min_importance = 0
        for sim_number in range(len(datas)):
            path = datas[sim_number]
            h = load_data(path)
            self.turn = h[len(h)-1][0]
            boards = [h[step][0] for step in range(len(h)-1)]
            step = 0
            while step + context_length < len(h) -1:
                print("step: ", step)
                board = boards[step]
                print(board)

                importance = self.getImportance(board, analist, path, step, baseline=1)
                print(importance)
                branch = boards[step-context_length: step+context_length+1]
                sample = (importance, board, branch, path)
                if importance > threshold:
                    store_data(sample, dirname+"/important")
                else:
                    store_data(sample, dirname+"/trivial")
                if importance > min_importance or len(summary) <= budget:
                    #branch = boards[step-context_length: step+context_length+1]
                    if len(summary) == budget:
                        branch = branch[1: ]
                    summary.append((importance, branch))
                    step += minimum_gap + context_length
                else:
                    step += 1

                summary.sort(key = lambda x: x[0])
                min_importance = summary[-1][0]
        
        summary = [summary[x][1] for x in range(len(summary))]
        return summary
    
    def myHighlights(self, datas, dirname="label", context_length=0, minimum_gap=0, 
                   budget=5, sthreshold=0.6, wthreshold=0.13, short=10, middle=20):
        '''
        [0.0030262592179335, 0.12491765714907145, 0.31279777024276917]
        [0.0012928681478933393, 0.1464111499397758, 0.17491868566599517]
        0.0030262592179335と0.17491868566599517を採用
        0.004 と0.18
        context_length, minimum_gapを０にすると普通のラベリングになる
        short, middleはもうちょい考えるぞ
        '''
        summary = [] # importance の昇順
        #mctsは後で考える
        min_importance = 0
        for sim_number in range(len(datas)):
            path = datas[sim_number]
            h = load_data(path)
            self.turn = h[len(h)-1][0]
            boards = [h[step][0] for step in range(len(h)-1)]
            step = 0
            while step + context_length < len(h) -1:
                save_path = dirname
                print("step: ", step)
                board = boards[step]
                print(board)

                simp = self.getMyImportance(board, 1, path, step)
                wimp = self.getMyImportance(board, -1, path, step)
                branch = boards[step-context_length: step+context_length+1]
                sample = (simp, board, branch, path, wimp)

                if simp > sthreshold:
                    save_path = os.path.join(save_path, "important")
                else:
                    save_path = os.path.join(save_path, "trivial")

                if wimp > wthreshold:
                    save_path = os.path.join(save_path, "important")
                else:
                    save_path = os.path.join(save_path, "trivial")
                
                if step < short:
                    save_path = os.path.join(save_path, "short")
                elif step < middle:
                    save_path = os.path.join(save_path, "middle")
                else:
                    save_path = os.path.join(save_path, "long")

                store_data(sample, save_path)  
                # trajはsimp優先で  
                if simp > min_importance or len(summary) <= budget:
                    #branch = boards[step-context_length: step+context_length+1]
                    if len(summary) == budget:
                        branch = branch[1: ]
                    summary.append((simp, wimp, branch))
                    step += minimum_gap + context_length
                else:
                    step += 1

                summary.sort(key = lambda x: x[0])
                min_importance = summary[-1][0]
        
        summary = [summary[x][1] for x in range(len(summary))]
        return summary
    
    def getDifference(self, pboard, cboard):
        #単純に何個違うか
        pboard = np.array(pboard).reshape(1, -1)
        cboard = np.array(cboard).reshape(1, -1)
        compare = (pboard == cboard)[0].tolist()
        return compare.count(False)
    
    def getDistance(self, pboard, cboard, simple=False):
        height, width = self.game.getBoardSize()
        #canonicalに
        #0の扱いはTrueかFalseか
        pf = np.array(pboard).reshape(1, -1)
        cf = np.array(cboard).reshape(1, -1)
        count = height*width - pf.tolist().count(0)
        if height*width - cf.tolist().count(0) > count:
            count = height*width - cf.tolist().count(0) 
        
        diff = (pf == cf)[0].tolist().count(False)
        #print(diff)
        simpleDistance = diff/count
        if simple:
            return simpleDistance
        
        pa = np.array([self.judgeActive(pboard, i) for i in range(height*width)])
        ca = np.array([self.judgeActive(cboard, i) for i in range(height*width)])
        #print(pa==ca)
        adiff = (pa == ca).tolist().count(False)
        activeDistance = adiff
        #print(activeDistance)
        return simpleDistance * (1 + 0.1 * activeDistance)   

    def remove_stone(self, board, player, column):
        #コピーで
        height, width = self.game.getBoardSize()
        available_idx, = np.where(board[:, column] != 0)
        #print("ava", available_idx)
        if len(available_idx) == 0 or board[available_idx[0]][column] != player:
            raise ValueError(
                "Can't remove column %s on board %s" % (column, self))
        #print(board[available_idx[0]][column])

        board[available_idx[0]][column] = 0
        return board
    
    def remove_random_stone(self, board, player, ex=-1):
        height, width = self.game.getBoardSize()
        available_idx = []
        for c in range (width):
            tmp = np.where(board[:, c] != 0)
            #print(tmp)
            available_idx.append(tmp)
        a = -1
        '''
        print(a)
        print(available_idx)
        print(available_idx[a][0])
        print("len",len(available_idx[a][0]))
        '''
        if len(available_idx[a][0])!=0:
                print("c", board[available_idx[a][0][0]][a])
        while a == -1 or len(available_idx[a][0]) == 0 or board[available_idx[a][0][0]][a]!= player:
            #print(a)
            if type(ex) != int:
                #print("ex active")
                #print(ex)
                #print(a in ex)
                a = choice([ c for c in range (width)])
                while a in ex:
                    print("ex")
                    a = choice([ c for c in range (width)])
            else:        
                a = choice([ c for c in range (width)])
            
            #print(available_idx[a][0])
            #print("len",len(available_idx[a][0]))
        
        return self.remove_stone(board.copy(), player, a), a

    def add_stone(self, board, player, column, number=False):
        "Create copy of board containing new stone."

        height, width = self.game.getBoardSize()
        available_idx = np.where(board[:, column] == 0)
        if len(available_idx[0]) == 0:
            raise ValueError(
                "Can't play column %s on board %s" % (column, self))

        board[available_idx[0][len(available_idx[0])-1]][column] = player

        if number:
            return board, available_idx[0][len(available_idx[0])-1]*width+column
        return board

    def add_random_stone(self, board, player, ex=-1):
        height, width = self.game.getBoardSize()
        available_idx = np.array([])
        for c in range (width):
            tmp = np.where(board[:, c] == 0)
            np.append(available_idx, tmp, axis=0)
            
        a = choice([ c for c in range (width)])
        while len(available_idx[a]) == 0:
            if type(ex) != int:
                if a in ex:
                    continue
            a = choice([ c for c in range (width)])
        
        return self.add_stone(board.copy(), player, a), a
    
    def tewari(self, path, step, board, analist, ex=-1, search=False):
        #取り除いてる方の視点
        print(path)
        h = load_data(path)
        tmp = h[step] # 注目する部分
        self.turn = h[len(h)-1][0]
        kboard, sNsa, bNsa, sv, bv, sVs, bVs = tmp
        player = -getCurrentPlayer(board)
        ex = self.detectAction(h[step-1][0], h[step][0])
        print("ex:", ex)
        print("current board")
        print(board)
        print("past board")
        print(h[step-1][0])
        vboard, a= self.remove_random_stone(board.copy(), player, ex=[ex])
        print("remove:", a)
        print("vboard")
        print(vboard)
        
        if search == True:
            agent = self.s_mcts if analist == 1 else self.b_mcts
            
            cp, cv = agent.nn_agent.predict(self.game.getCanonicalForm(board, -player))
            pp, pv = agent.nn_agent.predict(self.game.getCanonicalForm(h[step-1][0], player))
            vp, vv = agent.nn_agent.predict(self.game.getCanonicalForm(vboard, player))
            print(-cv, pv, vv)
            prob = agent.getActionProb(self.game.getCanonicalForm(vboard, player), temp=0, vc=True)
            print(prob)
            action = np.argmax(prob)
        else:
            cv = self.getPastValue(path, step, board, analist)
            pv = self.getPastValue(path, step, h[step-1][0], analist)
            vv = self.getPastValue(path, step, vboard, analist)
            print(cv, pv, vv)
            valids = self.game.getValidMoves(vboard, player)
            counts = self.getPastCount(path, step, vboard, analist)
            print(counts)
            action = np.argmax(self.getPastActionProb(path, step, vboard, 
                                                        analist, counts = counts))
        
        print("original: ", a, end=" ")
        print("another: ", action)

        
        

    
    def judgeActive(self, board, n):
        '''
        Parameters
        ----------
        board : TYPE
            盤面
        n : TYPE
            番号

        Returns
        -------
        TYPE
           　activeならTrue.

        '''
        height, width = self.game.getBoardSize()
        player = board[int(n/width)][n%width]
        if player == 0:
            return False
        
        if player == -1:
            board  = self.game.getCanonicalForm(board, -1)
        
        
        #nの石にあわせてください
        height, width = self.game.getBoardSize()
        win_length = self.game._base_board.win_length
        enemy = self.getHorizontalEnemy(board, n)
        
        l = enemy[bisect.bisect(enemy, (n%width))-1]
        r = enemy[bisect.bisect_right(enemy, (n%width))]
        #print("horizon", l,r, int(n%width))
        if r - l > win_length:
            return True
        enemy = self.getVerticalEnemy(board, n)
        l = enemy[bisect.bisect(enemy, int(n/width))-1]
        r = enemy[bisect.bisect(enemy, int(n/width))]
        #print("vertical", l, r, int(n/width))
        if r - l > win_length:
            return True
        
        return self.getDiagonalEnemy(board, n)
        
            
        
    
    def getHorizontalEnemy(self, board, n, check=False):
        if check == True:
            height, width = self.game.getBoardSize()
            player = board[int(n/width)][n%width]
            if player == 0:
                return [-1, len(board[0])]
            elif player == -1:
                board  = self.game.getCanonicalForm(board, -1)
        
        h = int(n/len(board[0]))
        lane = board[h]
        enemy = [ i for i in range(len(board[0])) if board[h][i]==-1]
        enemy.append(-1)
        enemy.append(len(board[0]))
        enemy.sort()
        return enemy
    
    def getVerticalEnemy(self, board, n, check=False):
        if check == True:
            height, width = self.game.getBoardSize()
            player = board[int(n/width)][n%width]
            if player == 0:
                return [-1, len(board)]
            elif player == -1:
                board  = self.game.getCanonicalForm(board, -1)
        
        h = int(n%len(board[0]))
        board = board.transpose()
        enemy = [ i for i in range(len(board[0])) if board[h][i]==-1]
        enemy.append(-1)
        enemy.append(len(board[0]))
        enemy.sort()
        return enemy
    
    def getDiagonalEnemy(self, board, n, check=False):
        if check == True:
            height, width = self.game.getBoardSize()
            player = board[int(n/width)][n%width]
            if player == 0:
                return []
            elif player == -1:
                board  = self.game.getCanonicalForm(board, -1)
        
        #右下
        height, width = self.game.getBoardSize()
        h = int(n/width)
        w = n % width
        number = w
        record = [-1]
        if h < w:
            number = h
        if w < h:
            sw = 0
            sh = h - w
        else:
            sw = w - h
            sh = 0
        while True:
            if sw > width -1 or sh > height -1:
                break
            record.append(board[sh][sw])
            sw += 1
            sh += 1
        
        record.append(-1)
        #print(record)
        enemy = [i for i in range(len(record)) if record[i]==-1]
        #print(enemy)
        win_length = self.game._base_board.win_length
        l = enemy[bisect.bisect(enemy, number)-1]
        r = enemy[bisect.bisect(enemy, number)]
        #print(l,r,number)
        if r - l > win_length:
            return True
        
        #左下
        if (width -1 - w< h):
            number = width -1 - w
            sw = width -1 
            sh = h - (width - 1 - w)
        else:
            number = h
            sw = w + h
            sh = 0
        
        record = [-1]
        while True:
            if sh > height - 1 or sw < 0 :
                break
            record.append(board[sh][sw])
            sw -= 1
            sh += 1
        record.append(-1)
        #print(record)
        enemy = [i for i in range(len(record)) if record[i]==-1]
        #print(enemy)
        l = enemy[bisect.bisect(enemy, number)-1]
        r = enemy[bisect.bisect(enemy, number)]
        #print(l,r,number)
        if r - l > win_length:
            return True
        return False
    
    def getLatest(self, path, step):
        h = load_data(path)
        tmp = h[step] # 注目する部分
        self.turn = h[len(h)-1][0]
        cboard = h[step][0]
        pboard = h[step-1][0]
        return self.detectAction(pboard, cboard)
  
    def detectAction(self, pboard, cboard):
        pboard = np.array(pboard).reshape(1, -1)
        cboard = np.array(cboard).reshape(1, -1)
        compare = (pboard == cboard)[0].tolist()
        action = compare.index(False)
        return action % self.game.getActionSize()
        
    def getActionNumber(self):
        action = input()
        
        while (action.isdecimal() == False or int(action) < 0 or 
        int(action) > self.game.getActionSize()):
            action = input()
        
        return int(action)
    
    
        
    
    def with_number_is_horizontal_winner(self, player_pieces, reach=False, per_group=False):
        #横向き
        object_length = self.game._base_board.win_length -1 if reach else self.game._base_board.win_length
        fatal = []
        run_lengths = [
            player_pieces[:, i:i + self.game._base_board.win_length].sum(axis=1)
            for i in range(len(player_pieces) - self.game._base_board.win_length + 2)
        ]

        #for i in range(len(player_pieces) - self.game._base_board.win_length + 2):
        run_lengths = np.array(run_lengths)
        
        run_lengths = run_lengths.transpose()
        
        height, width = self.game.getBoardSize()
        
        for i in range(height):
            #print(i)
            for j in range(len(player_pieces) - self.game._base_board.win_length + 2):
                if abs(run_lengths[i][j]) >= object_length:
                    #print(i, j)
                    seq = [x+j+width*i for x in range(object_length)]
                    if reach:
                        if j+object_length < width:
                            #print("ok1")
                            if player_pieces[i][j+object_length] == 0 and (i==height-1 or player_pieces[i+1][j+object_length]!= 0):
                                fatal.append(object_length+j+width*i)
                        if j -1 >= 0:
                            #print("ok2")
                            if player_pieces[i][j-1] == 0 and (i==height-1 or player_pieces[i+1][j-1]!= 0):
                                fatal.append(j-1+width*i)
                    else:
                        fatal.append(seq)
        #print(fatal)       
        if len(fatal) == 0:
            return None
        
        if per_group:
            return fatal
        fatal = np.unique(fatal)
        #print(fatal)
        
        
        return fatal.tolist()
        #return max([x.max() for x in run_lengths]) >= self.game._base_board.win_length
    def with_number_is_vertical_winner(self, player_pieces, reach=False, per_group=False):
        object_length = self.game._base_board.win_length -1 if reach else self.game._base_board.win_length
        player_pieces = player_pieces.transpose()
        
        fatal = []
        run_lengths = [
            player_pieces[:, i:i + self.game._base_board.win_length].sum(axis=1)
            for i in range(len(player_pieces) - self.game._base_board.win_length + 2)
        ]
        
        run_lengths = np.array(run_lengths)
        #print(run_lengths.transpose())
        #run_lengths = run_lengths.transpose()
        
        height, width = self.game.getBoardSize()
        for i in range(height - self.game._base_board.win_length+2):
            #print(i)
            for j in range(width):
                if abs(run_lengths[i][j]) >= object_length:
                    seq = [j+width*(x+i) for x in range(self.game._base_board.win_length)]
                    if reach:
                
                        if i <= height - self.game._base_board.win_length and i != 0:
                            
                            if player_pieces[j][i] == 0:
                                fatal.append(j+width*(i))
                        
                    else:
                        fatal.append(seq)
                    
        if len(fatal) == 0:
            return None
        
        if per_group:
            return fatal
        fatal = np.unique(fatal)
        #print(fatal)
        
        return fatal.tolist()
    
    def with_number_is_diagonal_winner(self, player_pieces, reach=False, per_group=False):
        """Checks if player_pieces contains a diagonal win."""
        object_length = self.game._base_board.win_length -1 if reach else self.game._base_board.win_length
        fatal = []
        win_length = self.game._base_board.win_length
        height, width = self.game.getBoardSize()
        
        for i in range(len(player_pieces) - object_length + 1):
            for j in range(len(player_pieces[0]) - object_length + 1):
                
                
                #print(([player_pieces[i + x][j + x] == -1 for x in range(object_length)]))
                
                if (all([player_pieces[i + x][j + x] == 1 for x in range(object_length)]) 
                    or all([player_pieces[i + x][j + x] == -1 for x in range(object_length)])):
                    
                    seq = [(i+x)*width+j+x for x in range(win_length)]
                    if reach:
                        if j+object_length < width and i+object_length < height:
                            if player_pieces[i+object_length][j+object_length] == 0 and (i+object_length == height-1 or player_pieces[i+1+object_length][j+object_length] != 0):
                                fatal.append(j+object_length+width*(object_length+i))
                        
                        if j-1 >= 0 and i-1 >= 0:
                            if player_pieces[i-1][j-1] == 0 and (i-1 == height-1 or player_pieces[i-1+1][j-1] != 0):
                                fatal.append((j-1)+width*(i-1))
                    else:
                        fatal.append(seq)
                    
            for j in range(object_length - 1, len(player_pieces[0])):
                
                if (all([player_pieces[i + x][j - x] == 1 for x in range(object_length)])
                    or all([player_pieces[i + x][j - x] == -1 for x in range(object_length)])):
                    
                    seq = [(i+x)*width+j-x for x in range(win_length)]
                    if reach:
                        if j-object_length >= 0 and object_length+i < height:
                            if player_pieces[i+object_length][j-object_length] == 0 and (i+object_length == height-1 or player_pieces[i+1+object_length][j-object_length] != 0):
                                print((j-object_length)+width*(object_length+i))
                                fatal.append((j-object_length)+width*(object_length+i))
                        
                        if j+1 < width and i-1 >= 0:
                            if player_pieces[i-1][j+1] == 0 and (i-1 == height-1 or player_pieces[i-1+1][j+1] != 0):
                                print((j+1)+width*(i-1))
                                fatal.append((j+1)+width*(i-1))
                            
                    else:
                        fatal.append(seq)
        if len(fatal) == 0:
            return None 
        
        if per_group:
            return fatal
        fatal = np.unique(fatal)
        #print(fatal)
                  
        return fatal.tolist()
    
    def detectCheckmate(self, board, step=1):
        '''
        実質詰んでる状況を取り出す
        Trueが詰み
        '''
        curPlayer = getCurrentPlayer(board)
        reach_stone = self.detectFatalStone(board, reach=True)
        if len(reach_stone) > 1:
            return True
        
        return False


    def removeStone(self, board, n):
        rboard = board.copy()
        height, width = self.game.getBoardSize()
        rboard[int(n/width)][n%width] = 0
        return rboard
    
    def removeFatalStone(self, board, random=True):
        fatal = self.detectFatalStone(board)
        height, width = self.game.getBoardSize()
        visual = [0 if i not in collections.Counter(fatal).keys() else collections.Counter(fatal)[i]
                  for i in range(height * width)]
        most_fatal = [i for i in range(len(visual)) if visual[i] == max(visual)]
        f = choice(most_fatal)
        if random == False:
            print(most_fatal)
            print("choose the stone to remove")
            a = input()
            while a.isdecimal() == False or int(a) not in most_fatal:
                a = input()
            f = int(a)
        
        return self.removeStone(board, f)
    
    def removePotentiallyFatalStone(self, path, step, board, onlyS=False, change=False, random=True):
        visual = self.observeTraverse(path, step, board, onlyS=onlyS, change=change,
                                     infinite=True)
        most_fatal = [i for i in range(len(visual)) if visual[i] == max(visual)]
        f = choice(most_fatal)
        if random == False:
            print(most_fatal)
            print("choose the stone to remove")
            a = input()
            while a.isdecimal() == False or int(a) not in most_fatal:
                a = input()
            f = int(a)
        
        return self.removeStone(board, f)
    def fatal_map(self, path, step, board, onlyS=True, change=False):
        h = load_data(path)
        tmp = h[step] # 注目する部分
        tboard, sNsa, bNsa, sv, bv, sVs, bVs = tmp
        
        curPlayer = getCurrentPlayer(board)
        valids = self.game.getValidMoves(
            self.game.getCanonicalForm(board, curPlayer), 1)
        futures = []
       
        
        for action in range(self.game.getActionSize()):
            if valids[action] == False:
                futures.append(-1)
            else:
                vboard, vplayer = self.game.getNextState(board, curPlayer,
                                                         action)
                vvboard = self.simpleSimulate(path, step, vboard,onlyS=onlyS
                                                   ,stepThreshold=5,
                                                   change=change, infinite=True)
                
                futures.append(vvboard)

        fatal = []
        for action in range(self.game.getActionSize()):
            if type(futures[action]) != int:
                f = self.detectFatalStone(futures[action])
                if f:
                    fatal.extend(f)

        height, width = self.game.getBoardSize()
        visual = [0 if i not in collections.Counter(fatal).keys() else collections.Counter(fatal)[i]
                    for i in range(height * width)]
        visual = np.array(visual).reshape(height, width)
        #print(visual) 
        return visual
        
    
        
        
    
    def detectFatalStone(self, board, reach=False, per_group=False):
        #二重にかかってるのはとりあえず無視
        fatal = []
        f1 = self.with_number_is_horizontal_winner(board, reach=reach, per_group=per_group)
        #print(f1)
        if f1:
            fatal.extend(f1)
        f2 = self.with_number_is_vertical_winner(board, reach=reach, per_group=per_group)
        #print(f2)
        if f2:
            fatal.extend(f2)
        f3 = self.with_number_is_diagonal_winner(board, reach=reach, per_group=per_group)
        #print(f3)
        if f3:
            fatal.extend(f3)
        
        if len(fatal) == 0:
            return None 
        
        if per_group:
            return fatal
        #print(fatal)
        fatal = np.array(fatal)
        fatal = np.unique(fatal.flatten())
        #print(fatal)
        return fatal.tolist()
    
    def visualizeFatalStone(self, board):
        fatal = self.detectFatalStone(board)
        height, width = self.game.getBoardSize()
        visual = ["x" if i in fatal else board[int(i/width)][i%width] 
                  for i in range(height * width)]
        visual = np.array(visual).reshape(height, width)
        print(visual)
    def reset_mcts(self):
        self.data = []
        #あと一個
        self.s_mcts.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.s_mcts.Nsa = {}  # stores #times edge s,a was visited
        self.s_mcts.Ns = {}  # stores #times board s was visited
        self.s_mcts.Ps = {}  # stores initial policy (returned by neural net)
        self.s_mcts.Es = {}  # stores game.getGameEnded ended for board s
        self.s_mcts.Vs = {}  # stores game.getValidMoves for board s
        self.s_mcts.V = {}
        
        self.b_mcts.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.b_mcts.Nsa = {}  # stores #times edge s,a was visited
        self.b_mcts.Ns = {}  # stores #times board s was visited
        self.b_mcts.Ps = {}  # stores initial policy (returned by neural net)
        self.b_mcts.Es = {}  # stores game.getGameEnded ended for board s
        self.b_mcts.Vs = {} 
        self.b_mcts.V = {}
    
    def estimate_params(self, path, analist=1):
        #puctの推定は難しいので秒数を予測
        elimit = 5
        h = load_data(path)
        self.turn = h[len(h)-1][0] #変える時は変えてください
        
        self.reset_mcts()
        mcts_players = [self.s_mcts, None, self.b_mcts] if self.turn != -1 else [self.b_mcts, None, self.s_mcts]
        curPlayer = 1
        pboard = None
        for step in range(len(h)-1):
            if self.turn == curPlayer or step == 0:
                board, sNsa, bNsa, sv, bv, sVs, bVs = h[step]
                pboard = board
                continue

            pcanonicalBoard = self.game.getCanonicalForm(pboard, getCurrentPlayer(pboard))
            ps = self.game.stringRepresentation(pcanonicalBoard)
            counts = self.getPastCount(path, getStep(pboard), analist)
            valids = self.game.getValidMoves(pboard, getCurrentPlayer(pboard))
            ranking = np.argsort(counts)
            ranking = ranking[::-1]
            valids = [i  for i in range(len(valids)) if valids[i]]
            if len(valids) < number+1:
                number = len(valids) - 1 # とりあえず最下位を渡す
            ranking = [i for i in ranking if i in valids]
            oa = self.detectAction(pboard, board)

            if oa not in ranking:
                elimit = 0
            
            orank = ranking.index(oa)
            crange = len(ranking)

            pvalue = self.getPastValueNoModification(path, getStep(pboard), pboard, analist)
            cvalue = -self.getPastValueNoModification(path, getStep(board), board, analist)
            #相手側からの視点だからー
            vboard = self.add_stone(board.copy(), getCurrentPlayer(board), ranking[0])
            vvalue = -self.getPastValueNoModification(path, getStep(vboard), vboard, analist)
            
            # ここで基準を
            criterion = orank * (vvalue - cvalue)
            #一回自分自身がどうなるのかをみるべき
            pboard = board


    def train_offline(self, path, dual=False):
        h = load_data(path)
        self.turn = h[len(h)-1][0] #変える時は変えてください
        
        self.reset_mcts()
        mcts_players = [self.s_mcts, None, self.b_mcts] if self.turn != -1 else [self.b_mcts, None, self.s_mcts]
        curPlayer = 1
        for step in range(len(h)-1):
            #print(step)
            tmp = []
            board, sNsa, bNsa, sv, bv, sVs, bVs = h[step]
            tmp.append(board)
            #curPlayer = getStep(board)
            canonicalBoard = self.game.getCanonicalForm(board, curPlayer)
            s = self.game.stringRepresentation(canonicalBoard)
            mcts_players[curPlayer + 1].getActionProb(canonicalBoard)
            dir_noise = mcts_players[curPlayer + 1].dirichlet_noise
            v = mcts_players[curPlayer + 1].search(canonicalBoard, dirichlet_noise=dir_noise)
            if dual == True:
                kv = mcts_players[-curPlayer + 1].search(canonicalBoard, dirichlet_noise=dir_noise)
            tmp.append(self.s_mcts.Nsa.copy())
            tmp.append(self.b_mcts.Nsa.copy())
            cp, cv = mcts_players[-curPlayer + 1].nn_agent.predict(canonicalBoard)
            
            if curPlayer != self.turn:
                tmp.append(-self.s_mcts.V[s])
                if s in self.b_mcts.V.keys():
                    tmp.append(-self.b_mcts.V[s])
                else:
                    tmp.append(-cv)
            else:
                # プレイヤーの視点に補正
                if s in self.s_mcts.V.keys():
                    tmp.append(self.s_mcts.V[s])
                else:
                    tmp.append(cv)
                tmp.append(self.b_mcts.V[s])
            
            tmp.append(self.s_mcts.V.copy())
            tmp.append(self.b_mcts.V.copy())
            curPlayer = -curPlayer
            self.data.append(tmp)
         
        self.data.append([self.turn, self.s_link, self.b_link, self.strong_timelimit, 
                         self.weak_timelimit, self.strong_puct, self.weak_puct])
        write_data(self.data, offline=True)
            
            
            
            
            
        
        
        
        
    def playGame(self,verbose=False):
        #ここでは 弱い方を先番とする
        if verbose==True:
            print("self turn: ", self.turn)
            print("change?[y/n]")
            answer = input()
            if answer == "y":
                self.turn = -self.turn
                print("new self turn", self.turn)
            
        #なんでこの順番？
        #players = [self.setting[1], None, self.setting[0]] 
        mcts_players = [self.b_mcts, None, self.s_mcts]
        if self.turn != -1:
            mcts_players = [self.s_mcts, None, self.b_mcts]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer) == 0:
            tmp = []
            tmp.append(board)
            it += 1
            
            if verbose:
                print("Turn ", str(it), "Player ", str(curPlayer))
                #self.display(board)
                print(board)      
            
            #一度すべてのアクションを出してる？
            #get_actionprobは直接実装してｖを取り出す
            canonicalBoard= self.game.getCanonicalForm(board, curPlayer)
            s = self.game.stringRepresentation(canonicalBoard)
            timelimit = self.strong_timelimit
            if curPlayer == self.turn:
                timelimit = self.weak_timelimit
            dir_noise = mcts_players[curPlayer + 1].dirichlet_noise
            start_time = time.time()
            while time.time() - start_time < timelimit:
                v = mcts_players[curPlayer + 1].search(canonicalBoard, dirichlet_noise=dir_noise)
            if verbose:
                print(v)
            #print(self.s_mcts.V[s])
            
            counts = [
                mcts_players[curPlayer+1].Nsa[(s, a)] if (s, a) in mcts_players[curPlayer+1].Nsa else 0
                for a in range(self.game.getActionSize())
            ]
            #tmp=0なのでこれでOK
            if verbose:
                print(counts, "by", curPlayer)
                print("")
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            action_prob = probs
           # action_prob = mcts_players[curPlayer + 1].getActionProb(
                #self.game.getCanonicalForm(board, curPlayer),
                #temp=0   
                #)
            action = np.argmax(action_prob)
            #action = players[curPlayer + 1](self.game.getCanonicalForm(
                #board, curPlayer))
            #手番でない方はpredictで求める
            tmp.append(self.s_mcts.Nsa.copy())
            tmp.append(self.b_mcts.Nsa.copy())
                  
            cp, cv = mcts_players[-curPlayer+1].nn_agent.predict(canonicalBoard)
            if curPlayer != self.turn:
                tmp.append(-self.s_mcts.V[s])
                if s in self.b_mcts.V.keys():
                    tmp.append(-self.b_mcts.V[s])
                else:
                    tmp.append(-cv)
            else:
                # プレイヤーの視点に補正
                if s in self.s_mcts.V.keys():
                    tmp.append(self.s_mcts.V[s])
                else:
                    tmp.append(cv)
                tmp.append(self.b_mcts.V[s])
            
            tmp.append(self.s_mcts.V.copy())
            tmp.append(self.b_mcts.V.copy())
        
            
            self.data.append(tmp)
            valids = self.game.getValidMoves(
                self.game.getCanonicalForm(board, curPlayer), 1)
            
            if valids[action] == 0:
                logger.error('Action {} is not valid!'.format(action))
                logger.debug('valids = {}'.format(valids))
                assert valids[action] > 0
            
            #両mctsのnを保存、ｖも
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        
        
        #終局図
        print(board)
        verbose = True
        if verbose:
           
            print("Game over: Turn ", str(it), "Result ",
                  str(self.game.getGameEnded(board, 1)))
            #self.display(board)
        self.data.append([self.turn, self.s_link, self.b_link, self.strong_timelimit, 
                         self.weak_timelimit, self.strong_puct, self.weak_puct])
        #得点ｖをかえしている
        #dataを保存する
        os.makedirs('./data/', exist_ok=True)
        write_data(self.data)
        
        return curPlayer * self.game.getGameEnded(board, curPlayer)
    
    def playGameWithPolicy(self, ps, pb, turn, think=False, reach=False, verbose=False):
        #ここでは 弱い方を先番とする
        self.turn = turn 
        print(f"turn: {turn}")
        if verbose==True:
            print("self turn: ", self.turn)
            print("change?[y/n]")
            answer = input()
            if answer == "y":
                self.turn = -self.turn
                print("new self turn", self.turn)
            
        #なんでこの順番？
        #players = [self.setting[1], None, self.setting[0]] 
        mcts_players = [self.b_mcts, None, self.s_mcts] if self.turn == -1 else [self.s_mcts, None, self.b_mcts]
        policies = [pb, None, ps] if self.turn == -1 else [ps, None, pb]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer) == 0:
            result = self.game.getGameEnded(board, curPlayer)
            tmp = []
            tmp.append(board)
            it += 1
            
            
            if verbose:
                print("Turn ", str(it), "Player ", str(curPlayer))
                #self.display(board)
                print(board)      
            
            judge = False
            if policies[curPlayer+1]:
                if verbose:
                    print("policy check")
                for p in policies[curPlayer+1]:
                    # policy適用の判断は素のboardで policyかち合いは後で
                    judge, action = self.ifPolicyApplicable(board, curPlayer, p)
                    
                    if judge:
                        if verbose:
                            print(f"judge True action = {action}")
                        break

            #一度すべてのアクションを出してる？
            #get_actionprobは直接実装してｖを取り出す
            canonicalBoard= self.game.getCanonicalForm(board, curPlayer)
            s = self.game.stringRepresentation(canonicalBoard)
            if not (policies[curPlayer+1] != None and judge and (not think)):

                timelimit = self.strong_timelimit
                if curPlayer == self.turn:
                    timelimit = self.weak_timelimit
                dir_noise = mcts_players[curPlayer + 1].dirichlet_noise
                start_time = time.time()
                while time.time() - start_time < timelimit:
                    v = mcts_players[curPlayer + 1].search(canonicalBoard, dirichlet_noise=dir_noise)
                
                #print(self.s_mcts.V[s])
                
                counts = [
                    mcts_players[curPlayer+1].Nsa[(s, a)] if (s, a) in mcts_players[curPlayer+1].Nsa else 0
                    for a in range(self.game.getActionSize())
                ]
                #tmp=0なのでこれでOK
                if verbose:
                    print(counts, "by", curPlayer)
                    print("")
                bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
                bestA = np.random.choice(bestAs)
                probs = [0] * len(counts)
                probs[bestA] = 1
                action_prob = probs
            # action_prob = mcts_players[curPlayer + 1].getActionProb(
                    #self.game.getCanonicalForm(board, curPlayer),
                    #temp=0   
                    #)
                action = np.argmax(action_prob)
                #action = players[curPlayer + 1](self.game.getCanonicalForm(
                    #board, curPlayer))
                #手番でない方はpredictで求める
            
            #リーチを両方に適用する場合はこれ
            if reach:
                valid = self.game.getValidMoves(board, getCurrentPlayer(board))
                valid = [i  for i in range(len(valid)) if valid[i]]
                candidate = []
                for a in valid:
                    vboard = self.add_stone(board.copy(), getCurrentPlayer(board), a)
                    if self.game.getGameEnded(vboard, getCurrentPlayer(board)):
                        candidate.append(a)

                if len(candidate) > 0:
                    a = choice(candidate)   
                    
            
            tmp.append(self.s_mcts.Nsa.copy())
            tmp.append(self.b_mcts.Nsa.copy())
                  
            cp, cv = mcts_players[-curPlayer+1].nn_agent.predict(canonicalBoard)
            if curPlayer != self.turn:
                if s in self.s_mcts.V.keys():
                    tmp.append(-self.s_mcts.V[s])
                else:
                    tmp.append(None) #価値にもとづいているわけではないので
                if s in self.b_mcts.V.keys():
                    tmp.append(-self.b_mcts.V[s])
                else:
                    tmp.append(-cv)
            else:
                # プレイヤーの視点に補正
                if s in self.s_mcts.V.keys():
                    tmp.append(self.s_mcts.V[s])
                else:
                    tmp.append(cv)
                
                if s in self.b_mcts.V.keys():
                    tmp.append(self.b_mcts.V[s])
                else:
                    tmp.append(None)
            
            tmp.append(self.s_mcts.V.copy())
            tmp.append(self.b_mcts.V.copy())
        
            
            self.data.append(tmp)
            valids = self.game.getValidMoves(
                self.game.getCanonicalForm(board, curPlayer), 1)
            
            if valids[action] == 0:
                print(board, action, curPlayer)
                logger.error('Action {} is not valid!'.format(action))
                logger.debug('valids = {}'.format(valids))
                
                action = choice([i for i in range(len(valids)) if valids[i]])
                
            
            #両mctsのnを保存、ｖも
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        
        
        #終局図
        #print(board)
        verbose = True
        if verbose:
           
            print("Game over: Turn ", str(it), "Result ",
                  str(self.game.getGameEnded(board, 1)))
            #self.display(board)
        self.data.append([self.turn, self.s_link, self.b_link, self.strong_timelimit, 
                         self.weak_timelimit, self.strong_puct, self.weak_puct, result])
        #得点ｖをかえしている
        #dataを保存する
        os.makedirs('./pdata/', exist_ok=True)
        #write_data(self.data, p=True)
        
        return curPlayer * self.game.getGameEnded(board, curPlayer)
    
    def ifPolicyApplicable(self, board, curPlayer, policy):
        # リーチ見逃しの可能性,リーチはつぶす方針でいいか
        height, width = self.game.getBoardSize()
        valid = self.game.getValidMoves(board, getCurrentPlayer(board))
        valid = [i  for i in range(len(valid)) if valid[i]]
        judge = False
        #リーチアリの場合はつぶす
        for a in valid:
            vboard = self.add_stone(board.copy(), getCurrentPlayer(board), a)
            if self.game.getGameEnded(vboard, getCurrentPlayer(board)):
                return True, a
        
        if policy == "v2lr":
            pattern = np.array(
            [[0, 0, 0],
            [0, 1, 0],
            [0, 1, 0]]
            )
            contain_indices, pure_indices = self.match_pattern(board, pattern)
            if len(contain_indices) > 0:
                candidate = []
                for s in contain_indices:
                    h = int(s/width)
                    w = s % width
                    if w > 0:
                        if board[h][w-1] == 0 and (h == height-1 or board[h+1][w-1] != 0):
                            judge = True
                            candidate.append(w-1)
                        
                    if w < height - 1:
                        if board[h][w+1] == 0 and (h == height-1 or board[h+1][w+1] != 0):
                           judge = True
                           candidate.append(w+1)
                    
                if not judge:
                    return judge, None
                
                action = choice(candidate)
                #print( a in valid)
                return judge, action

        if policy == "v2u":
            # パターンマッチング
            #　縦２が発生した瞬間つぶすので一つと仮定
            pattern = np.array(
            [[0, 0, 0],
            [0, 1, 0],
            [0, 1, 0]]
            )
            contain_indices, pure_indices = self.match_pattern(board, pattern)
            if len(contain_indices) > 0:
                candidate = []
                
                for s in contain_indices:
                    h = int(s/width)
                    w = s % width
                    if h-2 >= 0:
                        if board[h-2][w] == 0:
                            judge = True
                            candidate.append(w)
                
                if not judge:
                    return judge, None
                
                action = choice(candidate)
                #print( a in valid)
                return judge, action
        
        if policy == "h2lr":
            # 左右はランダム
            pattern = np.array(
            [[0, 0, 0, 0],
             [0, 1, 1, 0]]
            )
            contain_indices, pure_indices = self.match_pattern(board, pattern)
            if len(contain_indices) > 0:
                candidate = []
                
                for s in contain_indices:
                    h = int(s/width) - 1
                    w = s % width
                    if w > 0:
                        if board[h][w-1] == 0:
                           candidate.append(w-1)
                           judge = True

                    if w < width -2:
                        if board[h][w+2] == 0:
                            candidate.append(w+2)
                            judge = True

                if not judge:
                    return False, None

                action = choice(candidate)
                return judge, action
        
        return False, None
    
    def match_pattern(self, board, pattern):
        #その形が含まれているか
        #print(f"{pattern}\n=====================\n{board}")
        board = np.array(board)
        contain_indices = []
        pure_indices = []
        number = np.count_nonzero(pattern!=0)
        ph, pw = len(pattern), len(pattern[0])
        height = len(board)
        width = len(board[0])
        eboard = np.full((height+1, width+2), -2)
        eboard[1:, 1: len(eboard)+1] = board
        #print(eboard)
        purepattern = pattern.copy()
        purepattern[abs(purepattern) != 1] = 0
        #どっちや？？
        for i in range(height + 1 - ph + 1):
            for j in range(width + 2 - pw + 1):
                submatrix = eboard[i:i+ph, j:j+pw]
                inverse = submatrix * -1
                #print(submatrix)
                if (submatrix * purepattern == purepattern).all() or (inverse * purepattern == purepattern).all():
                    #含まれてる
                    #if (i!=0 and board[i-1][j]==0):
                    #    contain_indices.append((i)*width+j)
                    contain_indices.append((i+1)*width+j)
                    #print("contain")
                    #print(submatrix * purepattern)
                    #print(inverse * purepattern)
                    #print("------------------")
                    margin = submatrix - pattern
                    inverse_margin = inverse - pattern
                    surrounding = np.sum((abs(margin)!=0) | (abs(margin) <= 3))
                    inverse_surrounding = np.sum((abs(inverse_margin)!=0) | (abs(inverse_margin) <= 3))
                    if (surrounding == 0 or inverse_surrounding==0) :
                        
                        pure_indices.append(i*width+j)
                        #print("pure")
        
        return contain_indices, pure_indices
            

    
    def analyzeAll(self, path, threshold=0.5, simThreshold=5, seqThreshold=3,
                   interactive=True):
        # player 1の視点に補正
        h = load_data(path)
        check = False
        self.turn = h[len(h)-1][0] #変える時は変えてください
        analyze_data = {}
        actions = []
        past_v = h[0][3]
        past_bv = h[0][4]
        past_board = self.game.getInitBoard()
        print("step: ", 0, "turn: ", -getCurrentPlayer(past_board))
        print(past_board)
        for step in range(1, len(h)-1):
            board, sNsa, bNsa, sv, bv, sVs, bVs = h[step]
            current_v = sv
            current_bv = bv
            current_board = board
            info = {}
            print("step: ", step, "turn: ", getCurrentPlayer(board))
            print(board)
            # compare both side on value
            action = self.detectAction(past_board, current_board)
            actions.append(action)
            info["importance"] = self.getImportance(board, 1, path, step, importance=1)
         
            if sv - bv > threshold:
                info["view"] = "too pessimistic"
                check = True
            elif bv - sv > threshold:
                info["view"] = "too optimistic"
                check = True
            else:
                info["view"] = "normal"
            
            
            
            # record the point which value by the tearcher changes significantly
            
            if abs(current_v - past_v) > threshold:
                check = True
                if current_v < past_v:
                    info["change"] = "you approach to lose"
                else:
                    info["change"] = "you approach to win"
            
            if abs(current_bv - past_bv) > threshold:
                check = True
                if current_bv > past_bv:
                    info["feel"] = "might you have thought approach to lose"
                else:
                    info["feel"] = "might you have thought approach to win"
            
            analyze_data[step] = info
            past_v = current_v
            past_bv = current_bv
            past_board = current_board
            #GUI
            if interactive == True:
                print(info)
                if check:
                     print("analyze?[y/n]")
                     answer = yesOrNo()
                
                     if answer == "y":
                         print("activate onlyS?[y/n]")
                         panswer = yesOrNo()
                         if panswer == "y":
                             self.analyzeOne(path, step-1, onlyS=True)
                         else:
                             self.analyzeOne(path, step-1)
        points = detectSequence(actions, seqThreshold)
        flag = 0
        print(points)
        for step in range(len(points)):
            if points[step] == -1:
                analyze_data[step+1]["l/e"] = "into the sequence"
                flag = 1
                continue
            elif points[step] == 1:
                analyze_data[step+1]["l/e"] = "explore"
                flag = 0
            elif flag == 1:
                analyze_data[step+1]["l/e"] = "in the sequence"
            else:
                analyze_data[step+1]["l/e"] = "normal"
                
            
            
            
            
        
        if interactive == True:
            while True:
                print("want to see any step again? [y/n]")
                answer = yesOrNo()
                if answer == "n":
                    break
                else:
                    print("please input step you want to analyze")
                    answer = input()
                    
                    while (answer.isdecimal() == False or int(answer) < 0 
                    or int(answer) > len(h)-1):
                            answer = input()
                    answer = int(answer)
                    print(analyze_data[answer]["l/e"])
                    self.analyzeOne(path, (answer))
        return analyze_data
    
    def analyzeOne(self, path, step, inputBoard=-1, onlyS=False):
        h = load_data(path)
        tmp = h[step] # 注目する部分
        self.turn = h[len(h)-1][0]
        board, sNsa, bNsa, sv, bv, sVs, bVs = tmp
        if type(inputBoard) != int:
            board = inputBoard
        curPlayer = getCurrentPlayer(board)
        canonicalBoard = self.game.getCanonicalForm(board, curPlayer)
        s = self.game.stringRepresentation(canonicalBoard)
        print("current state, turn: ", getCurrentPlayer(board))
        print(board)
        print("-------------")
        if s in sVs.keys():
            if curPlayer == self.turn:
                print("svalue: ", sVs[s])
            else:
                print("svalue: ", -sVs[s])
        print("please input ")
        print("1 to see AI's next choice and possible play")
        print("2 to choose next choice on your own and see possible play ")
        
        answer = 0
        choice = [1, 2]
        while True:
            print("please input number")
            while answer not in choice:
                
                answer = int(input())
            if answer == 1:
                suggestion = self.suggestPastNextPosition(board, path, step)
                if suggestion == None:
                    print("see current AI suggestion?[y/n]")
                    a = input()
                    print("please input y or n")
                    while not (a == "y" or a == "n"):
                        a = input()
                    
                    if a == "y":
                        suggestion = np.argmax(self.s_mcts.getActionProb(board, curPlayer))
                    else:
                        print("quit?[y/n]")
                        a = input()
                        while not (a == "y" or a == "n"):
                            print("please input y or n")
                        if a == "y":
                            return 
                        else:
                            continue
                
                print("AI's choice: ", suggestion)
                nboard, nplayer = self.game.getNextState(board, curPlayer, suggestion)
                ncanonicalboard = self.game.getCanonicalForm(nboard, nplayer)
                ns = self.game.stringRepresentation(ncanonicalboard)
                if self.game.getGameEnded(nboard, nplayer):
                    print("like this")
                else:

                    if ns in sVs.keys():
                        if curPlayer == self.turn:
                            print("svalue :", sVs[ns])
                        else:
                            print("svalue :", -sVs[ns])
                    print(nboard)
                    print("----")
                    if onlyS == True:
                        self.simulate(path, step, inputBoard=nboard.copy(), onlyS=True)
                    else:
                        self.simulate(path, step, inputBoard=nboard.copy())
                
            elif answer == 2:
                valid = self.game.getValidMoves(board, curPlayer)
                print("please input your choice")
                while True:
                    action = self.getActionNumber()
                    while action < 0 or action > 6 or valid[action] == False :
                        print("please input a valid move: valid = ", valid)
                        action = int(input())
                    
                    print("your action is ", action, "rigtht?[y/n]")
                    print("please input y or n")
                    answer = yesOrNo()
                   
                    
                    if answer == "y":
                        break
                    else:
                        print("please input your choice")
                        continue
                nboard, nplayer = self.game.getNextState(board, curPlayer, action)
                ncanonicalboard = self.game.getCanonicalForm(nboard, nplayer)
                ns = self.game.stringRepresentation(ncanonicalboard)
                if self.game.getGameEnded(nboard, nplayer):
                    print("like this")
                else:
                    
                    if ns in sVs.keys():
                        print("svalue :", sVs[ns])
                        
                    if onlyS == True:
                        self.simulate(path, step, inputBoard=nboard, onlyS=True)
                    else:
                        self.simulate(path, step, inputBoard=nboard)
            
            print("quit?[y/n]")
            print("please input y or n")
            a = input()
            
            while not (a == "y" or a == "n"):
                a = input()
            if a == "y":
                return 
            else:
                print("please input ")
                print("1 to see AI's next choice and possible play")
                print("2 to choose next choice on your own and see possible play ")
                print(board)
                answer = 0
                continue
    
    def detectHotState(self, board, analist, path, step, threshold=0.1,limit=100, toend=False, mode="board"):
        '''
        thresholdはvalue同士の同じとみなされる最大のライン
        1か-1になった盤面もしくは木の果ての部分
        edgeが０、judgeが１, endが-1
        打てる所がない、そもそも登録されてないはNone
        limit: defaultは100(制限なし)
        toendがTrueだとjudgeで最後までいく
        '''
        zflag = True if analist == 0 else False

        
        traj = []

        #print(board)
        #print(type(board))
        h = load_data(path)
        tmp = h[step] # 注目する部分
        tboard, sNsa, bNsa, sv, bv, sVs, bVs = tmp
        
        self.turn = h[len(h)-1][0]
        curPlayer = getCurrentPlayer(board)
        modify = 1
        if self.turn != curPlayer:
            modify = -1
        
        vboard = board.copy()
        vcanonicalBoard = self.game.getCanonicalForm(vboard, curPlayer)
        vs = self.game.stringRepresentation(vcanonicalBoard)
        
        
        vstep = getStep(board) # countは差分で得られる
        if zflag:
            analist = getCurrentPlayer(board)
        counts = self.getPastCount(path, step, vboard, analist)
        #print(self.getPastValueNoModification( path, step, vboard, 1))
        if self.game.getGameEnded(board, curPlayer):
            #judge
            result = (board, -1) if mode == "board" else (board, -1, traj)
            return result
        
        
        if analist == 1:
            if vs not in sVs.keys():
                result = (None, None) if mode == "board" else (None, None, None)
                return result
        else:
            if vs not in bVs.keys():
                result = (None, None) if mode == "board" else (None, None, None)
                return result
        vvalue = sVs[vs] if analist ==1 else bVs[vs]
        vvalue *= vvalue
        
        vplayer = curPlayer
    
        while True:
            if zflag:
                analist = vplayer

            #print(vboard)
            #print(vvalue)
            #print("--------")
            if limit == 0:
                result = (vboard, 0) if mode == "board" else (vboard, 0, traj)
                return result
           
            valids = self.game.getValidMoves(vboard, vplayer)
            counts = self.getPastCount(path, step, vboard, analist)
            

            action = np.argmax(self.getPastActionProb(path, step, vboard, 
                                                      analist, counts = counts))
            traj.append(action)
                
            if valids[action] == 0:
                result = (vboard, None) if mode == "board" else (vboard, None, traj)
                return result
            
            if sum(counts) == 0:
                # edge
                result = (vboard, 0) if mode == "board" else (vboard, 0, traj)
                return result
                
            vnextBoard, vplayer = self.game.getNextState(vboard, vplayer, action)
            vcanonicalBoard = self.game.getCanonicalForm(vboard, -vplayer)
            vs = self.game.stringRepresentation( vcanonicalBoard)
           
            vstep += 1
            if analist == 1:
                if vs not in sVs.keys():
                    # edge
                    result = (vboard, 0) if mode == "board" else (vboard, 0, traj)
                    return result
            else:
                if vs not in bVs.keys():
                    # edge
                    result = (vboard, 0) if mode == "board" else (vboard, 0, traj)
                    return result
                
            vnextValue = sVs[vs] if analist ==1 else bVs[vs]
            vnextValue *= modify
            if not toend:
                if abs(vnextValue-vvalue) < threshold and abs(vvalue) == 1:
                    # judge
                    result = (vboard, 1) if mode == "board" else (vboard, 1, traj)
                    return result
          
            if self.game.getGameEnded(vnextBoard, vplayer):
                # end
                result = (vnextBoard, -1) if mode == "board" else (vnextBoard, -1, traj)
                return result
            
            
            
            vboard = vnextBoard
            vvalue = vnextValue
            modify = -modify
            limit -= 1
    
    def getBeforeCounts(self, board, analist, path, step):
        '''
        一つ前の盤面の訪問数を集める
        取り除けない列は-1になる
        '''
        h = load_data(path)
        tmp = h[step] # 注目する部分
        zboard, sNsa, bNsa, sv, bv, sVs, bVs = tmp
        Nsa = sNsa if analist == 1 else bNsa
        player = -getCurrentPlayer(board)
        height, width = self.game.getBoardSize()
        ravailable_idx = []
        for c in range (width):
            tmp = np.where(board[:, c] != 0)
            if tmp[0].size > 0 and board[tmp[0][0]][c] == player:
                ravailable_idx.append(c)
            
        bcounts = [-1 for i in range(width)]
        for c in ravailable_idx:
            vboard = self.remove_stone(board.copy(), player, c)
            vcanonicalBoard = self.game.getCanonicalForm(vboard, player) #取り除いたかあこれでok
            vs = self.game.stringRepresentation(vcanonicalBoard)
            if (vs, c) not in Nsa:
                bcounts[c] = 0
            else:
                bcounts[c] = Nsa[(vs, c)]
        
        return bcounts




            

    def suggestPastNextPosition(self, board, path=None, step=None):
        if path != None and step != None:
            h = load_data(path)
            tmp = h[step] # 注目する部分
            zboard, sNsa, bNsa, sv, bv, sVs, bVs = tmp
        
        curPlayer = getCurrentPlayer(board)        
        canonicalBoard = self.game.getCanonicalForm(board, curPlayer)
        
        probs = self.getPastActionProb(path, step, board, 1)
        
        if np.sum(probs) == 0:
            print("did not search this state")
            return None
        
        suggestion = np.argmax(probs)
        return suggestion
    
    def simulate(self, path, step, why=0, threshold=0.5, seqThreshold=3, 
                 stepThreshold=5, onlyS=False, inputBoard=-1, scurrent=False, 
                 bcurrent=False):
        '''
        
        とりあえずバーッと進める様の関数
        引数はinfoだけでええかもやけどとりあえずこっちで
        whyの場合分けはとりあえずなくてもいい気がした
        一手目は順番にかかわらずAI
        if 形成判断がおかしいとき why = 1(p), 2(o)
            一致するor終わるまで分岐を進める
        elif 形成悪化（悪化だけでええか？）のとき why = 3
        　　　これは終わりまで分岐を進める
    　　　
        '''
        #向き補正
        h = load_data(path)
        tmp = h[step] # 注目する部分
        board, sNsa, bNsa, sv, bv, sVs, bVs = tmp
        self.turn = h[len(h)-1][0]
        curPlayer = getCurrentPlayer(board)
        
        if type(inputBoard) != int:
            board = inputBoard

        if self.game.getGameEnded(board, curPlayer):
            return None
       
        #classify the type
        if why == 0:
            if sv - bv > threshold:
                why = 1 
            elif sv - bv < threshold:
                why = 2
            else:
                why = 3
        buffer = []
        curPlayer = getCurrentPlayer(board)
        print(curPlayer)
        # start simulation 一手目は手番通りそこから順番に
        print("let's start simulation!")
        print(board)
        buffer.append(board.copy())
        vboard = board.copy()
        vcanonicalBoard = self.game.getCanonicalForm(vboard, curPlayer)
        vs = self.game.stringRepresentation(vcanonicalBoard)
        vstep = step # countは差分で得られる
        flag = 0
        seqCount = 0
        paction = -1
        analist = -(self.turn * curPlayer)
        vplayer = curPlayer
        if onlyS == True:
            analist = 1
            
        while True:
            print("--------")
           
            valids = self.game.getValidMoves(vboard, vplayer)
            counts = self.getPastCount(path, step, vboard, analist)
            action = np.argmax(self.getPastActionProb(path, step, vboard, 
                                                      analist, counts = counts))
            if action == paction:
                seqCount += 1
            else:
                if seqCount >= seqThreshold:
                    print("explore")
                seqCount = 0
                
            while valids[action] == 0:
                #あとで
                action = np.argmax(self.getPastActionProb(path, step, vboard.copy(), analist))
            if sum(counts) == 0:
                print("did not visit this state")
                if flag == 0:
                    print("analist did not visit this state and refers to NNW")
                    print("continue or change to the current agent this time?")
                    flag = 1
                
                print("input y for continue, n for change")
                select = yesOrNo()
                if select == "n":
                    if analist == 1:
                        action = np.argmax(self.s_mcts.getActionProb(
                            vcanonicalBoard, temp=0, timelimit=self.strong_timelimit))
                    else:
                        action = np.argmax(self.b_mcts.getActionProb(
                            vcanonicalBoard, temp=0, timelimit=self.weak_timelimit))
                        
            
            print(vplayer, action, "buffer: ", vstep-step)      
            #予測になかった場合補足すべき？
            vboard, vplayer = self.game.getNextState(vboard, vplayer, action)
            vcanonicalBoard = self.game.getCanonicalForm(vboard, vplayer)
            vs = self.game.stringRepresentation(vcanonicalBoard)
            paction = action
            print("seqCount", seqCount)
            analist = -analist
            print(vboard)
            buffer.append([vboard.copy(), seqCount])
            
            if onlyS == True:
                analist = 1
            
            vstep += 1
            
            if seqCount >= seqThreshold:
                sleep(2)
                print("in the sequence")
            if self.game.getGameEnded(vboard, vplayer):
                print("like this")
                print(vstep - step)
                break
            
            if vs in sVs.keys():
                if vplayer == self.turn:
                    print("svalue: ", sVs[vs], end=" ")
                else:
                    print("svalue: ", -sVs[vs], end=" ")
            else:
                cp, cv = self.s_mcts.nn_agent.predict(vcanonicalBoard)
                if vplayer == self.turn:
                    print("svalue: ", cv, end=" ")
                else:
                    print("svalue: ", -cv, end=" ")
            
            if vs in bVs.keys():
                if vplayer == self.turn:
                    print("bvalue: ", bVs[vs])
                else:
                    print("bvalue: ", -bVs[vs])
            else:
                cp, cv = self.b_mcts.nn_agent.predict(vcanonicalBoard)
                if vplayer == self.turn:
                    print("bvalue: ", cv)
                else:
                    print("bvalue: ", -cv)
                
            if vs in sVs.keys() and vs in bVs.keys():
                #print("values", sVs[vs], bVs[vs])
                if abs(sVs[vs] - bVs[vs]) < threshold:
                    print("see further?[y/n]")
                    answer = yesOrNo()      
                    
                    if answer == "n":
                	    break
            if (vstep - step) % stepThreshold == 0:
                print("see further?[y/n]")
                answer = yesOrNo() 
                
                if answer == "n":
                    break
        print("from", board)
        print("to", vboard)
        print("want to go back?[y/n]")
        print(len(buffer))
        answer = yesOrNo()
        while answer == "y":
            
            print("input a number you want to see:")
            a = int(input())
            vboard = buffer[a][0].copy()
            vplayer = getCurrentPlayer(vboard)
            print("turn: ", vplayer)
            print(vboard)
            print("please input your choice")
            a = int(input())
            print("your action is:", a)
            vboard, vplayer = self.game.getNextState(vboard, vplayer, a)
            if buffer[a][1] >= seqThreshold:
                print("in the sequence")
            self.simulate(path, step, inputBoard=vboard.copy())
            
            print("want to go back?[y/n]")
            answer =input()
    
    def simpleSimulate(self, path, step, board, onlyS=False, stepThreshold=5,
                       change=False, infinite=False):
        #最初onlyS=TRUEで実験,動作根拠取り出しなのでchange=False
        h = load_data(path)
        tmp = h[step] # 注目する部分
        tboard, sNsa, bNsa, sv, bv, sVs, bVs = tmp
        self.turn = h[len(h)-1][0]
        
        curPlayer = getCurrentPlayer(board)
        if self.game.getGameEnded(board, curPlayer):
            return board
       
        vboard = board.copy()
        vcanonicalBoard = self.game.getCanonicalForm(vboard, curPlayer)
        
        vstep = step # countは差分で得られる
        flag = 0
        #seqCount = 0
        #paction = -1
        analist = -(self.turn * curPlayer)
        vplayer = curPlayer
        if onlyS == True:
            analist = 1
        
        
        
        while True:
            #print("--------")
           
            valids = self.game.getValidMoves(vboard, vplayer)
            counts = self.getPastCount(path, step, vboard, analist)
            action = np.argmax(self.getPastActionProb(path, step, vboard, 
                                                      analist, counts = counts))
                
            while valids[action] == 0:
                #あとで
                action = np.argmax(self.getPastActionProb(path, step, vboard.copy(), analist))
            if sum(counts) == 0:
                if change:
                    if analist == 1:
                        action = np.argmax(self.s_mcts.getActionProb(
                            vcanonicalBoard, temp=0, timelimit=self.strong_timelimit))
                    else:
                        action = np.argmax(self.b_mcts.getActionProb(
                            vcanonicalBoard, temp=0, timelimit=self.weak_timelimit))
                        
            
            #予測になかった場合補足すべき？
            vboard, vplayer = self.game.getNextState(vboard, vplayer, action)
            vcanonicalBoard = self.game.getCanonicalForm(vboard, vplayer)
            
            analist = -analist
           
            if onlyS == True:
                analist = 1
            
            vstep += 1
          
            if self.game.getGameEnded(vboard, vplayer):
                #詰み状態
                break
            if infinite == False and (vstep - step) % stepThreshold == 0:
               break
            #svは入れた方がいいか？？？
                
            #とりあえず進みたいだけ進む
        return vboard
    
    
    def observeTraverse(self, path, step, board, onlyS=False,stepThreshold=5,
                           change=False, infinite=False):
        h = load_data(path)
        tmp = h[step] # 注目する部分
        tboard, sNsa, bNsa, sv, bv, sVs, bVs = tmp
        
        curPlayer = getCurrentPlayer(board)
        valids = self.game.getValidMoves(
            self.game.getCanonicalForm(board, curPlayer), 1)
        futures = []
        futureValues = []
        suggestion = np.argmax(self.getPastActionProb(path, step, board, 
                                                  1))
            
        while valids[suggestion] == 0:
            #あとで
            suggestion = np.argmax(self.getPastActionProb(path, step, board, 
                                                          1))
        
        for action in range(self.game.getActionSize()):
            if valids[action] == False:
                futures.append(-1)
            else:
                vboard, vplayer = self.game.getNextState(board, curPlayer,
                                                         action)
                vvboard = self.simpleSimulate(path, step, vboard,onlyS=onlyS
                                                   ,stepThreshold=stepThreshold-1,
                                                   change=change, infinite=infinite)
              
                if self.game.getGameEnded(vvboard, curPlayer):
                    vvalue = self.game.getGameEnded(vvboard, curPlayer)
                else:
                    vvalue = self.getPastValue(path, step, vvboard, 1)
                futures.append(vvboard)
                futureValues.append(vvalue)
        
        suggested = futures[suggestion]
        difference = []
        distance = []
        for action in range(self.game.getActionSize()):
            if type(futures[action]) == int:
                m = 100
                k = 100
            else:

                m = self.getDifference(suggested, futures[action])
                k = self.getDistance(suggested, futures[action])
            difference.append(m)
            distance.append(k)
        #print(self.getPastCount(path, step, board,1))
        #print(difference)
        #print(distance)
        #print(futureValues)
        fatal = []
        for action in range(self.game.getActionSize()):
            #print(futures[action])
            if infinite == True and type(futures[action]) != int:
                f = self.detectFatalStone(futures[action])
                if f:
                    fatal.extend(f)
        
        
        if infinite == True:
            #print(fatal)
            #print(collections.Counter(fatal).items())
            #fatal = [k for k, v in collections.Counter(fatal).items() if v > 1]
            height, width = self.game.getBoardSize()
            
            visual = [0 if i not in collections.Counter(fatal).keys() else collections.Counter(fatal)[i]
                      for i in range(height * width)]
            visual = np.array(visual).reshape(height, width)
            
            #print(visual) 
            return visual
        
        return None
              
                
    def getPastCount(self, path, step, board, analist):
        '''
        analist: 手番とは区別、どちらを先番にするかにかかわらず１がｓ、ー１がｂ
        '''
        h = load_data(path)
        tmp = h[step] # 注目する部分
        zboard, sNsa, bNsa, sv, bv, sVs, bVs = tmp
        curPlayer = getCurrentPlayer(board)
       
        canonicalBoard= self.game.getCanonicalForm(board, curPlayer)
        s = self.game.stringRepresentation(canonicalBoard)
        
        if analist == 1:
            counts = [
                sNsa[(s, a)] if (s, a) in sNsa else 0
                for a in range(self.game.getActionSize())
            ]
        else:
            
            counts = [
                bNsa[(s, a)] if (s, a) in bNsa else 0
                for a in range(self.game.getActionSize())
            ]
        return counts
    
    def getAllPastValues(self, path, step, board, analist):
        height, width = self.game.getBoardSize()
        player = getCurrentPlayer(board)
        valid = self.game.getValidMoves(board, player)
        valid = [i  for i in range(len(valid)) if valid[i]]
        values = [self.getPastValueNoModification(path, step, self.game.getNextState(board.copy(), player, i)[0], analist) if i in valid else -1 for i in range(width) ]
        return values


    
    def getPastValue(self, path, step, board, analist):
        h = load_data(path)
        tmp = h[step] # 注目する部分
        zboard, sNsa, bNsa, sv, bv, sVs, bVs = tmp
        self.turn = h[len(h)-1][0]
        curPlayer = getCurrentPlayer(board)
        canonicalBoard= self.game.getCanonicalForm(board, curPlayer)
        s = self.game.stringRepresentation(canonicalBoard)
        modify = 1
        if self.turn != curPlayer:
            modify = -1
        
        if analist == 1:
            if s in sVs.keys():
                if type(sVs[s]) == int:
                    return modify * float(sVs[s]) 
                elif type(sVs[s]) == np.ndarray:
                    return modify * sVs[s].astype(np.float32).tolist()[0]
                return modify * sVs[s]
            else:
                cp, cv = self.s_mcts.nn_agent.predict(canonicalBoard)
                return modify * cv.astype(np.float32).tolist()[0]
        else:
            if s in bVs.keys():
                if type(bVs[s]) == int:
                    return modify * float(bVs[s]) 
                elif type(bVs[s]) == np.ndarray:
                    return modify * bVs[s].astype(np.float32).tolist()[0]
                
                return modify * bVs[s]
            else:
                cp, cv = self.b_mcts.nn_agent.predict(canonicalBoard)
                return modify * cv.astype(np.float32).tolist()[0]
    
    def getPastValueNoModification(self, path, step, board, analist):
        #手番の向きのが出てくる
        h = load_data(path)
        tmp = h[step] # 注目する部分
        zboard, sNsa, bNsa, sv, bv, sVs, bVs = tmp
        self.turn = h[len(h)-1][0]
        curPlayer = getCurrentPlayer(board)
        canonicalBoard= self.game.getCanonicalForm(board, curPlayer)
        s = self.game.stringRepresentation(canonicalBoard)
        if analist == 1:
            if s in sVs.keys():
                #print(self.getPastCount(path, step, board, 1))
                #print(s)
                #print(sVs[s])
                #print("in")
                if type(sVs[s]) == int:
                    return float(sVs[s]) 
                elif type(sVs[s]) == np.ndarray:
                    return sVs[s].astype(np.float32).tolist()[0]
                
                return sVs[s]
            else:
                cp, cv = self.s_mcts.nn_agent.predict(canonicalBoard)
                return cv.astype(np.float32).tolist()[0]
        else:
            if s in bVs.keys():
                
                if type(bVs[s]) == int:
                    return float(bVs[s]) 
                elif type(bVs[s]) == np.ndarray:
                    return bVs[s].astype(np.float32).tolist()[0]
                
                return bVs[s]
            else:
                cp, cv = self.b_mcts.nn_agent.predict(canonicalBoard)
                return cv.astype(np.float32).tolist()[0]


    def getPastActionProb(self, path, step, board, 
                          analist, counts=1):
        #　とりあえずtmp=0で, analistは視点
        '''
        analist: 手番とは区別、どちらを先番にするかにかかわらず１がｓ、ー１がｂ
        '''
        h = load_data(path)
        tmp = h[step] # 注目する部分
        zboard, sNsa, bNsa, sv, bv, sVs, bVs = tmp
        
        curPlayer = getCurrentPlayer(board)
        valids = self.game.getValidMoves(
            self.game.getCanonicalForm(board, curPlayer), 1)
        canonicalBoard= self.game.getCanonicalForm(board, curPlayer)
        s = self.game.stringRepresentation(canonicalBoard)
        if counts == 1:
            if analist == 1:
                counts = [
                    sNsa[(s, a)] if (s, a) in sNsa else 0
                    for a in range(self.game.getActionSize())
                ]
            else:
                counts = [
                    bNsa[(s, a)] if (s, a) in bNsa else 0
                    for a in range(self.game.getActionSize())
                ]
        
        #print(counts)
        
        if sum(counts) == 0:
            if analist == 1:
                p, v = self.s_mcts.nn_agent.predict(canonicalBoard)
            else:
                p, v = self.b_mcts.nn_agent.predict(canonicalBoard)
            
            counts = p.copy()
            counts = [x*100 for x in counts]
            #print("nnw", counts)
            #print(np.argmax(counts))
            action = np.argmax(counts)
            #print(action, )
            while valids[action] == 0:
                #print(action)
                counts[action] = 0
                #print(counts)
                action = np.argmax(counts)
                
                
        #print("ok")       
        bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
        bestA = np.random.choice(bestAs)
        probs = [0] * len(counts)
        probs[bestA] = 1
        return probs
       