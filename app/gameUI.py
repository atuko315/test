import sys
import base64
import inspect
import os
import io
from submission_sample import SimpleAgent, dotdict, MCTS
import numpy as np
from tqdm import tqdm
from parl.utils import logger
from connectx_try import encode_weight, load_buffer, System, saliency
from connect4_game import Connect4Game
from pathlib import Path
from connectx_try import load_data, getCurrentPlayer, getStep, store_data
from random import uniform
import random
from time import sleep
from collections import defaultdict
from scipy.stats import norm, entropy
import tkinter as tk
from PIL import Image, ImageDraw
import copy
import collections

game = Connect4Game()
init = game.getInitBoard()

class UImanager(tk.Frame):
    def __init__(self, game, system, master=None):
        tk.Frame.__init__(self, master)
        print(type(self))
        self.system = system
        self.game = game
        self.master.title("connect4")
        self.height, self.width = self.game.getBoardSize()
        self.memory  = []

        self.turns = [0, 0, 1]

        self.oval_size = 40
        self.edge_width = 2
        self.board = self.game.getInitBoard()
        self.text = [[ "" for _ in range(self.width)] for _ in range(self.height)]
        #　一回両方人間で作る
        self.c = tk.Canvas(self, width = self.oval_size*self.width, height = self.oval_size*self.height, highlightthickness = 0)
        self.c.create_rectangle(0, 0, self.oval_size*self.width, self.oval_size*self.height, width = 0.0, fill = '#00A0FF')
        self.c.bind('<Button-1>', self.turn_of_human)
        self.c.pack()
        
        end_button = tk.Button(self, text="end", command=self.end_all)
        end_button.pack()
        self.start_button = tk.Button(self, text="AI", command=self.turn_of_AI)
        self.start_button.pack()
        open_button = tk.Button(self, text="Open", command=self.open_canvas)
        open_button.pack()
        self.open = False
        self.complete = False
        self.rwindow = tk.Tk()
        self.ac = tk.Canvas(self.rwindow, width = self.oval_size*self.width, height = self.oval_size*self.height, highlightthickness = 0)
        self.ac.create_rectangle(0, 0, self.oval_size*self.width, self.oval_size*self.height, width = 0.0, fill = '#00A0FF')
        self.bc = tk.Canvas(self.rwindow, width = self.oval_size, height = self.oval_size, highlightthickness = 0)
        self.close_button = tk.Button(self.bc, text="Close", command=lambda: self.close_canvas())


        self.fwindow = tk.Tk()
        self.fc = tk.Canvas(self.fwindow, width = self.oval_size*self.width, height = self.oval_size*self.height, highlightthickness = 0)
        self.fc.create_rectangle(0, 0, self.oval_size*self.width, self.oval_size*self.height, width = 0.0, fill = '#00A0FF')
        self.fbc = tk.Canvas(self.fwindow, width = self.oval_size, height = self.oval_size, highlightthickness = 0)
        self.fbutton = tk.Button(self.fbc, text="forward", command=lambda: self.forward_one())
        self.bbutton = tk.Button(self.fbc, text="back", command=lambda: self.back_one())
        self.fclose_button = tk.Button(self.fbc, text="Close", command=lambda: self.end_feedback())
        self.fv_button = tk.Button(self.fbc, text="vector", command=lambda: self.show_fvector())
        self.ft_button = tk.Button(self.fbc, text="traj", command=lambda: self.show_ftraj())
        self.ff_button = tk.Button(self.fbc, text="map", command=lambda: self.show_saliency_map())
        self.fopen = False
        self.answer = defaultdict(lambda:[]) # keyはstep*analist
        
        self.on_draw(self.board, self.text, self.c)
        print("draw")
        #self.play()
        
        #self.play()
    def reset(self):
        self.memory = []
        self.system.reset_mcts()
    
    def end_all(self):
        self.fwindow.destroy()
        self.rwindow.destroy()
        self.destroy()
        
    def open_canvas(self):
    # 新しいキャンバスを作成
        if self.open:
            return

        
        self.ac.pack()
        self.bc.pack()
        if not self.open:
            self.close_button.pack()
        if not self.open:
            self.open = True
        
        # [Close]ボタンを新しいキャンバスに追加
        self.show_vector(self.board, self.text, 1, 0, 2)
        
        return
    
    def close_canvas(self):
        self.ac.create_rectangle(0, 0, self.oval_size*self.width, self.oval_size*self.height, width = 0.0, fill = '#00A0FF')
        self.open = False
        self.on_draw(self.board, self.text, self.c)
        self.ac.pack_forget()
        self.close_button.pack_forget()
    
    def inputValidMove(self, board, player, event):
        valids = self.game.getValidMoves(board, player)
        x = -1
        while  (x < 0 or x > self.width):
            x = int(event.x/self.oval_size)
            if (x < 0 or x > self.width):
                continue
            if not valids[x]:
                continue
        return x
    
    def turn_of_AI(self,mode="search", path=None, step=None):
       
        self.start_button.configure(text = str(getCurrentPlayer(self.board)))
        player = getCurrentPlayer(self.board)
        if self.turns[player+1] == 0:
            return
        canonicalboard = self.game.getCanonicalForm(self.board.copy(), player)
       
        if mode == "past":
            self.start_button.configure(text = str(getCurrentPlayer(self.board)))
            action = np.argmax(self.system.getPastCount(path, step, self.board, self.turns[player+1]))
        else:
            action =  np.argmax(self.system.s_mcts.getActionProb(canonicalboard, temp=0)) if self.turns[player+1] == 1 else np.argmax(self.system.b_mcts.getActionProb(canonicalboard, temp=0))
        
       
        self.memory.append([self.board.copy(), self.system.s_mcts.Nsa.copy(), self.system.b_mcts.Nsa.copy(), None, None, self.system.s_mcts.V.copy(), self.system.b_mcts.V.copy()])
        self.board, next_player = self.game.getNextState(self.board.copy(), player, action)
        
        self.on_draw(self.board, self.text, self.c)
        self.start_button.configure(text="AI")
        
        return
      
        
        #self.on_draw(self.board, self.text, self.c)
        
    def turn_of_human(self, event):

        
        player = getCurrentPlayer(self.board)
        if self.turns[player+1] != 0:
            return
        if self.game.getGameEnded(self.board, player) != 0:
            self.finish_game()
            print(self.game.getGameEnded(self.board, player))
            self.memory.append([self.board.copy(), self.system.s_mcts.Nsa.copy(), self.system.b_mcts.Nsa.copy(), None, None, self.system.s_mcts.V.copy(), self.system.b_mcts.V.copy()])
            self.board = self.game.getInitBoard() # ?
            self.on_draw(self.board, self.text, self.c)
            return
        
        #if player != turn:
        #    return 
        # 先手処理
        action = self.inputValidMove(self.board, player, event)
        self.memory.append([self.board.copy(), self.system.s_mcts.Nsa.copy(), self.system.b_mcts.Nsa.copy(), None, None, self.system.s_mcts.V.copy(), self.system.b_mcts.V.copy()])
        self.board, next_player = self.game.getNextState(self.board.copy(), player, action)
        
        self.on_draw(self.board, self.text, self.c)
        return
     
      
        
        #self.master.alter(1, self.turn_of_human) # ?
    
    def modify_piece(self, board, c, group):
        '''
        groupで強調したいのを指定
        '''
        for g in group:
            x = (g%self.width)
            y = int(g/self.width)
            dx = x * self.oval_size + self.edge_width
            dy = y * self.oval_size + self.edge_width
            c.create_oval(dx, dy, dx+self.oval_size-self.edge_width*2, dy+self.oval_size-self.edge_width*2, width=1.0, fill = 'green')
    
    def draw_piece(self, board, text, c, index):
        x = (index%self.width)
        y = int(index/self.width)
        dx = x * self.oval_size + self.edge_width
        dy = y * self.oval_size + self.edge_width

        if board[y][x] == 1:
            
            c.create_oval(dx, dy, dx+self.oval_size-self.edge_width*2, dy+self.oval_size-self.edge_width*2, width=1.0, fill = '#FF0000')
        elif board[y][x] == -1:
            c.create_oval(dx, dy, dx+self.oval_size-self.edge_width*2, dy+self.oval_size-self.edge_width*2, width=1.0, fill = '#FFFF00')
        
        if text[y][x] != "":
            c.create_text(dx+int(self.oval_size/2)-self.edge_width, dy+int(self.oval_size/2)-self.edge_width, text=text[y][x], font=("Helvetica", 20), fill="black")
    
    def on_draw(self, board, text, c):
        self.complete = False
        print(board)
        #c.delete('all')
        
        for i in range(self.height*self.width):
            x = (i % self.width) * self.oval_size + self.edge_width
            y = int(i / self.width) * self.oval_size + self.edge_width
            c.create_oval(x, y, x+self.oval_size-self.edge_width*2, y+self.oval_size-self.edge_width*2, width=1.0, fill='#FFFFFF')
        
        for i in range(self.height*self.width):
            
            # AI入れるときは手番分けして
            self.draw_piece(board.copy(), text, c, i)
        
        #player = getCurrentPlayer(self.board)
        #print(self.turns[player+1])
        #if self.turns[player+1] != 0:
        #    print("ai's turn")
        #    self.turn_of_AI()
        #    self.on_draw()
        
        self.complete = True
        print("complete")
    
    def finish_game(self):
        self.start_feedback()
    
    def start_feedback(self):
        if self.fopen:
            return
        self.fc.pack()
        self.fbc.pack()
        self.fbutton.pack()
        self.bbutton.pack()
        self.fclose_button.pack()
        self.fv_button.pack()
        self.ft_button.pack()
        self.ff_button.pack()
        self.fboard = self.game.getInitBoard() 
        text =  [[ "" for _ in range(self.width)] for _ in range(self.height)]
        self.on_draw(self.fboard, text, self.fc)
    
    def end_feedback(self):
        self.ac.create_rectangle(0, 0, self.oval_size*self.width, self.oval_size*self.height, width = 0.0, fill = '#00A0FF')
        self.fopen = False
        self.board = self.game.getInitBoard() 
        self.text = [[ "" for _ in range(self.width)] for _ in range(self.height)]
        self.on_draw(self.board, self.text, self.c)
        self.fc.pack_forget()
        self.fbc.pack_forget()
        self.memory = []
        self.answer = defaultdict(lambda: [])
        self.system.reset_mcts()
    
    def forward_one(self):
        self.fc.create_rectangle(0, 0, self.oval_size*self.width, self.oval_size*self.height, width = 0.0, fill = '#00A0FF')
        bstep = getStep(self.fboard)
        print(bstep, len(self.memory))
        if bstep >= len(self.memory):
            return
        
        self.fboard = self.memory[bstep+1][0]
        print(self.fboard)
        text =  [[ "" for _ in range(self.width)] for _ in range(self.height)]
        self.on_draw(self.fboard, text, self.fc)
    
    def back_one(self):
        self.fc.create_rectangle(0, 0, self.oval_size*self.width, self.oval_size*self.height, width = 0.0, fill = '#00A0FF')
        bstep = getStep(self.fboard)
        if bstep <= 0:
            return
        
        self.fboard = self.memory[bstep-1][0]
        text =  [[ "" for _ in range(self.width)] for _ in range(self.height)]
        self.on_draw(self.fboard, text, self.fc)
    
   


    def play(self):
        
        player = getCurrentPlayer(self.board)
        print(player)
        while True:
            if self.turns[player+1] != 0:
                print("turn")
                self.turn_of_AI()
                print(sleep(10))
            if self.game.getGameEnded(self.board, player) != 0:
                print(self.game.getGameEnded(self.board, player))
                self.board = self.game.getInitBoard() # ?
                self.on_draw(self.board, self.text, self.c)
               
                
    
    def show_traj(self, board, traj, c, by_step=True):
        '''
        boardは引数
        vboardをvtext付きで表示
        本編とは違うキャンバス(self.ac)に
        closeは自分で押してもらう
        '''
        c.create_rectangle(0, 0, self.oval_size*self.width, self.oval_size*self.height, width = 0.0, fill = '#00A0FF')
        c.pack()
        
        vboard = board.copy()
        vtext = [[ "" for _ in range(self.width)] for _ in range(self.height)]
        for i in range(len(traj)):
            vboard, number = self.system.add_stone(vboard, getCurrentPlayer(vboard), traj[i], number=True)
            vtext[int(number/self.width)][number%self.width] = i+1
            
       
        
        
        self.on_draw(vboard, vtext, c)
        return 
    

    def show_vector(self, board, text, key_c, c, vector, distance):
        '''
        abs(vector) < 0.5のときは両側に表示
        vectorが大きい側のdistance+-0.5の部分に色をつける
        distanceが2.5以上のときは端まで色をつける
        '''
        print(vector, distance)
        c.create_rectangle(0, 0, self.oval_size*self.width, self.oval_size*self.height, width = 0.0, fill = '#00A0FF')
        tmp_text = copy.deepcopy(text)
        self.ac.pack()
        self.bc.pack()
        
        left = False
        right = False

        color = "green"
        alpha = 128
      
        if vector < 0:
            left =True
        else:
            right = True
        
        if abs(vector) < 0.5:
            left = True
            right = True
        
        if left:
            width = int(self.oval_size/2)

            base = max(self.oval_size*((key_c+0.5) - distance), width)
            
            
            
            if distance < 2.5:
                self.ac.create_rectangle(base-width, 0, base+width, self.oval_size*self.height, width = 0.0, fill = 'green')
            
            else:
                self.ac.create_rectangle(0, 0, base+width, self.oval_size*self.height, width = 0.0, fill = 'green')
               
        if right:
            width = int(self.oval_size/2)
            base = min(self.oval_size*(distance + (key_c+0.5)), self.oval_size*self.width - width)
            

            if distance < 2.5:
                self.ac.create_rectangle(base-width, 0, base+width, self.oval_size*self.height, width = 0.0, fill = 'green')
            else:
                self.ac.create_rectangle(base-width, 0, self.oval_size*self.width, self.oval_size*self.height, width = 0.0, fill = 'green')

        available_idx = np.where(board[:, key_c] == 0)
        print(available_idx)
        print(tmp_text is self.text)
        tmp_text[available_idx[0][len(available_idx[0])-1]+1][key_c] = "*"
        self.on_draw(board, tmp_text, self.ac)

        print(self.text[available_idx[0][len(available_idx[0])-1]+1][key_c] is tmp_text[available_idx[0][len(available_idx[0])-1]+1][key_c])
        if not self.open:
            self.close_button.pack()
        if not self.open:
            self.open = True
    
    def detect_actual_reach(self):
        last = self.memory[len(self.memory)-1]
        fboard, sNsa, bNsa, sv, bv, sVs, bVs = last
        
        valid = self.game.getValidMoves(fboard, getCurrentPlayer(fboard))
        valid = [i  for i in range(len(valid)) if valid[i]]
        reach = []
        for a in valid:
            vboard = self.system.add_stone(fboard.copy(), getCurrentPlayer(fboard), a)
            vf = self.system.detectFatalStone(vboard, per_group=True)
            if vf:
                reach.extend(vf)
        
        return reach
    
    def check_frequent_traj(self, analist=1, mode="group"):
        #featureにおけるtrajmodeだけ
        self.on_draw(self.fboard, self.text, self.fc)
        key = getStep(self.fboard) * analist
        if not self.answer[key]:
            self.answer[key] = self.hot_states_one_way(analist, step=4, baseline=2, fix=-1)
        bfcount, bfdcount, new_trajs, gs4, gd2, groups, visual = self.answer[key]
        print(gd2)
        groups = dict(groups)
        
        traj = []
        if mode == "group":
            for g in gd2:
                if str(g) in groups.keys():
                    traj.extend(groups[str(g)])
        else:
            for s in gs4:
                for g in groups:
                    if s in g.eval():
                        traj.extend(g)
        return traj
    
    def show_ftraj(self, analist=1, mode="group"):
        trajs = self.check_frequent_traj(analist=1, mode=mode)
        print(trajs)
        if not trajs:
            return 
        self.show_traj(self.fboard, trajs[0], self.fc)


    
    def show_potential(self, analist=1, mode="group"):
        self.on_draw(self.fboard, self.text, self.fc)
        key = getStep(self.fboard) * analist
        if not self.answer[key]:
            self.answer[key] = self.hot_states_one_way(analist, step=4, baseline=2, fix=-1)
        bfcount, bfdcount, new_trajs, gs4, gd2, groups, visual = self.answer[key]

        if mode=="group":
            flat = np.array(gd2).reshape(1, -1)
            self.modify_piece(self.fboard, self.fc, flat)
        else:
            self.modify_piece(self.fboard, self.fc, gs4)
    
    def show_fvector(self):
        vector, distance, metric = self.hot_vector_one_way()
        print(vector, distance)
        key_c = self.system.detectAction(self.memory[getStep(self.fboard)-1][0], self.fboard)
        self.show_vector(self.fboard, self.text, key_c, self.fc, vector, distance)
    

    
    def interpolate_color(self, value, start=(128, 0, 128), end=(255, 255, 0)):
        r1, g1, b1 = start
        r2, g2, b2 = end
        r = int(r1 + (r2 - r1) * value)
        g = int(g1 + (g2 - g1) * value)
        b = int(b1 + (b2 - b1) * value)
        print(r, g, b)
        return "#{:02X}{:02X}{:02X}".format(r, g, b)
    
    def show_map(self, visual):
        max_index = np.argmax(visual)
        max_value = visual[int(max_index/self.width)][(max_index%self.width)]
        for index in range(self.width*self.height):
            
            x = (index%self.width)
            y = int(index/self.width)
            color = self.interpolate_color(visual[y][x]/max_value)
            dx = x * self.oval_size + self.edge_width
            dy = y * self.oval_size + self.edge_width
            self.fc.create_oval(dx, dy, dx+self.oval_size-self.edge_width*2, dy+self.oval_size-self.edge_width*2, width=1.0, fill = color)
            self.fc.create_text(dx+int(self.oval_size/2)-self.edge_width, dy+int(self.oval_size/2)-self.edge_width, text="", font=("Helvetica", 20), fill="black")

    def show_fatal_map(self, analist=1):
        self.fc.create_rectangle(0, 0, self.oval_size*self.width, self.oval_size*self.height, width = 0.0, fill = '#00A0FF')
        key = getStep(self.fboard) * analist
        if not self.answer[key]:
            self.answer[key] = self.hot_states_one_way(analist, step=4, baseline=2, fix=-1)
        bfcount, bfdcount, new_trajs, gs4, gd2, groups, visual = self.answer[key]
        print(visual)
        self.show_map(visual)
        
        
    
    def show_saliency_map(self, analist=1, mode="policy",path=-1, step=-1):
        #countにするのは真ん中に不利なので今回policyはニューロだけに
        boards = [self.system.removeStone(self.fboard.copy(), i) for i in range(self.height*self.width)]
        if analist == 1:
            agent = self.system.s_mcts
        else:
            agent = self.system.b_mcts
        
        #元が前
        if mode == "value":
            #手番の問題で裏返し
            saliencies = saliencies = [saliency(
                -agent.nn_agent.predict(self.game.getCanonicalForm(self.fboard, getCurrentPlayer(self.fboard)))[1],
                agent.nn_agent.predict(self.game.getCanonicalForm(boards[i], getCurrentPlayer(boards[i])))[1]
                ) for i in range(self.height*self.width)]   
        else:
            saliencies = [saliency(
                agent.nn_agent.predict(self.game.getCanonicalForm(self.fboard, getCurrentPlayer(self.fboard)))[0],
                agent.nn_agent.predict(self.game.getCanonicalForm(boards[i], getCurrentPlayer(boards[i])))[0]
                ) for i in range(self.height*self.width)]   
        saliencies = np.array(saliencies).reshape(self.height, self.width)
        print(saliencies)
        self.show_map(saliencies)
    
   
    def hot_vector_one_way(self, analist=1, step=4, baseline=2, fix=-1):
        assert step > 0
        reach = self.detect_actual_reach()
        #print(reach)
        bstep = getStep(self.fboard)
        if fix != -1:
            bstep = fix
        
        assert bstep > 0
        
        
        latest = self.system.detectAction(self.memory[bstep-1][0], self.fboard)
        key_c = latest % self.width
       
        
        
        trajs = self.collect_promising_vector( key_c, analist, step, baseline=baseline, fix = bstep, mode="vector")
        vector = 0
        distance = 0
        print(trajs)
        for traj in trajs:
            if traj:
                vecs = [t[0] for t in traj]
                vector += np.mean(np.array(vecs))
                dists = [t[1] for t in traj]
                distance += np.mean(np.array(dists))
        vector /= len(trajs)
        distance /= len(trajs)
        #　このmetricはvec方向あり
        
        metric = vector * distance

        return vector, distance, metric


    
    def hot_states_one_way(self, analist=1, step=4, baseline=2, fix=-1):
        reach = self.detect_actual_reach()
        bstep = getStep(self.fboard)
        if fix != -1:
            bstep = fix
        
        latest = self.system.detectAction(self.memory[bstep-1][0], self.memory[bstep][0])
        key_c = latest % self.width
        btrajs, boards = self.collect_promising_vector_sub(self.fboard, key_c, analist, step, baseline=baseline, fix=fix, mode="normal")
        new_trajs = []
           
        if not btrajs:
            return None
        
        bfcount, bfdcount, trajs, gs4, gd2, groups, visual = self.check_convergence(boards, reach, bstep,  analist, btraj=btrajs)
        return bfcount, bfdcount, trajs, gs4, gd2, groups, visual
    
    def collect_promising_vector(self, key_c, analist, step, baseline, fix = -1, mode="normal"):
        trajs, boards = self.collect_promising_vector_sub(self.fboard, key_c, analist, step, baseline, fix = getStep(self.fboard), mode=mode)
        return trajs

    
    def collect_promising_vector_sub(self, boards, key_c, analist, step=4, baseline=2, fix=-1, mode="normal"):
        new_trajs = []
        new_boards = []
        if step == 1:
            board = boards
            boards = self.collect_promising_per_step(board, analist, baseline=baseline, fix=fix)
            for b in boards:
                a = self.system.detectAction(board, b)
                if mode != "vector":
                    new_trajs.append([a])
                    continue
                relative = self.detect_relative_distance(key_c, a)
                new_trajs.append([relative])
        
            return new_trajs, boards
        
        ftrajs, fboards = self.collect_promising_vector_sub(boards, key_c, analist, step-1, baseline, fix=fix, mode=mode)
       
        for i in range(len(ftrajs)):
            traj = copy.deepcopy(ftrajs[i])
            b = fboards[i]
            
            nboards = self.collect_promising_per_step(b, analist, baseline=baseline, fix=fix)
            for nb in nboards:
                traj = copy.deepcopy(ftrajs[i])
                a = self.system.detectAction(b, nb)
                if mode != "vector":
                    traj.append(a)
                    
                else:
                    relative = self.detect_relative_distance(key_c, a)
                    traj.append(relative)
                new_trajs.append(traj)
                
            new_boards.extend(nboards)
        
        return new_trajs, new_boards
    
    def collect_promising_per_step(self, board, analist, baseline=2, fix=-1):
        
        max_step = len(self.memory) - 1
        if analist == 0:
            analist = getCurrentPlayer(board)
        valid = self.game.getValidMoves(board, getCurrentPlayer(board))
        valid = [i  for i in range(len(valid)) if valid[i]]
        l = len(valid) if len(valid) < baseline else baseline
        #print(fpath, getStep(board))
        bstep = getStep(board)
        bstep = max_step if bstep > max_step else bstep
        if fix != -1:
            bstep = fix
        
        counts =  self.getPastCount(bstep, board, analist)
        counts = np.argsort(np.array(counts))

        counts = [c for c in counts if c in valid]
        counts = counts[-l:]
        fboards = []

        for c in counts:
                fboards.append(self.system.add_stone(board.copy(), getCurrentPlayer(board), c))
    
        return fboards

    
    def check_convergence(self, boards, reach, bstep,  analist, btraj=None):
        gd = defaultdict(lambda: 0)
        gs = defaultdict(lambda: 0)
        trajs = []
        groups = defaultdict(lambda: [])
        size = len(boards)
        if size < 1:
            return [], [], []
        
        c=0
        bcount = 0
        bfcount = 0
        bfdcount = 0
        index = 0

        for b in boards:
            group, stones, traj = self.check_convergence_per_board(b, reach, bstep, analist)
            if traj and btraj[index]:
                btraj[index].extend(traj)
                traj = btraj[index]
                
                index += 1
                trajs.append(traj)
            if group:
                for g in group:
                    
                    gd[str(g)] += 1
                    if  traj:
                        groups[str(g)].append(traj)

            if stones:
                for s in stones:
                    gs[s] += 1
        
        visual = [0 if i not in collections.Counter(gs).keys() else collections.Counter(gs)[i]
                        for i in range(self.height * self.width)]
        visual = np.array(visual).reshape(self.height, self.width)
        
        #print(visual)
        
        gs_sorted = sorted(dict(gs).items(), reverse=True, key=lambda x : x[1])
        
        if len(gs_sorted) < 4:
            gs4 = [gs_sorted[i][0] for i in range(len(gs_sorted))]
        else:
            gs4 = [gs_sorted[i][0] for i in range(4)]
        gd_sorted = sorted(dict(gd).items(), reverse=True, key=lambda x : x[1])
        print(gd_sorted)
        if len(gd_sorted) < 2:
            gd2 = [gd_sorted[i][0] for i in range(len(gd_sorted))]
        else:
            gd2 = [gd_sorted[i][0] for i in range(2)]
        gd2 = [eval(g) for g in gd2]
        
        #多い起動をオンライン的に取り出す場合はg2, g4を取り出す
        fu = np.unique(gs4.copy()).tolist() if gs4 else [-1]
        ru = np.unique(reach.copy()).tolist() if reach else [-2]
        bfdcount = 0
        bfcount = 0
        if len(set(ru)) > 0:
            bfdcount = (len(set(fu) & set(ru))) / 4
        #print(fu, ru, len(set(fu) & set(ru)))つくる
        if gd2:
            for g in gd2:
                for i in range(len(reach)):
                    r = reach[i]
                    if set(r).issubset(set(g)):
                        bfcount = 1
        
        print( bfcount, bfdcount, trajs, gs4, gd2, groups, visual)
        return bfcount, bfdcount, trajs, gs4, gd2, groups, visual
    
    def getPastCount(self, bstep, board, analist):
       
        zboard, sNsa, bNsa, sv, bv, sVs, bVs = self.memory[bstep]
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
    
    def check_convergence_per_board(self, board, reach, bstep, analist):
        bcount = 0
        bfcount = 0
        bfdcount = 0
        #print(board)
        hot = self.detectHotState(board, analist, bstep) # mode=traj, toend=True

        #print(hot[1])
        if hot[1] == None:
            return None, None, None
            
        end = self.game.getGameEnded(hot[0], getCurrentPlayer(hot[0]))
        if end:
            
            bcount = 1
            fatal = self.system.detectFatalStone(hot[0], per_group=True)
            fu = np.unique(fatal.copy()).tolist() if fatal else []
            return fatal, fu, hot[2]

        
            
            #print("fatal")
            #print(fatal)
            fu = np.unique(fatal.copy()).tolist() if fatal else [-1]
            ru = np.unique(reach.copy()).tolist() if reach else [-2]
            if len(set(ru)) > 0:
                bfdcount = (len(set(fu) & set(ru))) / 4
            #print(fu, ru, len(set(fu) & set(ru)))つくる
            if fatal:
                for g in fatal:
                    for i in range(len(reach)):
                        r = reach[i]
                        if set(r).issubset(set(g)):
                            bfcount = 1
                '''
                gs = str(g)
                #print(fatal_group.keys())
                if gs not in fatal_group.keys():
                    fatal_group[gs] = 1
                else:
                    fatal_group[gs] += 1
                '''
        
        return bfcount, bfdcount, hot[2]

    def detectHotState(self, board, analist, step):
        zboard, sNsa, bNsa, sv, bv, sVs, bVs = self.memory[step]
        zflag = True if analist == 0 else False

        
        traj = []

        curPlayer = getCurrentPlayer(board)
        
        vboard = board.copy()
        vcanonicalBoard = self.game.getCanonicalForm(vboard, curPlayer)
        vs = self.game.stringRepresentation(vcanonicalBoard)
        
        
        vstep = getStep(board) # countは差分で得られる
        if zflag:
            analist = getCurrentPlayer(board)
        counts = self.getPastCount(step, vboard, analist)
        #print(self.getPastValueNoModification( path, step, vboard, 1))
        if self.game.getGameEnded(board, curPlayer):
            #judge
            result = (board, -1, traj)
            return result
        
        
        if analist == 1:
            if vs not in sVs.keys():
                result = (None, None, None)
                return result
        else:
            if vs not in bVs.keys():
                result = (None, None, None)
                return result
       
        
        vplayer = curPlayer
    
        while True:
            if zflag:
                analist = vplayer

            #print(vboard)
            #print(vvalue)
            #print("--------")
           
            valids = self.game.getValidMoves(vboard, vplayer)
            counts = self.getPastCount(step, vboard, analist)

            if sum(counts) == 0:
                # edge
                result = (vboard, 0, traj)
                return result
            

            action = np.argmax(counts)
            traj.append(action)
                
            if valids[action] == 0:
                result = (vboard, None, traj)
                return result
            
            
                
            vnextBoard, vplayer = self.game.getNextState(vboard, vplayer, action)
            vcanonicalBoard = self.game.getCanonicalForm(vboard, -vplayer)
            vs = self.game.stringRepresentation( vcanonicalBoard)
           
            vstep += 1
            if analist == 1:
                if vs not in sVs.keys():
                    # edge
                    result = (vboard, 0, traj)
                    return result
            else:
                if vs not in bVs.keys():
                    # edge
                    result = (vboard, 0, traj)
                    return result
                
            
          
            if self.game.getGameEnded(vnextBoard, vplayer):
                # end
                result = (vnextBoard, -1, traj)
                return result
            
            
            
            vboard = vnextBoard
    
    def detect_relative_distance(self, pa, ca, limit=3):
        '''
        左から-1, 0, 1
        '''
        l = 0
        if ca < pa:
            l = -1
        elif ca > pa:
            l = 1
        
        return (l, min(abs(pa-ca), limit))
           
    
  


            

   

         
    







    
    
    
    






        
        

