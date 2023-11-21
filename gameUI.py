import sys
import base64
import inspect
import os
import io
from submission_sample import SimpleAgent, dotdict, MCTS
import numpy as np
from tqdm import tqdm
from parl.utils import logger
from connectx_try import encode_weight, load_buffer, System
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
        self.memory  = {}

        self.turns = [0, 0, 0]

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
        self.fopen = False
        
        self.on_draw(self.board, self.text, self.c)
        print("draw")
        #self.play()
        
        #self.play()
    def reset(self):
        self.system.reset_mcts()
    
    def end_all(self):
        self.rwindow.destroy()
        self.destroy()
        
    def open_canvas(self):
    # 新しいキャンバスを作成
        if self.open:
            return
        
        
        
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
        
       
        self.memory.append(self.board.copy(), self.s_mcts.Nsa.copy(), self.b_mcts.Nsa.copy(), None, None, self.s_mcts.V.copy(), self.b_mcts.V.copy())
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
            self.board = self.game.getInitBoard() # ?
            self.on_draw(self.board, self.text, self.c)
            return
        
        #if player != turn:
        #    return 
        # 先手処理
        action = self.inputValidMove(self.board, player, event)
        self.memory.append(self.board.copy(), self.s_mcts.Nsa.copy(), self.b_mcts.Nsa.copy(), None, None, self.s_mcts.V.copy(), self.b_mcts.V.copy())
        self.board, next_player = self.game.getNextState(self.board.copy(), player, action)
        
        self.on_draw(self.board, self.text, self.c)
        return
     
      
        
        #self.master.alter(1, self.turn_of_human) # ?
    
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
    
    def forward_one(self):
        bstep = getStep(self.fboard)
        if bstep > len(self.memory):
            return
        
        self.foard = self.memory[bstep+1][0]
        text =  [[ "" for _ in range(self.width)] for _ in range(self.height)]
        self.on_draw(self.fboard, text, self.fc)
    
    def back_one(self):
        bstep = getStep(self.fboard)
        if bstep <= 0:
            return
        
        self.foard = self.memory[bstep-1][0]
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
               
                
    
    def show_traj(self, board, traj, by_step=True):
        '''
        boardは引数
        vboardをvtext付きで表示
        本編とは違うキャンバス(self.ac)に
        closeは自分で押してもらう
        '''
        
        self.ac.pack()
        self.bc.pack()
        vboard = board.copy()
        vtext = [[ "" for _ in range(self.width)] for _ in range(self.height)]
        for i in range(len(traj)):
            vboard, number = self.system.add_stone(vboard, getCurrentPlayer(vboard), traj[i], number=True)
            vtext[int(number/self.width)][number%self.width] = i+1
           
       
        if not self.open:
            self.close_button.pack()
        if not self.open:
            self.open = True
        
        self.on_draw(vboard, vtext, self.ac)
        return 
    

    def show_vector(self, board, text, key_c, vector, distance):
        '''
        abs(vector) < 0.5のときは両側に表示
        vectorが大きい側のdistance+-0.5の部分に色をつける
        distanceが2.5以上のときは端まで色をつける
        '''
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
            base = max(self.oval_size*((key_c/2) - distance), width)
            
            print(base, width)
            
            if distance < 2.5:
                self.ac.create_rectangle(base-width, 0, base+width, self.oval_size*self.height, width = 0.0, fill = 'green')
            
            else:
                self.ac.create_rectangle(0, 0, base+width, self.oval_size*self.height, width = 0.0, fill = 'green')
               
        if right:
            width = int(self.oval_size/2)
            base = min(self.oval_size*(distance + (key_c/2)), self.oval_size*self.width - width)
            

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
        
        
       
        






            
            

