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
from scipy.stats import norm, entropy
import tkinter as tk
game = Connect4Game()
init = game.getInitBoard()

class UImanager(tk.Frame):
    def __init__(self, game, system, master=None):
        tk.Frame.__init__(self, master)
        self.system = system
        self.game = game
        self.master.title("connect4")
        self.height, self.width = self.game.getBoardSize()

        self.turns = [1, 0, 0]

        self.oval_size = 40
        self.edge_width = 2
        self.board = self.game.getInitBoard()
        self.text = [[ "1" for _ in range(self.width)] for _ in range(self.height)]
        #　一回両方人間で作る
        self.c = tk.Canvas(self, width = self.oval_size*self.width, height = self.oval_size*self.height, highlightthickness = 0)
        self.c.bind('<Button-1>', self.turn_of_human)
        self.c.pack()
        self.ac = tk.Canvas(self, width = self.oval_size*self.width, height = self.oval_size*self.height, highlightthickness = 0)
        self.on_draw()
        
        #self.play()
        
        
    
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
        
        player = getCurrentPlayer(self.board)
        if self.turns[player+1] == 0:
            return
        canonicalboard = self.game.getCanonicalForm(self.board.copy(), player)
        if mode == "past":
            action = np.argmax(self.system.getPastCount(path, step, self.board, self.turns[player+1]))
        else:
            action =  np.argmax(self.system.s_mcts.getActionProb(canonicalboard, temp=0)) if self.turns[player+1] == 1 else np.argmax(self.system.b_mcts.getActionProb(canonicalboard, temp=0))
        
        print(action)
        self.board, next_player = self.game.getNextState(self.board.copy(), player, action)
        self.on_draw()
        
    def turn_of_human(self, event):
       
        player = getCurrentPlayer(self.board)
        if self.turns[player+1] != 0:
            return
        if self.game.getGameEnded(self.board, player) != 0:
            print(self.game.getGameEnded(self.board, player))
            self.board = self.game.getInitBoard() # ?
            self.on_draw()
            return
        
        #if player != turn:
        #    return 
        # 先手処理
        action = self.inputValidMove(self.board, player, event)
        self.board, next_player = self.game.getNextState(self.board.copy(), player, action)
        
        self.on_draw()
        
        #self.master.alter(1, self.turn_of_human) # ?
    
    def draw_piece(self, index):
        x = (index%self.width)
        y = int(index/self.width)
        dx = x * self.oval_size + self.edge_width
        dy = y * self.oval_size + self.edge_width

        if self.board[y][x] == 1:
            self.c.create_oval(dx, dy, dx+self.oval_size-self.edge_width*2, dy+self.oval_size-self.edge_width*2, width=1.0, fill = '#FF0000')
        elif self.board[y][x] == -1:
            self.c.create_oval(dx, dy, dx+self.oval_size-self.edge_width*2, dy+self.oval_size-self.edge_width*2, width=1.0, fill = '#FFFF00')
        
        if self.text[y][x] != "":
            self.c.create_text(dx+int(self.oval_size/2)-self.edge_width, dy+int(self.oval_size/2)-self.edge_width, text=self.text[y][x], font=("Helvetica", 10), fill="black")
    
    def on_draw(self):
        print(self.board)
        self.c.delete('all')
        self.c.create_rectangle(0, 0, self.oval_size*self.width, self.oval_size*self.height, width = 0.0, fill = '#00A0FF')
        for i in range(self.height*self.width):
            x = (i % self.width) * self.oval_size + self.edge_width
            y = int(i / self.width) * self.oval_size + self.edge_width
            self.c.create_oval(x, y, x+self.oval_size-self.edge_width*2, y+self.oval_size-self.edge_width*2, width=1.0, fill='#FFFFFF')
        
        for i in range(self.height*self.width):
            
            # AI入れるときは手番分けして
            self.draw_piece(i)
        
        player = getCurrentPlayer(self.board)
        print(self.turns[player+1])
        if self.turns[player+1] != 0:
            print("ai's turn")
            self.turn_of_AI()
            self.on_draw()
       
            
    
    def play(self):
        player = getCurrentPlayer(self.board)
        print(player)
        self.on_draw()
        if self.turns[player+1] != 0:
            self.turn_of_AI()
        
        self.on_draw()
        if self.game.getGameEnded(self.board, player) != 0:
            print(self.game.getGameEnded(self.board, player))
            self.board = self.game.getInitBoard() # ?
            self.on_draw()
            self.play()
            #return
    
    def show_traj(self, board, traj):
        self.board = board
        for i in range(len(traj)):
            def
            

