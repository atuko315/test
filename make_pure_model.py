# -*- coding: utf-8 -*-
"""
Created on Mon May 15 19:02:29 202

@author: uguis
"""
import torch
from alphazero_agent import create_agent
from connect4_game import Connect4Game

game = Connect4Game()
pure_agent = create_agent(game)
print("create")
pure_agent.save("/home/student/PARL/benchmark/torch/AlphaZero/pure.pth.tar")