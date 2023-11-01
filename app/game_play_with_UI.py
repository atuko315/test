from connectx_try import System
from connect4_game import Connect4Game
from gameUI import UImanager

sample_s_path = './best_200.pth.tar'
sample_b_path = './checkpoint_1.pth.tar'

game = Connect4Game()
strong_timellimit = 5
weak_timelimit = 0.5
strong_puct = 1
weak_puct = 0.1
sample_system = System(game, sample_s_path, sample_b_path, turn=1, strong_timelimit=strong_timellimit,
                        weak_timelimit=weak_timelimit, strong_puct=strong_puct, weak_puct=weak_puct)

ui = UImanager(game, sample_system)
ui.pack()
ui.mainloop()