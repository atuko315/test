
import numpy as np
from feature import DatasetManager
from pathlib import Path
from connectx_try import load_data, getCurrentPlayer, System, getStep
from connect4_game import Connect4Game, Board
from collections import defaultdict
from random import choice
game = Connect4Game()
size = len(sorted(Path('./label/important/important/short').glob('*.board')))
paths = sorted(Path('./label/important/important/short').glob('*.board'))[-size: ]

_, board, _, _, _  = load_data(paths[0])
dataset = DatasetManager(game, paths)
pattern = np.array(
        [
         [0, 0, 0],
         [0, 1, 0],
         [0, 1, 0],
         [0, 1, 0]]
        )
dataset.make_board_set()
#print(len(dataset.board_set))
#print(dataset.accumulate_pattern(pattern))
#print("trivial/trivial")
#horizon, vertical, diagonal_1, diagonal_2 = dataset.accumulate_variance()
#print(f"{horizon}\n{vertical}\n{diagonal_1}\n{diagonal_2}")
#print(dataset.multiple_accumulate_pattern())

def simple_run_length(board, mode='horizon'):
        '''
        列または行または斜めの行ごとに石が何個あるかを返す
        両方左下から返す

        diagonal_1:左下
        diagonal_2:右下
        '''
        print(mode)
        height = len(board)
        width = len(board[0])
        win_length = 4
        encoding = []
        
        if mode == "diagonal_1" :
            maximum_number = height * width
            i = 0
            while i < maximum_number:
                print(i)
                if mode == "diagonal_1":
                    seq = getDiagonalNumber(board, i, mode= "left_down")
                else:
                    seq = getDiagonalNumber(board, i, mode= "right_down")
                if seq >= 0:
                    encoding.append(seq)
                
                i = i+1 if i < width-1 else i+width
            return encoding
        elif mode == "diagonal_2":
            maximum_number = height * width
            i = width * (height-1)
            while i >= 0:
                seq = getDiagonalNumber(board, i, mode= "right_down")
                if seq >= 0:
                    encoding.append(seq)
                
                i = i - width
            for i in range(1, width):
                seq = getDiagonalNumber(board, i, mode= "right_down")
                if seq >= 0:
                    encoding.append(seq)
            return encoding
        f, s = width, height
        if mode == "vertical":
            board = np.transpose(board)
            f, s = height, width
        
        
        for i in range(f):
            print(f"f{i}")
            extraction = np.array([board[j][i] for j in range(s)])
            count = np.count_nonzero(extraction != 0)
            encoding.append(count)
        
        if mode == "vertical":
            encoding.reverse()

        return encoding
    
def getDiagonalNumber( board, n, check=False, mode="right_down"):
    height = len(board)
    width = len(board[0])
    h = int(n/width)
    w = n % width
    #右下
    if mode == "right_down":

        
        number = w
        record = []
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
        
        
    
    else:
        #左下
        if (width -1 - w< h):
            number = width -1 - w
            sw = width -1 
            sh = h - (width - 1 - w)
        else:
            number = h
            sw = w + h
            sh = 0
        
        record = []
        while True:
            if sh > height - 1 or sw < 0 :
                break
            record.append(board[sh][sw])
            sw -= 1
            sh += 1
        
        
    record = np.array(record)
    if len(record) < 4:
        return -1
    return np.count_nonzero(record != 0)

board1 = np.array(
[[ 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 0,-1, 0, 0, 0],
 [ 0, 0, 0, 1, 0, 0, 0],
 [ 0, 0, 0,-1, 0, 1, 0],
 [ 0, 0, 1,-1, 0, 1, 0]], dtype=np.int32)     
        

board4 = np.array(
[[ 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 0, 1, 0, 0, 0],
 [ 0, 0, 0,-1, 0, 0, 0],
 [ 0, 0, 0, 1, 0, 0, 0],
 [ 0, 0, 0,-1, 0, 0, 0],
 [ 0, 0, 1, 1, 0, 1, 0]], dtype=np.int32)

zboard = np.array(
[[ 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 0, 0, 0, 0, 0],
 [ 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32)


path = sorted(Path('./label/important/important/short').glob('*.board'))[-2]
game = Connect4Game()
strong_timellimit = 5
weak_timelimit = 0.5
strong_puct = 1
weak_puct = 0.1
sample_s_path = '/home/student/PARL/benchmark/torch/AlphaZero/best_200.pth.tar'
sample_b_path = '/home/student/PARL/benchmark/torch/AlphaZero/saved_model/checkpoint_1.pth.tar'
sample_system = System(game, sample_s_path, sample_b_path, turn=1, strong_timelimit=strong_timellimit,
                        weak_timelimit=weak_timelimit, strong_puct=strong_puct, weak_puct=weak_puct)
size1 = len(sorted(Path('./label/important/important/short').glob('*.board')))
size4 = len(sorted(Path('./label/trivial/trivial/short').glob('*.board')))
paths1 = sorted(Path('./label/important/important/short').glob('*.board'))[-size1: ]
paths4 = sorted(Path('./label/trivial/trivial/short').glob('*.board'))[-size4: ]
dataset1 = DatasetManager(game, paths1)
dataset4 = DatasetManager(game, paths4)
#print(dataset1.match_pattern(board, pattern))
dataset1.make_board_set()
dataset4.make_board_set()
#print(size1)
height, width = game.getBoardSize()

path1 = dataset1.retrieve_board(board1)
path4 = dataset4.retrieve_board(board4)
#print(len(path1))

imp, board, branch, fpath1, importance = load_data(path1[0])
_, _, _, fpath4, _ = load_data(path4[0])
#l1 = dataset1.label_board(board, fpath1, getStep(board), sample_system)
#l4 = dataset4.label_board(board, fpath4, getStep(board), sample_system)
#print(f"olabel: {l1}, label by another: {l4}")
h = load_data(fpath4)
h1 = load_data(fpath1)
board, _, _, _, _, _, _ = h[0]
b1, _, _, _, _, _, _ = h1[0]
board = np.array(board)
b = board4.copy()
curPlayer = getCurrentPlayer(board)



#zboard = np.zeros([6, 7], dtype=np.int32)

s2 = game.stringRepresentation(game.getCanonicalForm(b1, curPlayer))
s1 = game.stringRepresentation(game.getCanonicalForm(zboard, curPlayer))

'''
print(curPlayer)
print(board.shape, board4.shape)
print(type(board)==type(board4))
print(board.tolist()==board4.tolist())
print(game.getCanonicalForm(board4, curPlayer).tolist()==game.getCanonicalForm(board, curPlayer).tolist())
print(board.tostring()==board4.tostring())
print(board)
print(board4)
s2 = game.stringRepresentation(game.getCanonicalForm(board4, curPlayer))
s1 = game.stringRepresentation(game.getCanonicalForm(board, curPlayer))

print(board is board4)
curPlayer = getCurrentPlayer(board4)
print(curPlayer)

print(s1==s2)
print(s2)
print(s1)
'''
print(sample_system.detectHotState(board1, 1, fpath1, 7))
print(sample_system.detectHotState(board4, 1, fpath1, 7))
'''
for i in range(10):
    print(i)
    board, _, _, _, _, _, _ = h[i]
    print(sample_system.detectHotState(board, 1, fpath4, i))
    #print("*******")
    #print(sample_system.detectImpactState(board, 1, fpath4, i))
'''

'''
for i in range(5):
    print("change")
    change = 1
    new_board = dataset4.generate_alternative_board(board, change=change, verbose=True)
    label = dataset4.label_board(new_board, fpath, getStep(board), sample_system)
    print(f"new_label: {label}")
'''
'''
print(dataset1.pattern_set[-4])
data1 = dataset1.make_pattern_set(dataset1.pattern_set[0])
data4 = dataset4.make_pattern_set(dataset1.pattern_set[0])
distribution_1 = np.array([len(data1[i]) for i in range(height * width) ]).reshape(-1, width)
distribution_4 = np.array([len(data4[i]) for i in range(height * width) ]).reshape(-1, width)

print(distribution_1)
print()
print(distribution_4)

factor = 31
d1 = defaultdict(lambda: [])
d4 = defaultdict(lambda: [])

for board in data1[factor]:
    #print(board)
    step = getStep(np.array(board.copy()))
    
    d1[step].append(np.array(board))
for board in data4[factor]:
    step = getStep(np.array(board.copy()))
    d4[step].append(np.array(board))

print([i for i in range(height * width) if len(d1[i])!=0])
print([i for i in range(height * width) if len(d1[i])!=0])
for j in [i for i in range(height * width) if len(d1[i])!=0]:
    print(f"step: {j}")
    print("1: ")
    print((d1[j]))
    print("**********************")
    print("4: ")
    print((d4[j]))
    print("================")
'''

#dataset.generate_alternative_board(path, sample_system, "alter/", verbose=True)

#print(simple_run_length(board, mode="diagonal_1"))

#print(simple_run_length(board, mode="diagonal_2"))
#print(getDiagonalNumber(board, 1))