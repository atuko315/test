from flask import Flask, render_template,jsonify, request
import numpy as np
from connectx_try import System, getCurrentPlayer, getStep, saliency
from connect4_game import Connect4Game
import collections
from collections import defaultdict
import copy
import os
import pickle
import pathlib
import os
from pathlib import Path 
from pathlib import PosixPath

def load_data(path):
    with path.open(mode='rb') as f:
        return pickle.load(f)
sample_s_path = './best_200.pth.tar'
sample_b_path = './checkpoint_1.pth.tar'

game = Connect4Game()
strong_timellimit = 5
weak_timelimit = 0.5
strong_puct = 1
weak_puct = 0.1
analist = 1

game =  Connect4Game()
app = Flask(__name__)

phases = ["short", "middle", "long"]
imp_label = ["important", "trivial"]
labels = [[0, 0], [0, 1], [1, 0], [1, 1]]
label = 1
phase = 1
paths = os.path.join("../sotsuron/label", imp_label[labels[label][0]], imp_label[labels[label][0]], phases[phase])
print("start")
paths = sorted(Path(paths).glob('*.board'))

number = np.random.randint(0, len(paths))
print(number)
content = load_data(paths[number])

print(type(paths[number]))
if len(content) < 5:
    importance, board, brance, fpath = content
else:
    imp, board, branch, fpath, importance = content

point = getStep(board)
print("point")
print(point)
print(fpath)
fpath = PosixPath(os.path.join("../sotsuron", fpath))
benchmark = [("../sotsuron/offdata/20231110082733.history", 19), ("../sotsuron/offdata/20231109003421.history", 17)]
fpath, point = benchmark[1]
print(fpath, point)
fpath =  PosixPath(fpath)
memory = load_data(fpath)

memory = memory[0:len(memory)-1]




# global 変数
board = game.getInitBoard()
system = System(game, sample_s_path, sample_b_path, turn=1, strong_timelimit=strong_timellimit,
                        weak_timelimit=weak_timelimit, strong_puct=strong_puct, weak_puct=weak_puct)
turn = [-1, 0, 1]
answer = defaultdict(lambda:[])
# とりあえず人間先番、強いAI後番で

# 各セルのクラスを設定するための辞書
cell_class = {1: 'red', -1: 'yellow', 0: 'white'}

@app.route('/')
def connect4():
    return render_template('compare_board.html', board=np.transpose(board), cell_class=cell_class)


@app.route('/get_board')
def get_board():
    return jsonify({'board': board.tolist()})

@app.route('/update_board', methods=['POST'])
def update_board():
    global board
    global memory
    data = request.get_json()
    action = data['action']
    
    memory.append([board.copy(), system.s_mcts.Nsa.copy(), system.b_mcts.Nsa.copy(), None, None, system.s_mcts.V.copy(), system.b_mcts.V.copy()])
    s_value = get_past_value(board, getStep(board), analist=1)
    b_value = get_past_value(board, getStep(board), analist=-1)
    memory[-1][3] = s_value
    memory[-1][4] = b_value
    next_board, next_player = game.getNextState(board, getCurrentPlayer(board), action) 
    result = game.getGameEnded(board, next_player)
    print(result)
    if result != 0:
        next_board = game.getInitBoard()
        
    board = np.array(next_board, dtype=np.int32) # クライアントからのデータを受け取り
    print(board)
    # ボードを更新
    # 例: board = updated_board
    # 更新されたボードをクライアントに返す
    response_data = {
        'updatedBoard': board.tolist(),
        'gameResult': result,
        'player': next_player,
        # 他の情報をここで追加
    }
    return jsonify(response_data)
@app.route('/turn_of_AI', methods=['POST'])
def turn_of_AI():
    global board
    global memory
    print("memory", len(memory))
    data = request.get_json()
    board = np.array(data['board'], dtype=np.int32)
    #analist = data['analist']
    player = getCurrentPlayer(board)
    canonicalboard = game.getCanonicalForm(board.copy(), player)
    action = np.argmax(system.s_mcts.getActionProb(canonicalboard, temp=0))
    
    if analist == 1:
        action = np.argmax(system.s_mcts.getActionProb(canonicalboard, temp=0))
    else:
        action = np.argmax(system.b_mcts.getActionProb(canonicalboard, temp=0))
    
    #saction = np.argmax(system.s_mcts.getActionProb(canonicalboard, temp=0))
    #waction = np.argmax(system.b_mcts.getActionProb(canonicalboard, temp=0))
    #action = saction if analist == 1 else waction
    memory.append([board.copy(), system.s_mcts.Nsa.copy(), system.b_mcts.Nsa.copy(), None, None, system.s_mcts.V.copy(), system.b_mcts.V.copy()])
    s_value = get_past_value(board, getStep(board), analist=1)
    b_value = get_past_value(board, getStep(board), analist=-1)
    memory[-1][3] = s_value
    memory[-1][4] = b_value
    next_board, next_player = game.getNextState(board, getCurrentPlayer(board), action) 
    result = game.getGameEnded(board, next_player)
    if result != 0:
        next_board = game.getInitBoard()
        
    board = next_board # クライアントからのデータを受け取り
    print(board)
    # ボードを更新
    # 例: board = updated_board
    # 更新されたボードをクライアントに返す
    response_data = {
        'updatedBoard': board.tolist(),
        'gameResult': result,
        'player': next_player,
        # 他の情報をここで追加
    }
    return jsonify(response_data)
    

@app.route('/reset', methods=['POST'])
def reset():
    global memory
    global system
    global board
    global answer
    memory = []
    answer = defaultdict(lambda:[])
    system.reset_mcts()
    board = game.getInitBoard()
    player = 1
    response_data = {
        'board': board.tolist(),
        'player': player,
        # 他の情報をここで追加
    }
    return jsonify(response_data)

@app.route('/forward_one', methods=['POST'])
def forward_one():
    global memory
    global system
    global board
    print(len(memory))
    data = request.get_json()
    fboard = np.array(data['board'], dtype=np.int32)
    bstep = getStep(fboard)
    if bstep >= len(memory)-1:
        return jsonify({'board': fboard.tolist()})
    fboard = memory[bstep+1][0]
    counts = getPastCount(bstep, fboard, analist)
    s_value = memory[bstep+1][3]

    print(fboard)
    response_data = {
        'board': fboard.tolist(),
        'counts': counts,
        'value': s_value,        # 他の情報をここで追加
    }
    return jsonify(response_data)

@app.route('/back_one', methods=['POST'])
def back_one():
    global memory
    global system
    global board
    data = request.get_json()
    fboard = np.array(data['board'], dtype=np.int32)
    bstep = getStep(fboard)
    counts = []
    s_value = 0
    if bstep <= 0:
        return jsonify({'board': fboard.tolist()})
    else:
        counts = getPastCount(bstep, fboard, analist)
        s_value = memory[bstep-1][3]
    fboard = memory[bstep-1][0]
    
    print(fboard)
    response_data = {
        'board': fboard.tolist(),
        'counts': counts, 
        'value': s_value,
        # 他の情報をここで追加
    }
    return jsonify(response_data)
@app.route('/saliency_map', methods=['POST'])
def saliency_map(analist=1, mode="policy"):
    global system
    data = request.get_json()
    analist = data['analist']
    agent = system.s_mcts if analist == 1 else system.b_mcts
    fboard = np.array(data['board'], dtype=np.int32)
    boards = [system.removeStone(fboard.copy(), i) for i in range(42)]
    if mode == "value":
            #手番の問題で裏返し
            saliencies = saliencies = [saliency(
                -agent.nn_agent.predict(game.getCanonicalForm(fboard, getCurrentPlayer(fboard)))[1],
                agent.nn_agent.predict(game.getCanonicalForm(boards[i], getCurrentPlayer(boards[i])))[1]
                ) for i in range(42)]   
    else:
        saliencies = [saliency(
            agent.nn_agent.predict(game.getCanonicalForm(fboard, getCurrentPlayer(fboard)))[0],
            agent.nn_agent.predict(game.getCanonicalForm(boards[i], getCurrentPlayer(boards[i])))[0]
            ) for i in range(42)]   
    saliencies = np.array(saliencies).reshape(6, 7)
    max_value = saliencies[int(np.argmax(saliencies)/7)][np.argmax(saliencies)%7]
    saliencies /= max_value
    print(saliencies)
    response_data = {
        'board': fboard.tolist(),
        'saliency': saliencies.tolist()
        # 他の情報をここで追加
    }
    return jsonify(response_data)
@app.route('/fatal_map', methods=['POST'])
def fatal_map(analist=1):
        global answer
        data = request.get_json()
        agent = system.s_mcts
        fboard = np.array(data['board'], dtype=np.int32)
        analist = data['analist']
        key = getStep(fboard) * analist
        if not answer[key]:
            answer[key] = hot_states_one_way(fboard, analist, step=4, baseline=2, fix=-1)
        bfcount, bfdcount, new_trajs, gs4, gd2, groups, visual = answer[key]
        visual = np.array(visual)
        max_value = visual[int(np.argmax(visual)/7)][np.argmax(visual)%7]
        visual = visual.tolist()
        visual /= max_value
        print(visual)
        response_data = {
        'board': fboard.tolist(),
        'fatals': visual.tolist()
        # 他の情報をここで追加
        }
        return jsonify(response_data)
@app.route('/start_feedback', methods=['POST'])
def start_feedack(analist=1):
    fboard = memory[point][0]
    print(fboard, point)
    response_data = {
        'board': fboard.tolist(),
    }
    counts = getPastCount(point, fboard, analist)
    print(counts)
    return jsonify(response_data)

@app.route('/traj_plus', methods=['POST'])
def traj_plus():
    data = request.get_json()
    fboard = np.array(data['board'], dtype=np.int32)
    traj = data['traj']
    text = [[ -1 for _ in range(7)] for _ in range(6)]
    vboard = fboard.copy()
    for i in range(len(traj)):
            vboard, number = system.add_stone(vboard, getCurrentPlayer(vboard), traj[i], number=True)
            text[int(number/7)][number%7] = i+1
    #あとでｖに
    fatal = system.detectFatalStone(vboard)
    response_data = {
        'board': vboard.tolist(),
        'text': text,
        'fatal':fatal,
        'tail':len(traj),
        # 他の情報をここで追加
    }

    return jsonify(response_data)


@app.route('/show_traj', methods=['POST'])
def show_traj(analist=1, mode="group"):
    global system
    data = request.get_json()
    agent = system.s_mcts
    fboard = np.array(data['board'], dtype=np.int32)
    vboard = fboard.copy()
    trajs = check_frequent_traj(fboard.copy(), analist=1, mode=mode)
    print(trajs, len(trajs))
    text = [[ -1 for _ in range(7)] for _ in range(6)]
    vboard = fboard.copy()
    if trajs:
        min_traj = extract_min(trajs)
        print(min_traj)
        for i in range(len(min_traj)):
            vboard, number = system.add_stone(vboard, getCurrentPlayer(vboard), min_traj[i], number=True)
            text[int(number/7)][number%7] = i+1
    #あとでｖに
    fatal = system.detectFatalStone(vboard)
    response_data = {
        'board': vboard.tolist(),
        'text': text,
        'fatal':fatal,
        'tail':len(min_traj),
        # 他の情報をここで追加
    }

    return jsonify(response_data)
@app.route('/show_vec', methods=['POST'])
def show_vec(analist=1):
    data = request.get_json()
    fboard = np.array(data['board'], dtype=np.int32)
    vector, distance, metric = hot_vector_one_way(fboard)
    print(vector, distance)
    key_c = system.detectAction(memory[getStep(fboard)-1][0], fboard)
    response_data = {
        'board': fboard.tolist(),
        'distance': distance,
        'vector': vector,
        'key_c': key_c,
    }

    return jsonify(response_data)

@app.route('/get_valids', methods=['POST'])
def get_valids():
    data = request.get_json()
    fboard = np.array(data['board'], dtype=np.int32)
    valids = system.game.getValidMoves(fboard, getCurrentPlayer(fboard))
    valids = [i  for i in range(len(valids)) if valids[i]]
    response_data = {
        'valids': valids,
    }
    print(valids)
    return jsonify(response_data)

@app.route('/hot_traj', methods=['POST'])
def hot_traj(analist = 1):
    
    data = request.get_json()
    action = data['action']
    fboard = np.array(data['board'], dtype=np.int32)
    bstep = getStep(fboard)
    analist = data['analist']
    traj = [action]
    vboard, _ = game.getNextState(fboard, getCurrentPlayer(fboard), action)
    _, _, atraj = detectHotState(vboard, analist, getStep(vboard)-1) # 一手勝手に打ってるから
    vboard = fboard.copy()
    traj.extend(atraj)
    text = [[ -1 for _ in range(7)] for _ in range(6)]
    
    vboard, number = system.add_stone(vboard, getCurrentPlayer(vboard), traj[0], number=True)
    text[int(number/7)][number%7] = 0
    value = get_past_value(fboard, bstep, analist=1)
    if traj:
        for i in range(1, len(traj)):
            vboard, number = system.add_stone(vboard, getCurrentPlayer(vboard), traj[i], number=True)
            text[int(number/7)][number%7] = i+1
    fatal = system.detectFatalStone(vboard)
    
    response_data = {
        'board': vboard.tolist(),
        'text': text,
        'fatal': fatal, 
        'tail': len(traj),
        'value': value,
    }
    return jsonify(response_data)

@app.route('/my_hot_traj', methods=['POST'])
def my_hot_traj(analist = 1, mode="group", tail=3):
    action = -1
    data = request.get_json()
    action = data['action']
    fboard = np.array(data['board'], dtype=np.int32)
    bstep = getStep(fboard)
    analist = data['analist']
    #print(fboard)
    bstep = getStep(fboard)
    if action != -1:
        print(action)
        traj = [action]
    else:
        traj = []
    vboard, _ = game.getNextState(fboard, getCurrentPlayer(fboard), action)
    trajs = my_hot_traj_sub(vboard, bstep, analist=analist, mode=mode) # 一手勝手に打ってるから
    #print(trajs)
    new_trajs = []
    if action != -1:
        for i in range(len(trajs)):
            tmp = [action]
            t = trajs[i].copy()
            print(t)
            if t:
                tmp.extend(t)
                print(tmp)
                tmp = np.array(tmp, dtype=np.int32).tolist()
                new_trajs.append(tmp)
        
    else:
        new_trajs = trajs.tolist()
    #print(new_trajs) 
    min_traj = extract_min(new_trajs)
    #print(min_traj)
    traj = min_traj
    vboard = fboard.copy()
    text = [[ -1 for _ in range(7)] for _ in range(6)]
    
    
    if traj[0] != -1:
        #print(traj)
        vboard, number = system.add_stone(vboard, getCurrentPlayer(vboard), traj[0], number=True)
        value = get_past_value(vboard, bstep, analist=1)
        if len(traj)>0: 
            text[int(number/7)][number%7] = 0
            for i in range(1, len(traj)):
                vboard, number = system.add_stone(vboard, getCurrentPlayer(vboard), traj[i], number=True)
                
                text[int(number/7)][number%7] = i+1
                
    fatal = system.detectFatalStone(vboard)
    #new_trajs = np.array(new_trajs, dtype=np.int32).tolist()
    print(vboard)
    
    size = len(traj)
    response_data = {
        'board': vboard.tolist(),
        'text': text,
        'fatal':fatal,
        'tail': size,
        'trajs':new_trajs,
        'value':value,
    }
    return jsonify(response_data)



def extract_min(trajs):
    min = 100
    min_traj = []
   
    for t in trajs:
        tmp = len(t)
        if len(t) < min:
            min_traj = t.copy()
            min = len(t)
    return min_traj
    
def my_hot_traj_sub(fboard, bstep, analist=1, step=2, baseline=4, mode="group"):
    answer = hot_states_one_way(fboard.copy(), analist=analist, step=step, baseline=baseline, fix=bstep)
    if not answer:
        return []
    bfcount, bfdcount, new_trajs, gs4, gd2, groups, visual = answer
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


def check_frequent_traj(fboard, analist=1, mode="group"):
    global answer
    key = getStep(fboard) * analist
    if not answer[key]:
        answer[key] = hot_states_one_way(fboard.copy(), analist=analist, step=4, baseline=2, fix=-1)
    bfcount, bfdcount, new_trajs, gs4, gd2, groups, visual = answer[key]
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

def hot_states_one_way(fboard, analist=1, step=4, baseline=2, fix=-1):
    reach = detect_actual_reach()
    bstep = getStep(fboard)
    if fix != -1:
        bstep = fix
    
    latest = system.detectAction(memory[bstep-1][0], memory[bstep][0])
    key_c = latest % 7
    btrajs, boards = collect_promising_vector_sub(fboard, key_c, analist, step, baseline=baseline, fix=fix, mode="normal")
    new_trajs = []
        
    if not btrajs:
        return None
    
    bfcount, bfdcount, trajs, gs4, gd2, groups, visual = check_convergence(boards, reach, bstep,  analist, btraj=btrajs)
    return bfcount, bfdcount, trajs, gs4, gd2, groups, visual

def check_convergence(boards, reach, bstep,  analist, btraj=None):
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
        group, stones, traj = check_convergence_per_board(b, reach, bstep, analist)
        print(group, traj)
        if btraj[index]:
            if traj:
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
                    for i in range(42)]
    visual = np.array(visual).reshape(6, 7)
    
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

def check_convergence_per_board(board, reach, bstep, analist):
        bcount = 0
        bfcount = 0
        bfdcount = 0
        print(board)
        hot = detectHotState(board.copy(), analist, bstep) # mode=traj, toend=True
        print(hot)
        #print(hot[1])
        if hot[1] == None:
            return None, None, None
            
        end = game.getGameEnded(hot[0], getCurrentPlayer(hot[0]))
        if abs(end) == 1:
            
            bcount = 1
            fatal = system.detectFatalStone(hot[0], per_group=True)
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

def collect_promising_vector(fboard, key_c, analist, step, baseline, fix = -1, mode="normal"):
        trajs, boards = collect_promising_vector_sub(fboard, key_c, analist, step, baseline, fix = getStep(fboard), mode=mode)
        return trajs

    

def collect_promising_vector_sub(boards, key_c, analist, step=2, baseline=4, fix=-1, mode="normal"):
        new_trajs = []
        new_boards = []
        if step == 1:
            board = boards
            boards = collect_promising_per_step(board, analist, baseline=baseline, fix=fix)
            for b in boards:
                a = system.detectAction(board, b)
        
                if mode != "vector":
                    if a != -1:
                        new_trajs.append([a])
                    continue
                relative = detect_relative_distance(key_c, a)
                new_trajs.append([relative])
        
            return new_trajs, boards
        
        ftrajs, fboards = collect_promising_vector_sub(boards, key_c, analist, step-1, baseline, fix=fix, mode=mode)
       
        for i in range(len(ftrajs)):
            traj = copy.deepcopy(ftrajs[i])
            b = fboards[i]
            
            nboards = collect_promising_per_step(b, analist, baseline=baseline, fix=fix)
            nboards = np.array(nboards)
            if nboards.shape == (6, 7):
                nboards = nboards[np.newaxis]
               
            
            for nb in nboards:
                traj = copy.deepcopy(ftrajs[i])
                a = system.detectAction(b, nb)
                if a != -1:
                    if mode != "vector":
                        traj.append(a)
                        
                    else:
                        relative = detect_relative_distance(key_c, a)
                        traj.append(relative)
                new_trajs.append(traj)
                
            new_boards.extend(nboards)
        
        return new_trajs, new_boards

def collect_promising_per_step(board, analist, baseline=4, fix=-1):
    if game.getGameEnded(board, getCurrentPlayer(board)):
        fboards = []
        for i in range(baseline):
            fboards.append(board)
        return fboards

    #    return board こっちが終わり切り捨て番

    max_step = len(memory) - 1
    if analist == 0:
        analist = getCurrentPlayer(board)
    valid = game.getValidMoves(board, getCurrentPlayer(board))
    valid = [i  for i in range(len(valid)) if valid[i]]
    l = len(valid) if len(valid) < baseline else baseline
    #print(fpath, getStep(board))
    bstep = getStep(board)
    bstep = max_step if bstep > max_step else bstep
    if fix != -1:
        bstep = fix
    
    counts =  getPastCount(bstep, board, analist)
    counts = np.argsort(np.array(counts))

    counts = [c for c in counts if c in valid]
    counts = counts[-l:]
    fboards = []

    for c in counts:
            fboards.append(system.add_stone(board.copy(), getCurrentPlayer(board), c))

    return fboards

def detect_actual_reach():
        global memory
        last = memory[len(memory)-1]
        fboard, sNsa, bNsa, sv, bv, sVs, bVs = last
        
        valid = game.getValidMoves(fboard, getCurrentPlayer(fboard))
        valid = [i  for i in range(len(valid)) if valid[i]]
        reach = []
        for a in valid:
            vboard = system.add_stone(fboard.copy(), getCurrentPlayer(fboard), a)
            vf = system.detectFatalStone(vboard, per_group=True)
            if vf:
                reach.extend(vf)
        
        return reach

def getPastCount(bstep, board, analist):
       
        zboard, sNsa, bNsa, sv, bv, sVs, bVs = memory[bstep]
        curPlayer = getCurrentPlayer(board)
       
        canonicalBoard= game.getCanonicalForm(board, curPlayer)
        s = game.stringRepresentation(canonicalBoard)
        
        if analist == 1:
            counts = [
                sNsa[(s, a)] if (s, a) in sNsa else 0
                for a in range(game.getActionSize())
            ]
        else:
            
            counts = [
                bNsa[(s, a)] if (s, a) in bNsa else 0
                for a in range(game.getActionSize())
            ]
        return counts


def hot_vector_one_way(fboard,  analist=1, step=2, baseline=4, fix=-1):
    assert step > 0
    reach = detect_actual_reach()
    #print(reach)
    bstep = getStep(fboard)
    if fix != -1:
        bstep = fix
    
    assert bstep > 0
    
    
    latest = system.detectAction(memory[bstep-1][0], fboard)
    key_c = latest % 7
    
    
    
    trajs = collect_promising_vector(fboard, key_c, analist, step, baseline=baseline, fix = bstep, mode="vector")
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
    print(vector, distance, metric)

    return vector, distance, metric

def detectHotState(board, analist, step):
        zboard, sNsa, bNsa, sv, bv, sVs, bVs = memory[step]
        zflag = True if analist == 0 else False

        
        traj = []

        curPlayer = getCurrentPlayer(board)
        
        vboard = board.copy()
        vcanonicalBoard = game.getCanonicalForm(vboard, curPlayer)
        vs = game.stringRepresentation(vcanonicalBoard)
        
        vstep = getStep(board) # countは差分で得られる
        if zflag:
            analist = getCurrentPlayer(board)
        counts = getPastCount(step, vboard, analist)
        if sum(counts) == 0:
                # edge
                result = (vboard, 0, traj)
                return result
                
        #print(self.getPastValueNoModification( path, step, vboard, 1))
        if game.getGameEnded(board, curPlayer):
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
           
            valids = game.getValidMoves(vboard, vplayer)
            counts = getPastCount(step, vboard, analist)

            if sum(counts) == 0:
                # edge
                result = (vboard, 0, traj)
                return result
            

            action = np.argmax(counts)
            traj.append(action)
            #print(traj)
                
            if valids[action] == 0:
                result = (vboard, None, traj)
                return result
            
            
                
            vnextBoard, vplayer = game.getNextState(vboard, vplayer, action)
            vcanonicalBoard = game.getCanonicalForm(vboard, -vplayer)
            vs = game.stringRepresentation( vcanonicalBoard)
           
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
                
            
          
            if game.getGameEnded(vnextBoard, vplayer):
                # end
                result = (vnextBoard, -1, traj)
                return result
            
            
            
            vboard = vnextBoard

def detect_relative_distance(pa, ca, limit=3):
        '''
        左から-1, 0, 1
        '''
        l = 0
        if ca < pa:
            l = -1
        elif ca > pa:
            l = 1
        
        return (l, min(abs(pa-ca), limit))

def get_past_value(fboard, step, analist=1):
    zboard, sNsa, bNsa, sv, bv, sVs, bVs = memory[step]
    canonicalboard = game.getCanonicalForm(fboard, getCurrentPlayer(fboard))
    s = game.stringRepresentation(canonicalboard)
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
                cp, cv = system.s_mcts.nn_agent.predict(canonicalboard)
                return cv.astype(np.float32).tolist()[0]
    else:
        if s in bVs.keys():
            
            if type(bVs[s]) == int:
                return float(bVs[s]) 
            elif type(bVs[s]) == np.ndarray:
                return bVs[s].astype(np.float32).tolist()[0]
            
            return bVs[s]
        else:
            cp, cv = system.b_mcts.nn_agent.predict(canonicalboard)
            return cv.astype(np.float32).tolist()[0]

    
           


    


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 80))
    print(port)
    app.run(host='0.0.0.0', port=80, debug=False)
    
