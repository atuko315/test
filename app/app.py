from flask import Flask, render_template,jsonify, request
import numpy as np
from connectx_try import System, getCurrentPlayer
from connect4_game import Connect4Game
sample_s_path = './best_200.pth.tar'
sample_b_path = './checkpoint_1.pth.tar'

game = Connect4Game()
strong_timellimit = 5
weak_timelimit = 0.5
strong_puct = 1
weak_puct = 0.1

game =  Connect4Game()
app = Flask(__name__)

# global 変数
board = game.getInitBoard()
memory = []
system = System(game, sample_s_path, sample_b_path, turn=1, strong_timelimit=strong_timellimit,
                        weak_timelimit=weak_timelimit, strong_puct=strong_puct, weak_puct=weak_puct)
turn = [-1, 0, 1]
# とりあえず人間先番、強いAI後番で

# 各セルのクラスを設定するための辞書
cell_class = {1: 'red', -1: 'yellow', 0: 'white'}

@app.route('/')
def connect4():
    return render_template('board.html', board=np.transpose(board), cell_class=cell_class)


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
    next_board, next_player = game.getNextState(board, getCurrentPlayer(board), action) 
    result = game.getGameEnded(next_board, next_player)
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
    global system
    data = request.get_json()
    board = np.array(data['board'], dtype=np.int32)
    player = getCurrentPlayer(board)
    canonicalboard = game.getCanonicalForm(board.copy(), player)
    action =  np.argmax(system.s_mcts.getActionProb(canonicalboard, temp=0))
    memory.append([board.copy(), system.s_mcts.Nsa.copy(), system.b_mcts.Nsa.copy(), None, None, system.s_mcts.V.copy(), system.b_mcts.V.copy()])
    next_board, next_player = game.getNextState(board, getCurrentPlayer(board), action) 
    result = game.getGameEnded(next_board, next_player)
    print(result)
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
    


def reset():
    global memory
    global system
    memory = []
    system.reset_mcts()

if __name__ == '__main__':
    app.run(debug=True)
