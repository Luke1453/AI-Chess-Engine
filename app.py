from flask import Flask, request
import chess
import time
import chess.svg
import traceback
from state import State
import torch
from train import Net

MAXVAL = 10000

# Valuator class that uses nn to evalueate the board
class Valuator(object):
    def __init__(self):

        vals = torch.load("nets/value1M.pth", map_location=lambda storage, loc: storage)
        self.model = Net()
        self.model.load_state_dict(vals)

    def __call__(self, bState):
        brd = bState.serialize()[None]
        output = self.model(torch.tensor(brd).float())
        return float(output.data[0][0])


# chess board, "engine" and flask app
state = State()
valuator = Valuator()
app = Flask(__name__)

def computer_minimax(s, v, depth, a, b, big=False):
    if depth >= 3 or s.board.is_game_over():
        return v(s)

    # white is maximizing player
    turn = s.board.turn
    if turn == chess.WHITE:
        ret = -MAXVAL
    else:
        ret = MAXVAL

    if big:
        bret = []

    # can prune here with beam search
    isort = []
    for e in s.board.legal_moves:
        s.board.push(e)
        isort.append((v(s), e))
        s.board.pop()
    move = sorted(isort, key=lambda x: x[0], reverse=s.board.turn)
    # beam search beyond depth 3
    if depth >= 3:
        move = move[:10]

    for e in [x[1] for x in move]:
        s.board.push(e)
        tval = computer_minimax(s, v, depth+1, a, b)
        s.board.pop()
        if big:
            bret.append((tval, e))
        if turn == chess.WHITE:
            ret = max(ret, tval)
            a = max(a, ret)
            if a >= b:
                break  # b cut-off
        else:
            ret = min(ret, tval)
            b = min(b, ret)
            if a >= b:
                break  # a cut-off
    if big:
        return ret, bret
    else:
        return ret


def explore_leaves(s, v):
    ret = []
    start = time.time()
    begining_eval = v(s)
    print("Human move Eval: ", begining_eval, flush=True)
    cval, ret = computer_minimax(s, v, 0, a=-MAXVAL, b=MAXVAL, big=True)
    eta = time.time() - start
    print("%.2f -> %.2f: explored %d moves in %.3f seconds" % (begining_eval, cval, len(ret), eta), flush=True)
    return ret


# def to_svg(s):
#     return base64.b64encode(chess.svg.board(board=s.board).encode('utf-8')).decode('utf-8')

@app.route("/")
def hello():
    print("get /", flush=True)
    ret = open("index.html").read()
    return ret.replace('start', state.board.fen())


def computer_move(state, valuator):
    # computer moves
    moves = sorted(explore_leaves(state, valuator), key=lambda x: x[0], reverse=state.board.turn)
    if len(moves) == 0:
        return
    
    # print(*moves, sep='\n', flush=True)
    print("top 3:", flush=True)
    for i, m in enumerate(moves[0:3]):
        print("  ", m, flush=True)
    print(state.board.turn, "moving", moves[0][1], flush=True)
    
    state.board.push(moves[0][1])




# moves given as coordinates of piece moved
@app.route("/move_coordinates")
def move_coordinates():
    if not state.board.is_game_over():
        source = int(request.args.get('from', default=''))
        target = int(request.args.get('to', default=''))
        promotion = True if request.args.get('promotion', default='') == 'true' else False

        move = state.board.san(chess.Move(source, target, promotion=chess.QUEEN if promotion else None))
        move_check = chess.Move(source, target, promotion=chess.QUEEN if promotion else None)

        # Checking if player move is legal
        if move_check not in state.board.legal_moves:
            print("illegal move", flush=True)
            response = app.response_class(response=state.board.fen(), status=0)
            return response

        # If move is legal AI makes its move
        if move is not None and move != "":
            print("human moves", move, flush=True)
            try:
                state.board.push_san(move)
                computer_move(state, valuator)
            except Exception:
                traceback.print_exc()

        response = app.response_class(response=state.board.fen(), status=200)
        return response

    print("GAME IS OVER")
    response = app.response_class(
        response="Game Over!",
        status=200
    )
    return response


@app.route("/newgame")
def newgame():
    print("Game was reset!", flush=True)
    state.board.reset()
    response = app.response_class(
        response=state.board.fen(),
        status=200
    )
    return response


if __name__ == "__main__":
    app.run()
