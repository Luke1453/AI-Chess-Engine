import chess


class State(object):
    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board

    def key(self):
        return (self.board.board_fen(), self.board.turn,
        self.board.castling_rights, self.board.ep_square)

    # Serializes the board to get the training data
    def serialize(self):
        import numpy as np
        assert self.board.is_valid()

        # translating board into array of len = 8*8
        board_state = np.zeros(64, np.uint8)
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece is not None:
                # print(i, piece.symbol())
                board_state[i] = {"P": 1, "N": 2, "B": 3,
                                  "R": 4, "Q": 5, "K": 6,
                                  "p": 9, "n": 10, "b": 11,
                                  "r": 12, "q": 13, "k": 14}[piece.symbol()]

        # encoding castling rights into board state 
        # using 7(W) and 15(B) as the Rook symbol if it has castling rights
        if self.board.has_queenside_castling_rights(chess.WHITE):
            assert board_state[0] == 4
            board_state[0] = 7
        if self.board.has_kingside_castling_rights(chess.WHITE):
            assert board_state[7] == 4
            board_state[7] = 7
        if self.board.has_queenside_castling_rights(chess.BLACK):
            assert board_state[56] == 8+4
            board_state[56] = 8+7
        if self.board.has_kingside_castling_rights(chess.BLACK):
            assert board_state[63] == 8+4
            board_state[63] = 8+7

        # encoding empersant rights into board state 
        # using 8 to encode ep square
        if self.board.ep_square is not None:
            #chacking if ep square is empty
            assert board_state[self.board.ep_square] == 0
            board_state[self.board.ep_square] = 8
        
        # reshaping board to 8x8 array
        board_state = board_state.reshape(8, 8)

        # binary state
        state = np.zeros((5, 8, 8), np.uint8)

        # 0-3 columns to binary
        state[0] = (board_state >> 3) & 1
        state[1] = (board_state >> 2) & 1
        state[2] = (board_state >> 1) & 1
        state[3] = (board_state >> 0) & 1

        # 4th column is who's turn it is
        state[4] = (self.board.turn*1.0)
        
        # print(board_state)
        # print(state)
        
        # state data is stored in 5x8x8 bit array 
        # 3D 5 is treated as one bit cause only usefull info 
        # it encodes is whos turn it is 
        # 0 - black  1 - white
        # so board state is encoded in 8x8x4 +1 = 257 bits  
        return state

    def edges(self):
        return list(self.board.legal_moves)


if __name__ == "__main__":
    s = State()
