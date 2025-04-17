from Graphics import *
import numpy as np
import random
import torch

class State:
    def __init__(self, board=None, falling_piece=None, next_piece=4):
        self.board = self.init_board()
        self.end_of_game = False
        self.score = 0
        self.FALL_SPEED = 20
        self.fall_speed = self.FALL_SPEED
        self.falling_piece = falling_piece #(row, col, piece)
        self.next_piece = next_piece
    
    def init_board(self): # אתחול הלוח
            board = np.zeros((ROWS, COLS), dtype=int) # לוח מלא ב0
            return board

    def down (self):
        row, col, piece = self.falling_piece
        self.falling_piece = row + 1, col, piece

    def falling_piece_to_tensor (self):
         piece_t = torch.zeros((4,4), dtype=torch.float32)
         row, col, piece = self.falling_piece
         piece_t[0:piece.shape[0], 0:piece.shape[1]] = torch.tensor(piece.copy())
         result = torch.cat((torch.tensor([row]), torch.tensor([col]), piece_t.flatten()))
         return result

    def toTensor (self, device = torch.device('cpu')):
        array = self.board.reshape(-1)
        tensor = torch.tensor(array, dtype=torch.float32, device=device)

        # falling_piece_array = np.array(self.falling_piece, dtype=np.float32)
        tensor1 = self.falling_piece_to_tensor()

        tensor2 = torch.tensor([self.next_piece], dtype=torch.float32, device=device)

        tensor3 = torch.tensor([self.fall_speed], dtype=torch.float32, device=device)

        tensor = torch.cat((tensor, tensor1, tensor2, tensor3))

        return tensor.view(1, -1)    

    def board_bin (self): # הופך ללוח בינארי
        return (self.board !=0).astype(int)
        
    def get_board_w_piece(self): # הוספת החלק ללוח  
        board = np.copy(self.board)
        row, col, piece = self.falling_piece # מוצא את מיקום וצורת החלק
        rows, cols = piece.shape # מוצא את אורך ורוחב החלק
        board[row:row+rows, col:col+cols] += piece # מוסיף את החלק במקום המתאים על הלוח
        return board
    
    def add_piece (self):
        row, col, piece = self.falling_piece # מוצא את מיקום וצורת החלק
        rows, cols = piece.shape # מוצא את אורך ורוחב החלק
        self.board[row:row+rows, col:col+cols] += piece # מוסיף את החלק במקום המתאים על הלוח
        self.falling_piece = None # החלק כבר לא נופל

    def state_w_piece (self):
         state = self.copy()
         state.board = self.get_board_w_piece()
         return state
    
    def copy_falling_piece(self):
         row, col, piece = self.falling_piece
         new_fallingPiece = (row, col, piece.copy())   # Gilad
         return new_fallingPiece

    def copy (self):
         newBoard = np.copy(self.board)
         row, col, piece = self.falling_piece
         new_fallingPiece = (row, col, piece.copy())   # Gilad
         new_nextPiece = self.next_piece

         return State(board=newBoard, falling_piece=new_fallingPiece, next_piece=new_nextPiece)
    
    def __repr__(self):
         return str(self.board)