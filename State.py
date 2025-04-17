from Graphics import *
import numpy as np
import random
import torch

class State:
    def __init__(self, board=None, falling_piece=None, next_piece=None):
        if board is None:
            self.board = self.init_board()
        else:
            self.board = board            
        self.end_of_game = False
        self.score = 0
        self.FALL_SPEED = 20
        self.fall_speed = self.FALL_SPEED
        if falling_piece is None:
            self.init_falling_piece() 
        else:
            self.falling_piece = falling_piece
        if next_piece is None:
            self.next_piece = random.randint(1, 7)
        else:
            self.next_piece = next_piece
                
    def init_board(self): # אתחול הלוח
            board = np.zeros((ROWS, COLS), dtype=int) # לוח מלא ב0
            return board

    def init_falling_piece (self):
        falling_piece_id = random.randint(1, 7) # 
        falling_piece_shape = self.pieces()[falling_piece_id] # שומר את הצורה של החלק הנבחר
        falling_piece_col = COLS // 2 - len(falling_piece_shape[0]) // 2 #ממקם אותו באמצע השורה העליונה
        self.falling_piece = 0, falling_piece_col, falling_piece_shape # ממקם את החלק על הלוח

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
    
    def pieces(self):
        pieces = {
            1:np.array([[1, 1, 1, 1]]),
            2:np.array([[2, 0, 0],[2, 2, 2]]),
            3:np.array([[0, 0, 3],[3, 3, 3]]),
            4:np.array([[4, 4],[4, 4]]),
            5:np.array([[0, 5, 5],[5, 5, 0]]),
            6:np.array([[0, 6, 0],[6, 6, 6]]),
            7:np.array([[7, 7, 0],[0, 7, 7]])
        }
        return pieces

    def __repr__(self):
         return str(self.board)