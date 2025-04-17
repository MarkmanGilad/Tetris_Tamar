from Graphics import *
import numpy as np
import random
import torch

class State:
    def __init__(self, board=None, falling_piece=None, next_piece=None):
        if board is not None:
            self.board = board
        else:
            self.board = self.init_board()
        self.end_of_game = False
        self.score = 0
        self.FALL_SPEED = 20
        self.fall_speed = self.FALL_SPEED
        if falling_piece == None:
            self.falling_piece = None #(row, col, piece)
        else:
             self.falling_piece = falling_piece
        if next_piece == None:
            self.next_piece = 4
            # self.next_piece = random.randint(1, 7) # בוחר מה יהיה החלק הבא
        else:
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
    
    def copy (self):
         newBoard = np.copy(self.board)
         row, col, piece = self.falling_piece
         new_fallingPiece = (row, col, piece)
         new_nextPiece = self.next_piece

         return State(board=newBoard, falling_piece=new_fallingPiece, next_piece=new_nextPiece)