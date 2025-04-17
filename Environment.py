import pygame
import numpy as np
from Graphics import *
from State import *
import random
state = State()


class Environment():

    landing = 20

    def __init__(self, state=None):
        self.state:State = state
        self.train = True
        self.reward = 0

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
    
    def select_falling_piece(self, state): #בוחר חלק רנדומלי
        falling_piece_id = state.next_piece # לוקח את החלק שנבחר קודם
        falling_piece_shape = self.pieces()[falling_piece_id] # שומר את הצורה של החלק הנבחר
        falling_piece_col = COLS // 2 - len(falling_piece_shape[0]) // 2 #ממקם אותו באמצע השורה העליונה
        state.falling_piece = 0, falling_piece_col, falling_piece_shape # ממקם את החלק על הלוח
        state.next_piece = random.randint(1, 7) # בוחר חלק הבא חדש
        # state.next_piece = 4

    def down_piece (self, state): # הורדת שורה
        state.down() #החלק יורד שורה

    def move (self, state, action): # הזזת החלק
        row, col, piece = state.falling_piece

        if action == 0:
            return # לא קורה כלום
        if not self.no_move(state, action):
            if action == 1:
                state.falling_piece = row, col - 1, piece # הזזה שמאלה
            elif action == 2:
                state.falling_piece = row, col + 1, piece # הזזה ימינה
            elif action == 3:
                state.falling_piece = row, col, np.rot90(piece, k=3) # סיבוב
            elif action == 4:
                state.fall_speed = 4 # הגדלת מהירות הנפילה
            elif action == 5:
                state.fall_speed = state.FALL_SPEED #החזרת מהיות הנפילה לרגיל

    def no_move(self, state, action): # בודק אם ,תזוזה אפשרית 
        row, col, piece = state.falling_piece
        is_no_move = False

        if action == 0:
            pass # לא קורה כלום
        elif action == 1:
            is_no_move = self.is_collision(state, falling_piece=state.falling_piece, dCol=-1) # בודקים אם תהיה התנגשות בהזזה שמאלה
        elif action == 2:
            is_no_move = self.is_collision(state, falling_piece=state.falling_piece, dCol=1) # בודקים אם תהיה התנגשות בהזזה ימינה
        elif action == 3:
            falling_piece = row, col, np.rot90(piece, k=3)
            is_no_move = self.is_collision(state, falling_piece=falling_piece) # בודקים אם תהיה התנגשות בסיבוב
        
        return is_no_move

    def is_collision(self, state, falling_piece, dRow = 0, dCol = 0): # בודק אם תהיה התנגשות
        row, col, piece = falling_piece # מציאת מיקום וצורת החלק
        rows, cols = piece.shape # מציאת אורך ורוחב החלק

        if row + rows + dRow > ROWS or col + cols + dCol > COLS or col + dCol < 0: # אם יגיע לרצפה או לאחד הצדדים 
            return True
        # check if will collide
        result = (state.board[row+dRow:row+rows+dRow, col+dCol:col+cols+dCol] * piece).sum() # בודק אם במקרה של הזזה סכום מכפלת האזור בלוח והחלק עצמו יהיה גדול מאפס
        
        if result > 0: # אם כן זאת אומרת שכבר יש שם חלק
            return True
        else:
            return False
        
    def reached_top(self, state): # בודק אם חלק הגיע עד למעלה
        row, col, piece = state.falling_piece
        if self.is_collision(state, falling_piece=state.falling_piece, dRow=1) and row == 0: # אם הוא נקתע וגם נמצא בשורה העליונה
            # print(state.score) # מדפיס את הניקוד
            return True # מחזיר אמת - הגיע עד למעלה
        return False
    
    def is_done(self, state):
        heights = self.columns_heights(state)
        return 20 in heights

    def clear_rows(self, state): # מחיקת שורות מלאות מהלוח
        temp_board = state.board.copy() # יצירת לוח זמני
        temp_board[temp_board != 0] = 1 # החלפת כל המספרים שלא 0 ב1
        row_sums = np.sum(temp_board, axis=1) # מציאת הסכום של כל שורה
        full_rows = np.where(row_sums == 10)[0] # יצירת מערך עם מספרי השורות המלאות - סכום 10
              

        for i in full_rows: # מעבר על מערך זה
            state.board[1:i+1] = state.board[0:i] # הורדת כל השורות בלוח עד השורה המלאה ב1
            state.board[0] = 0 # שם שורה ריקה חדשה למעלה
        
        if len(full_rows) > 0: # אם יש לפחות שורה מלאה אחת
            self.update_score(state, len(full_rows)) # עדכון הניקוד
            if not self.train:
                line_clear = pygame.mixer.Sound('sounds/line_clear.mp3') # ניגון צליל
                line_clear.play()
        return len(full_rows)

    def update_score(self, state, num): # עדכון הניקוד
        if num == 1:
            state.score += 40
            self.reward += 0.8
        elif num == 2:
            state.score += 100
            self.reward += 2
        elif num == 3:
            state.score += 300
            self.reward += 6
        elif num == 4:
            state.score += 1200
            self.reward += 20

    def get_reward(self, state):
        if self.reached_top(state):
            self.reward -= 10
        # else:
            # self.reward += 0.5 #################################

        r = self.reward
        self.reward = 0

        return r
    
    def newState(self):
        state = State()
        self.select_falling_piece(state) # בוחר חלק ראשון
        self.add_piece(state)

        
        dqn_state = np.array([
                0, #holes
                ROWS, #landing height
                0, #wells
                0, #bumpiness
                ROWS*COLS, #total height
                0, #full rows
                0 #done=false #################
            ], dtype=np.float32)
        return state, dqn_state
    
    def next_state(self, state, action):
        '''
        a. copy state
        a. take action
        b. if not legal go back
        c. if not landed after action:
            go down
        d. if landed:
            add piece to state
            clear rows
            get new piece
        '''
        next_state = state.copy()
        if action: # אם היא חוקית
            self.move(next_state, action) # מזיז את החלק

        if self.is_collision(next_state, falling_piece=next_state.falling_piece):
            next_state.falling_piece = state.copy_falling_piece()               # delete move
        
        if not self.landed(next_state):       # landed
            next_state.down()
        if self.landed(next_state):     # check again after down
            next_state.add_piece()
            self.clear_rows(next_state)
            self.select_falling_piece(next_state)
        return next_state

    #region next_state
    # def next_state(self, state, action): #לדאוג שהפרס תואם - לא נותן פעמיים פרס על שורה למטה, נותן פרס על הורדת שורה בנקסט סטייט
    #     next_state = state.copy()
    #     row, col, piece = next_state.falling_piece # מציאת מיקום וצורת החלק
    #     rows, cols = piece.shape # מציאת אורך ורוחב החלק
    #     new_piece = False
    #     done = False
    #     landing = self.landing
    #     full_rows = 0
        
    #     if action: # אם היא חוקית
    #         self.move(next_state, action) # מזיז את החלק

    #     if not self.is_collision(next_state, falling_piece=next_state.falling_piece, dRow=1):
    #         self.down_piece(next_state)    # יורד שורה

    #     if self.is_collision(next_state, falling_piece=next_state.falling_piece, dRow=1):
    #         full_rows = self.count_full_rows(next_state)
    #         landing = self.landing_height(next_state)
    #         self.clear_rows(next_state) # מוחק שורות אם צריך  
    #         self.select_falling_piece(next_state) #אתחול המשתנים
    #         self.add_piece(next_state)
    #         new_piece = True
    #         done = self.reached_top(next_state)
    #         if not done:
    #             self.reward += 0.5

    #     next_dqn_state = np.array([
    #         self.count_holes(next_state),
    #         landing,
    #         self.wells(next_state),
    #         self.bumpiness(next_state),
    #         self.total_height(next_state),
    #         full_rows,
    #         done ##########################
    #     ], dtype=np.float32)

    #         # return dqn_state, self.get_reward(next_state), done, new_piece

    #     return next_state, next_dqn_state, self.get_reward(next_state), done, new_piece
    #endregion

    def to_DQN_state(self, state, landing=None, full_rows=None, done=None):
        heights = self.columns_heights(state)
        DQN_state = np.append(heights, heights.max())
        return DQN_state

    def landed (self, state):
        return self.is_collision(state, state.falling_piece, dRow=1)

    # def highest_piece_in_cols(self, state):
    #     highest_board = np.full((1, COLS), 0, dtype=int)

    #     for i in range(COLS):
    #         column = state.board[:, i]
    #         for j in range (0, ROWS):
    #             if column[j] != 0:
    #                 highest_board[0, i] = j
    #                 break
    #     return highest_board
    
    def count_holes(self, state):
        highest_board = self.highest_piece_in_cols(state)
        count = 0

        for i in range(COLS):
            column = state.board[:, i]
            height = highest_board[0, i]
            for j in range (height + 1, ROWS):
                if column[j] == 0:
                    count += 1
        # self.reward -= 0.1*count ########################
        return count
    
    def landing_height(self, state):
        row, col, piece = state.falling_piece # מציאת מיקום וצורת החלק
        self.landing = row
        # if row < 10:
        #     self.reward -= 1 ###########################
        # elif row > 15:
        #     self.reward += 1 ###########################
        return row
    
    def wells(self, state):
        highest_board = self.highest_piece_in_cols(state)
        wells = 0
        for j in range (1, COLS-1):
            if highest_board[0, j-1] < highest_board[0, j] and highest_board[0, j+1] < highest_board[0, j]:
                wells += highest_board[0, j] - min(highest_board[0, j-1], highest_board[0, j+1])

        if highest_board[0, 1] < highest_board[0, 0]:
            wells += highest_board[0, 0] - highest_board[0, 1]

        if highest_board[0, COLS - 2] < highest_board[0, COLS - 1]:
            wells += highest_board[0, COLS -1] - highest_board[0, COLS - 2]

        return wells

    def bumpiness(self, state):
        highest_board = self.highest_piece_in_cols(state)
        bumps = 0
        for i in range (1, COLS):
            bumps += abs(highest_board[0, i] - highest_board[0, i-1])
        return bumps

    def columns_heights (self, state):
        board = state.board
        nonzero_mask = board != 0
        # use argmax on the mask — it returns the first True in each column
        # for columns with no non-zero values, this will return 0, so we must handle that
        first_nonzero_rows = np.where(nonzero_mask.any(axis=0), nonzero_mask.argmax(axis=0), 20)
        heights = 20 - first_nonzero_rows
        return heights

    def count_full_rows(self, state):
        temp_board = state.board.copy() # יצירת לוח זמני
        temp_board[temp_board != 0] = 1 # החלפת כל המספרים שלא 0 ב1
        row_sums = np.sum(temp_board, axis=1) # מציאת הסכום של כל שורה
        full_rows = np.where(row_sums == 10)[0] # יצירת מערך עם מספרי השורות המלאות - סכום 10

        return len(full_rows)
              
    def drop_piece (self, state):
        state = state.copy()
        while not self.landed(state):
            state.down()
        state.add_piece()
        self.select_falling_piece(state)
        return state            
            
    def all_states (self, state):
        states = []
        states_dqn = []
        actions = []
        cleared_rows = []
        _, _, piece = state.falling_piece
       
        piece_num = piece.max()
        if piece_num in (4,):
            rotate_num = 1
        elif piece_num in (1, 5, 7):
            rotate_num = 2
        elif piece_num in (2, 3, 6):
            rotate_num = 4

        for rotate in range(rotate_num):
            state2 = state.copy()
            for i in range(rotate):
                self.move(state2, action=3)
            
            width = state2.falling_piece[2].shape[1]
            for new_col in range(10-width):
                next_state = state2.copy()
                row, col, piece = next_state.falling_piece
                col = new_col
                next_state.falling_piece = row, col, piece

                next_state = self.drop_piece(next_state)
                cleared = self.clear_rows(next_state)

                state_dqn = self.to_DQN_state(next_state)
                states_dqn.append(state_dqn)
                states.append(next_state)
                actions.append((rotate, new_col))
                cleared_rows.append(cleared)

        return states, states_dqn, actions, cleared_rows

