import pygame

FPS = 60

normal_size = 1
smaller_size = 0.8
bigger_size = 1.4

normal_font = 30
smaller_font = 25
bigger_font = 38

normal = normal_size, normal_font
bigger = bigger_size, bigger_font

FACTOR, FONT_SIZE = normal

SQUARE_SIZE = round(25 * FACTOR)
LINE_WIDTH = round(1 * FACTOR)
ROWS, COLS = 20, 10
HEADER_SIZE = 100 * FACTOR
WIDTH, HEIGHT = COLS * SQUARE_SIZE , ROWS * SQUARE_SIZE + HEADER_SIZE 

H_WIDTH, H_HEIGHT = round(250 * FACTOR), round(100 * FACTOR)
M_WIDTH, M_HEIGHT = round(250 * FACTOR), round(520 * FACTOR)
I_WIDTH, I_HEIGHT = round(60 * FACTOR), round(40 * FACTOR)
E_WIDTH, E_HEIGHT = round(250 * FACTOR), round(200 * FACTOR)

RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
LIGHTGRAY = (160,160,160)
GREEN = (57, 255, 20)
CADETBLUE1 = (152,245,255)
TURQUOISE = (64, 224, 208)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
PURPLE = (160, 32, 240)

class Graphics:
    def __init__(self) -> None:
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.header_surf = pygame.Surface((H_WIDTH, H_HEIGHT))
        self.main_surf = pygame.Surface((M_WIDTH, M_HEIGHT))
        self.img_surf = pygame.Surface((I_WIDTH, I_HEIGHT))
        self.end_surf = pygame.Surface((E_WIDTH, E_HEIGHT))
        pygame.display.set_caption('TETRIS')
    
    def draw(self, state):
        self.header_surf.fill(BLACK)
        self.main_surf.fill(LIGHTGRAY)
        self.draw_score(state=state)
        self.draw_next_piece(state=state)
        self.draw_pieces(state=state)
        self.draw_Lines()

        self.screen.blit(self.header_surf, (0,0))
        self.screen.blit(self.main_surf, (0, HEADER_SIZE))
        self.screen.blit(self.img_surf, (WIDTH - I_WIDTH - 10, 40))

    def draw_score(self, state):
        font = pygame.font.Font('fonts/score_font.ttf', FONT_SIZE)
        score_text = font.render(f'Score: {state.score}', True, WHITE)
        self.header_surf.blit(score_text, (20, 10 * FACTOR))
        pygame.display.update()

    def draw_next_piece(self, state):
        font = pygame.font.Font('fonts/score_font.ttf', FONT_SIZE)
        score_text = font.render(f'Next piece:', True, WHITE)
        self.header_surf.blit(score_text, (20, 50 * FACTOR))
        id = state.next_piece
        if id == 1:
            img = pygame.image.load('img/cyan_piece.png')
        elif id == 2:
            img = pygame.image.load('img/blue_piece.png')
        elif id == 3:
            img = pygame.image.load('img/orange_piece.png')
        elif id == 4:
            img = pygame.image.load('img/yellow_piece.png')
        elif id == 5:
            img = pygame.image.load('img/green_piece.png')
        elif id == 6:
            img = pygame.image.load('img/purple_piece.png')
        elif id == 7:
            img = pygame.image.load('img/red_piece.png')
        
        if id == 4:
            img = pygame.transform.scale(img, (I_WIDTH - 0.5*I_HEIGHT, I_HEIGHT))
        elif id == 1:
            img = pygame.transform.scale(img, (I_WIDTH, I_WIDTH*0.25))
        else:
            img = pygame.transform.scale(img, (I_WIDTH, I_HEIGHT))
        self.img_surf.fill(BLACK)
        if id == 1:
            self.img_surf.blit(img, (0, 15))
        else:
            self.img_surf.blit(img, (0, 0))

        pygame.display.update()
    
    def draw_pieces (self, state):
        board = state.board
        for row in range(ROWS): #עבור כל משבצת
            for col in range(COLS):
               self.draw_piece((row,col), board[row, col]) #צביעת המשבצת בצבע המתאים

    def draw_piece(self, row_col, color_num):   
        row, col = row_col
        x = col * SQUARE_SIZE  
        y = row * SQUARE_SIZE

        if color_num == 0:
            pygame.draw.rect(self.main_surf, LIGHTGRAY, (x,y,SQUARE_SIZE,SQUARE_SIZE))
        elif color_num == 1:
            pygame.draw.rect(self.main_surf, TURQUOISE, (x,y,SQUARE_SIZE,SQUARE_SIZE))
        elif color_num == 2:
            pygame.draw.rect(self.main_surf, BLUE, (x,y,SQUARE_SIZE,SQUARE_SIZE))
        elif color_num == 3:
            pygame.draw.rect(self.main_surf, ORANGE, (x,y,SQUARE_SIZE,SQUARE_SIZE))
        elif color_num == 4:
            pygame.draw.rect(self.main_surf, YELLOW, (x,y,SQUARE_SIZE,SQUARE_SIZE))
        elif color_num == 5:
            pygame.draw.rect(self.main_surf, GREEN, (x,y,SQUARE_SIZE,SQUARE_SIZE))
        elif color_num == 6:
            pygame.draw.rect(self.main_surf, PURPLE, (x,y,SQUARE_SIZE,SQUARE_SIZE))
        elif color_num == 7:
            pygame.draw.rect(self.main_surf, RED, (x,y,SQUARE_SIZE,SQUARE_SIZE))



    def draw_Lines(self):
        for i in range(ROWS+1):
            pygame.draw.line(self.main_surf, BLACK, (0, i * SQUARE_SIZE), 
                         (M_WIDTH, i * SQUARE_SIZE), width=LINE_WIDTH)
        
        for i in range(COLS+1):
            pygame.draw.line(self.main_surf, BLACK, (i * SQUARE_SIZE, 0), 
                         (i * SQUARE_SIZE, M_HEIGHT), width=LINE_WIDTH)
            

    def draw_end(self):
        self.draw_endgame()
        self.draw_restart()


    def draw_endgame(self):
        font = pygame.font.Font('fonts/score_font.ttf', 60)
        score_text = font.render(f'Game over', True, BLACK)
        text_rect = score_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        self.screen.blit(score_text, text_rect.topleft)

    def draw_restart(self):
        font = pygame.font.Font('fonts/score_font.ttf', 20)
        score_text = font.render(f'press [space] to restart', True, BLACK)
        text_rect = score_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
        self.screen.blit(score_text, text_rect.topleft)

    def draw_opening_screen(self):
        img = pygame.image.load('img/tetris.png')
        img = pygame.transform.scale(img, (WIDTH, 80))
        self.screen.blit(img, ((WIDTH - img.get_width()) // 2, (HEIGHT - img.get_height()) // 2))
        pygame.display.update()


