import pygame
from Graphics import *
from Environment import *
from State import *
from HumanAgent import *
from RandomAgent import *
from DQN_Agent import *

pygame.init()
pygame.mixer.init()

clock = pygame.time.Clock()
graphics = Graphics()
state = State()
env = Environment(state=state)
dqn = DQN_Agent(env=env)

# player = HumanAgent(env)
# player = RandomAgent(env)
player = dqn

FPS = 60

graphics.draw_opening_screen()
waiting = True
while waiting: # במצב של סוף המשחק
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    action = player.get_end_Action(events)
    if action == 6: # אם השחקן בוחר להתחיל משחק חדש
        waiting = False


def main():
    env.select_falling_piece(state) # בוחר חלק ראשון
    step = 0
    graphics.draw(state=state) # מצייר את הלוח
    pygame.display.update()
    run = True
    
    if player == dqn:
            state.fall_speed = 1
            state.FALL_SPEED = 1

    # action = player.get_Action(state, events, step) # מקבל את הפעולה שנבחרה

    # if action: # אם היא חוקית
    #     env.move(state, action) # מזיז את החלק

    while run:
        step += 1
        
        events = pygame.event.get()
        for event in events:
            pygame.event.pump()
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        

        if step % state.fall_speed==0: # אם עבר מספיק זמן
            if not env.is_collision(state, falling_piece=state.falling_piece, dRow=1):
                env.down_piece(state)    # יורד שורה
            else:
                state.add_piece()
                env.clear_rows(state) # מוחק שורות אם צריך  
                env.select_falling_piece(state) #אתחול המשתנים
        
        action = player.get_Action(state, events, step) # מקבל את הפעולה שנבחרה

        if action: # אם היא חוקית
            env.move(state, action) # מזיז את החלק


        if env.reached_top(state): # אם חלק "נתקע" למעלה
            game_over = pygame.mixer.Sound('sounds/game_over.mp3') # ניגון צליל
            game_over.play()
            run = False # נגמר המשחק

        if step % 5000 == 0 and state.FALL_SPEED > 5:
            state.FALL_SPEED -= 1
            state.fall_speed = state.FALL_SPEED
        
        graphics.draw(state=state) # ציור הלוח
        pygame.display.update()
        clock.tick(FPS)
    
    waiting = True
    while waiting: # במצב של סוף המשחק
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                waiting = False
        action = player.get_end_Action(events)
        if action == 6: # אם השחקן בוחר להתחיל משחק חדש
            if player != HumanAgent:
                pygame.time.wait(1000)
            state.__init__() # אתחול ה state
            main() # חזרה לתחילת ה main

        graphics.draw_end()
        pygame.display.update()
    
    pygame.quit()

if __name__ == '__main__':
    main()