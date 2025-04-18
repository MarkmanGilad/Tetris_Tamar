import pygame
from DQN import DQN
from DQN_Agent import DQN_Agent
from Environment import *
from Graphics import *
from ReplayBuffer import ReplayBuffer
from State import *
import torch 
import wandb
import os

pygame.init()
pygame.mixer.init()




def main (chkpt):
    num = chkpt

    epochs = 300000
    C = 30
    batch = 64
    learning_rate = 0.0001
    path = f"Data\DQN_PARAM_{num}.pth"
    replay_path = f"Data\Replay_{num}.pth"
    best_path = f"Data\DQN_PARAM_BEST{num}.pth"

    wandb.init(
        project = "Tetris",
        id = f"Tetris{num}",

        config={
            "name": f"Tetris{num}",
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch": batch,
            "C": C,
        }
    )
    state = State()
    env = Environment(state=state)
    graphics = Graphics()
    player = DQN_Agent(env=env, train=True)
    player_hat = DQN_Agent(env=env,train=True)
    Q = player.Q
    player_hat.DQN = player.DQN.copy()
    Q_hat = player_hat.Q
    
    replay = ReplayBuffer()
    optim = torch.optim.Adam(player.DQN.parameters(), lr=learning_rate)

    loss = 0
    avg_score = 0
    best_score = 0
    for epoch in range(epochs):
        state = env.state
        done = False
        moves = 0    
        score = 0
        new_pieces = 1
        while not done:
            graphics.draw(state=state) # מצייר את הלוח 
            pygame.display.update()
            pygame.event.pump()
            next_state, next_states_dqn, action, cleared_rows = player.get_Action(state, epoch=epoch, train=True)
            reward = cleared_rows*10 + 1
            moves += len(action)
            print(f'epoch: {epoch}   moves: {moves}', end="\r")
            new_pieces += 1
            done = env.is_done(next_state)
            score += next_state.score
            if done:
                reward -= 10
            replay.push(state, action, reward, next_state, next_states_dqn, done)
            state = next_state  # env.move

            if len(replay) < 100:
                continue
            
            _, _, rewards, next_states, next_states_dqn_tensor, dones = replay.sample(batch)
            Q_values = player.Q(next_states_dqn_tensor)   # Q(s,a) = Q(next_state)

            next_next_states_dqn, _ = player.get_next_states_Values(next_states) # DDQN
            
            with torch.no_grad():
                Q_hat_Values = player_hat.Q(next_next_states_dqn)
            
            loss = player.DQN.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim.step()
            optim.zero_grad()


        if epoch % C == 0:
            player_hat.DQN.load_state_dict(player.DQN.state_dict())

        if epoch!=0 and epoch%1 == 0:
            # player.save_param(path)
            # wandb.log({"score": score, "loss": loss, "new pieces": new_pieces})
            # wandb.log({"loss": loss})
            # wandb.log({"new pieces": new_pieces})
            avg_score += score
            
            print (f'epoch: {epoch} moves: {moves} loss: {loss:.7f}  score: {score}  new pieces: {new_pieces}')

        if epoch%100 == 0:
            avg_score = avg_score/100
            if avg_score > best_score:
                best_score = avg_score
                # player.save_param(best_path)
            avg_score = 0


    player.save_param(path)
    torch.save(replay, replay_path)

if __name__ == '__main__':
    if not os.path.exists("Data/checkpoit_num"):
        torch.save(101, "Data/checkpoit_num")    
    
    chkpt = torch.load("Data/checkpoit_num", weights_only=False)
    chkpt += 1
    torch.save(chkpt, "Data/checkpoit_num")    
    main (chkpt)