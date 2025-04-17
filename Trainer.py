import pygame
from DQN import DQN
from DQN_Agent import DQN_Agent
from Environment import *
from Graphics import *
from ReplayBuffer import ReplayBuffer
from State import *
import torch 
import wandb

pygame.init()
pygame.mixer.init()

num = 41

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


def main ():
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
        state, dqn_state = env.newState()
        done = False
        moves = 0    
        score = 0
        new_pieces = 1
        while not done:
            graphics.draw(state=state) # מצייר את הלוח 
            pygame.display.update()
            pygame.event.pump()
            action = player.get_Action(torch.from_numpy(dqn_state), epoch=epoch, train=True)
            moves += 1
            print(f'epoch: {epoch}   moves: {moves}', end="\r")
            next_state, next_dqn_state, reward, done, new_piece = env.next_state(state, action)
            if new_piece:
                new_pieces += 1
            # done = env.reached_top(next_state)
            score += next_state.score

            replay.push(dqn_state, action, reward, next_dqn_state, done)
            state = next_state  # env.move
            dqn_state = next_dqn_state
            # state.score += score

            if len(replay) < 1000:
                continue
            dqn_states, actions, rewards, next_dqn_states, dones = replay.sample(batch)
            Q_values = Q(dqn_states, actions)
            next_actions, Q_hat_Values = player_hat.get_Actions_Values(next_dqn_states)
            with torch.no_grad():
                Q_hat_Values = Q_hat(next_dqn_states, next_actions)
            
            loss = player.DQN.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim.step()
            optim.zero_grad()


        if epoch % C == 0:
            player_hat.DQN.load_state_dict(player.DQN.state_dict())

        if epoch!=0 and epoch%1 == 0:
            player.save_param(path)
            wandb.log({"score": score, "loss": loss, "new pieces": new_pieces})
            # wandb.log({"loss": loss})
            # wandb.log({"new pieces": new_pieces})
            avg_score += score
            
            print (f'epoch: {epoch} moves: {moves} loss: {loss:.7f}  score: {score}  new pieces: {new_pieces}')

        if epoch%100 == 0:
            avg_score = avg_score/100
            if avg_score > best_score:
                best_score = avg_score
                player.save_param(best_path)
            avg_score = 0


    player.save_param(path)
    torch.save(replay, replay_path)

if __name__ == '__main__':
    main()