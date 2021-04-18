import gym
import chess
from gym.spaces.box import Box
import gym_chess
from gym_chess.alphazero import BoardEncoding
import numpy as np 
import random 

env = gym.make('ChessAlphaZero-v0')
env.reset()
print(env.decode(0))


# while not done:
#     action = random.sample(env.legal_actions, 1)
#     # board.push(action[0])
#     state = env.step(action[0])[0]
#     print(state.shape)
#     print(type(env))
#     print(env.render(mode="unicode"))