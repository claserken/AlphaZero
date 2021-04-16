import gym
import math
import chess
import gym_chess
import random
from gym_chess.alphazero import BoardEncoding
from chess import Board

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
from tensorflow.keras.models import load_model


def make_chessmaster_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, 3))
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(4000))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(4673))
    
old_bobby = make_chessmaster_model()

print(gym_chess)

env = gym.make('ChessAlphaZero-v0')

C = 1
class Node:
    def __init__(self, parent, board, p, last_move=None, children=[]):
        self.parent = parent
        self.board = board
        self.children = children
        self.isRoot = self.parent is None    
        self.q = 0  # Average value of this node # Actually from the parent's perspective
        self.p = p
        self.n = 0
        self.last_move = last_move
        self.visited = False
        self.outcome = self.getOutcome()
        
    def getOutcome(self):
        print(self.board)
        if self.board.reward() == "*":
            return None
        if self.board.result() == "1/2-1/2":
            return 0
        if (self.board.result() == "0-1" and self.board.turn == chess.BLACK) or (self.board.result() == "1-0" and self.board.turn == chess.WHITE):
            return 1
        return -1


    # Returns a copy of the board with the move made
    def make_move(self, m):
        newboard = self.board.copy()
        newboard.step(newboard.decode(m))
        return newboard

    def policy_component(output):
        return output.numpy()[:-1]

    # Return propQ, after expanding OR continuing tree search
    def get_propQ(self):
        # If game ended
        if self.outcome is not None:
            return -self.outcome

        # First time; run network, store policy and value, return value
        if not self.visited:
            self.network_output = old_bobby(self.state)
            self.visited = True
            self.q = self.network_output[-1] # Q is from this node's perspective which is incorporated into state
            self.n += 1
            return -self.q
        # Second time; expand
        if not self.children:
            legal_actions = env.legal_actions
            policy = self.policy_component(self.network_output)[legal_actions]           
            for m, p in zip(legal_actions, policy):
                self.addChild(m, p)
        # Second time and beyond
        bestChild = None
        bestU = -float("inf")
        for child in self.children:
            u = child.calcU()
            if u > bestU:  # Randomize ties later? Right now just keeps the incumbent bestChild.
                bestChild = child
                bestU = u
        
        propQ = bestChild.get_propQ()
        self.q = (self.q*self.n + propQ)/(self.n + 1)
        self.n += 1
        return propQ
        
    def addChild(self, move, p):
        child = Node(self, self.make_move(move), p, last_move=move)
        self.children.append(child)
        return child
    
    def calcU(self):
        return self.q + C * self.p * np.sqrt(self.parent.n - 1)/(1+self.n)

class AlphaChess:
    def __init__(self, board):
        self.rootNode = Node(None, board, None)

    def choose(self, choice):
        self.rootNode = choice
        self.rootNode.parent = None
        
    def get_action_probabilistic(self, iterations):
        for _ in range(iterations):
            self.rootNode.get_propQ()

        rand = random.uniform(0,1)
        prob_cumulative = 0
        the_chosen_one = None
        for child in self.rootNode.children:
            prob_cumulative += child.n / (self.n - 1)
            if prob_cumulative > rand:
                the_chosen_one = child
                break
            
        self.choose(the_chosen_one)
        return self.rootNode.last_move
        
    def external_move(self, move):
        found_child = False
        for child in self.rootNode.children:
            if child.last_move == move:
                self.choose(child)
                found_child = True
                break
        if not found_child:
            child = self.rootNode.add_child(move)
            self.choose(child)

def let_the_games_begin():
    env.reset()
    newBoard = chess.Board()
    print(newBoard)
    print(type(newBoard))
    # print(newBoard.result())
    print(env.compute_reward)
    player = AlphaChess(newBoard)
    #p2 = AlphaChess()
    print(player.get_action_probabilistic(1000))
        
let_the_games_begin()
# while not done:
#     action = random.sample(env.legal_actions, 1)
#     print(BoardEncoding(env))
#     # board.push(action[0])
#     env.step(action[0])
#     print(env.render(mode="unicode"))

# env.close()
