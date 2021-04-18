import gym
import math
import chess
import gym_chess
import random
from gym_chess.alphazero import BoardEncoding
from chess import Board
import copy
import sys

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
from tensorflow.keras.models import load_model

sys.setrecursionlimit(1500)

def make_chessmaster_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, 3, input_shape=(8,8,119)))
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(4000))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(4673))
    return model
    
old_bobby = make_chessmaster_model()

C = 1
class Node:
    def __init__(self, parent, env, p, last_move=None, absolute_outcome=None, board_encoding=None, turn=1):
        self.parent = parent
        self.env = env
        self.children = []  #copy.copy(children)
        self.isRoot = self.parent is None    
        self.q = 0  # Average value of this node # Actually from the parent's perspective
        self.p = p
        self.n = 0
        self.last_move = last_move
        self.visited = False
        self.board_encoding = board_encoding
        self.turn = turn
        self.outcome = self.convertOutcome(absolute_outcome)
        
    def convertOutcome(self, absolute_outcome):
        if absolute_outcome is not None:
            return self.turn * absolute_outcome
        return None

    # Returns a copy of the board with the move made
    def make_move(self, m):
        newboard = copy.deepcopy(self.env)
        # print(m)
        encoding, reward, done, _ = newboard.step(m)
        return newboard, encoding, reward, done

    def policy_component(self, output):
        return output.numpy()[:-1]

    # def isValidMove(move):
    #     if move in legalMoves:
    #         return True
    #     else:
    #         return False

    # Return propQ, after expanding OR continuing tree search
    def get_propQ(self, depth=0):
        # print(depth)
        # If game ended
        if self.outcome is not None:
            return -self.outcome

        # First time; run network, store policy and value, return value
        if not self.visited:
            # print("First time")
            # visitedList = list(map(lambda e: e.visited, self.children))
            # print(visitedList)
            self.network_output = old_bobby(self.board_encoding.reshape((1, 8, 8, 119)))
            self.visited = True
            # print(self.network_output)
            self.q = self.network_output[0][-1] # Q is from this node's perspective which is incorporated into state
            self.n += 1
            return -self.q
        # Second time; expand
        # print("Second time")
        # visitedList = []
        # for child in self.children:
        #     visitedList.append(child.visited)
        # print(visitedList)
        if not self.children:
            legal_actions = self.env.legal_actions
            # print(self.policy_component(self.network_output[0]).size)
            # print(legal_actions)
            policy = self.policy_component(self.network_output[0])[legal_actions]
            # policy = filter(isValidMove, (self.network_output[0])
            # print(policy)    
            for m, p in zip(legal_actions, policy):
                self.addChild(m, p)
        # Second time and beyond
        # print("Second time and beyond")
        # visitedList = []
        # for child in self.children:
        #     visitedList.append(child.visited)
        # print(visitedList)
        bestChild = None
        bestU = -float("inf")
        for child in self.children:
            u = child.calcU()
            if u > bestU:  # Randomize ties later? Right now just keeps the incumbent bestChild.
                bestChild = child
                bestU = u
        
        propQ = bestChild.get_propQ(depth+1)
        self.q = (self.q*self.n + propQ)/(self.n + 1)
        self.n += 1
        return propQ
        
    def addChild(self, move, p):
        env, encoding, reward, done = self.make_move(move)
        if not done:
            reward = None
        child = Node(self, env, p, last_move=move, absolute_outcome=reward, board_encoding=encoding, turn=-self.turn)
        self.children.append(child)
        return child
    
    def calcU(self):
        return self.q + C * self.p * np.sqrt(self.parent.n - 1)/(1+self.n)

class AlphaChess:
    def __init__(self, env, encoding):
        self.rootNode = Node(None, env, None, board_encoding=encoding)

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
            prob_cumulative += child.n / (self.rootNode.n - 1)
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
    env = gym.make('ChessAlphaZero-v0')
    encoding = env.reset()
    #env.step(env.encode(chess.Move.from_uci('e2e4')))
    #env.step(env.encode(chess.Move.from_uci('e7e5')))
    #env.step(env.encode(chess.Move.from_uci('d1h5')))
    #env.step(env.encode(chess.Move.from_uci('b8c6')))
    #env.step(env.encode(chess.Move.from_uci('f1c4')))
    #env.step(env.encode(chess.Move.from_uci('g8f6')))
    # env.step(env.encode(chess.Move.from_uci('h5f7')))

    # env.step(env.encode(chess.Move.from_uci('f2f3')))
    # env.step(env.encode(chess.Move.from_uci('e7e6')))
    # env.step(env.encode(chess.Move.from_uci('g2g4')))
    # env.step(env.encode(chess.Move.from_uci('d8h4')))
    # From https://en.wikipedia.org/wiki/Scholar%27s_mate
    # 1. e4 e5
    # 2. Qh5 Nc6
    # 3. Bc4 Nf6??
    # 4. Qxf7#

    # observation, reward, done, info = env.step(env.encode(chess.Move.from_uci('g2g4')))
    # print(observation)
    # print(reward)
    # print(done)
    # print(info)

    # newBoard = chess.Board()
    # print(newBoard)
    # print(type(newBoard))
    # # print(newBoard.result())
    # print(env.compute_reward)
    player = AlphaChess(env, encoding)
    #p2 = AlphaChess()
    print(env.decode(player.get_action_probabilistic(1000)))
        
let_the_games_begin()
# while not done:
#     action = random.sample(env.legal_actions, 1)
#     print(BoardEncoding(env))
#     # board.push(action[0])
#     env.step(action[0])
#     print(env.render(mode="unicode"))

# env.close()
