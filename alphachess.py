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
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.mixed_precision.experimental import policy

def sigmoid(x):
  return 1 / (1 + tf.exp(-x))

class AlphaLayer(tf.keras.layers.Layer):
    def __init__(self, units=4673, input_dim=200):
        super(AlphaLayer, self).__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        # Result is (None, 4673) so individual examples are the rows

        result = tf.matmul(inputs, self.w) + self.b

        probs = result[:, :-1]
        val = tf.reshape(result[:, -1], shape=[-1,1])

        probs = tf.exp(probs)
        probs /= tf.math.reduce_sum(probs, axis=1)[:, None] 

        val = 2*sigmoid(val) - 1
    
        return tf.concat([probs, val], 1)

# class InsertMoreLayer(tf.keras.layers.Layer, array):
#     def __init__(self, units=4673, input_dim=200):
#         super(tf.keras.layers.Linear, self).__init__()
#         self.w = self.add_weight(
#             shape=(input_dim, units), initializer="random_normal", trainable=True
#         )
#         for elem in array:
#             self.w = self.add_weight(shape=(input_dim, units), initializer="random_normal", trainable=True)
#         self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

#     def call(self, inputs):
#         return tf.matmul(inputs, self.w) + self.b

def make_chessmaster_model():
    boardinput = layers.Input(shape=(8,8,6))
    metainput = layers.Input(shape=(8,))
    
    conv1 = layers.Conv2D(64, 3)(boardinput)
    flatten = layers.Flatten()(conv1)
    batchNorm = layers.BatchNormalization()(flatten)
    added = layers.Concatenate()([batchNorm, metainput])
 
    dense = layers.Dense(200)(added)
    dropout = layers.Dropout(0.3)(dense)
    
    output = AlphaLayer(4673)(dropout)
    model = tf.keras.models.Model(inputs=[boardinput, metainput], outputs=output)

    
    # model.add(layers.Conv2D(64, 3, input_shape=(8,8,6)))
    # model.add(layers.Flatten())
    # model.add(layers.BatchNormalization())
    # model.add(layers.Dense(200))
    # model.add(layers.Dropout(0.3))
    # model.add(AlphaLayer(4673))
    return model
'''
input1 = tf.keras.layers.Input(shape=(16,))
x1 = tf.keras.layers.Dense(8, activation='relu')(input1)
input2 = tf.keras.layers.Input(shape=(32,))
x2 = tf.keras.layers.Dense(8, activation='relu')(input2)
# equivalent to `added = tf.keras.layers.add([x1, x2])`
added = tf.keras.layers.Add()([x1, x2])
out = tf.keras.layers.Dense(4)(added)
model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)
'''

'''
def encode(board):
    
    
def decode(encoded):

def encode_move(move):   
def decode_move(n):
    #Queen
    #vertical
    if n < 8:
        #go to row n    
    n -= 8
    #horizontal
    if n < 8:

    n -= 8
    # northeast diagonal
    if n < 8:

    n -= 8
    # northwest diagonal
    if n < 8:
    
    n -= 8
    #Left rook
    # vertial
    # horzonal

    #Right rook
    # vertical
    # horizontal

    #Light squared bishop

    #Dark squared bishop
    # northeast
    # northwest

    #Left knight
    # 1-8, moving clockwise from the top

    #Right knight
    # 1-8, moving clockwise from the top

    #Pawn moves
    if n < 32:
        row = 6 #2nd rank

    n -= 32
    if n < 24*5:
        row = 5 - n//24

    n -= 24*5
    if n < 32:
        row = 0
'''
old_bobby = make_chessmaster_model()
print(old_bobby.summary())

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
        self.outcome = absolute_outcome
        
    # def convertOutcome(self, absolute_outcome):
    #     if absolute_outcome is not None:
    #         return self.turn * absolute_outcome
    #     return None

    # Returns a copy of the board with the move made
    def make_move(self, m):
        # print("make_move() called")
        newboard = copy.deepcopy(self.env)
        # print(m)
        encoding, reward, done, _ = newboard.step(m)
        # print('reward: {}'.format(reward))
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
        self.n += 1
        # print(depth)
        # If game ended
        # print(self.env.render(mode="unicode"))
        if self.outcome is not None:
            print("depth: {}".format(depth))
            return abs(self.outcome)

        # print(depth)
        # First time; run network, store policy and value, return value
        if not self.visited:
            # print("First time")
            # visitedList = list(map(lambda e: e.visited, self.children))
            # print(visitedList)
            # print(self.board_encoding)
            self.network_output = old_bobby(inputs=[self.board_encoding[0].reshape((1, 8, 8, 6)), self.board_encoding[1].reshape((1, 8))])
            self.visited = True
            # print(self.network_output)
            # self.q = self.network_output[0][-1] # Q is from this node's perspective which is incorporated into state
            self.q = 0
           # try:
            #    print(self.env.decode(self.last_move))
            #except:
             #   print("error")
            #print(depth)
            return -self.q
        # Second time; expand
        # print("Second time")
        # visitedList = []
        # for child in self.children:
        #     visitedList.append(child.visited)
        # print(visitedList)
        if not self.children:
            # print("if there are no children")
            legal_actions = self.env.legal_actions
            # print(self.policy_component(self.network_output[0]).size)
            # print(legal_actions)

            policy = self.policy_component(self.network_output[0])[legal_actions]
            # policy = np.array([1.0]*len(legal_actions))/len(legal_actions)

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
        #print((len(self.children), self.outcome))
        for child in self.children:
            u = child.calcU()
            # print("Last move: ", child.last_move)
            # print("U-Val: ", u)
            # if child.last_move == self.env.encode(chess.Move.from_uci("a1f1")):
            #     print("Hit a1f1")
            #     print(child.turn)
            #     print(child.outcome)
            #     exit()
            if u > bestU:  # Randomize ties later? Right now just keeps the incumbent bestChild.
                bestChild = child
                bestU = u
        
        propQ = bestChild.get_propQ(depth+1)
        self.q = (self.q*self.n + propQ)/(self.n + 1)
        return -propQ
        
    def addChild(self, move, p):
        env, encoding, reward, done = self.make_move(move)
        if not done:
            reward = None
        child = Node(self, env, p, last_move=move, absolute_outcome=reward, board_encoding=encoding, turn=-self.turn)
        self.children.append(child)
        return child
    
    def calcU(self):
        if self.outcome is not None:
            return abs(self.outcome)
        return -self.q + C * self.p * np.sqrt(self.parent.n - 1)/(1+self.n)

class AlphaChess:
    def __init__(self, env, encoding):
        self.rootNode = Node(None, env, None, board_encoding=encoding)

    def choose(self, choice):
        self.rootNode = choice
        self.rootNode.parent = None
        
    def get_action_probabilistic(self, iterations):
        for itCount in range(iterations):
            if itCount % 20 == 0:
                print(itCount)
            self.rootNode.get_propQ()
        
        print("Q-Value: ", self.rootNode.q)

        rand = random.uniform(0,1)
        prob_cumulative = 0
        the_chosen_one = None
        for child in self.rootNode.children:
            prob = child.n / (self.rootNode.n - 1)
            # prob_cumulative += prob
            print((self.rootNode.env.decode(child.last_move), prob))
            if prob > prob_cumulative:
                prob_cumulative = prob
                the_chosen_one = child
                # break # commented out for debugging
            
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
    encoding = env.reset(newBoard=chess.Board(fen="8/8/8/8/8/4k3/3p4/4K3 w - - 0 2"))
    print(encoding[1].shape)
    print(env.render(mode="unicode"))
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
    print(env.decode(player.get_action_probabilistic(5000)))
        
let_the_games_begin()
# while not done:
#     action = random.sample(env.legal_actions, 1)
#     print(BoardEncoding(env))
#     # board.push(action[0])
#     env.step(action[0])
#     print(env.render(mode="unicode"))

# env.close()
