import gym
import chess
import gym_chess
import random
import copy

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.models import load_model

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
    
    return model

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

    # Returns a copy of the board with the move made
    def make_move(self, m):
        newboard = copy.deepcopy(self.env)
        encoding, reward, done, _ = newboard.step(m)
        return newboard, encoding, reward, done

    def policy_component(self, output):
        return output.numpy()[:-1]

    # Return propQ, after expanding OR continuing tree search
    def get_propQ(self, depth=0):
        self.n += 1
        # If game ended
        if self.outcome is not None:
            print("depth: {}".format(depth))
            return abs(self.outcome)

        # First time; run network, store policy and value, return value
        if not self.visited:
            self.network_output = old_bobby(inputs=[self.board_encoding[0].reshape((1, 8, 8, 6)), self.board_encoding[1].reshape((1, 8))])
            self.visited = True
            self.q = 0
            return -self.q

        # Second time; expand
        if not self.children:
            legal_actions = self.env.legal_actions
            policy = self.policy_component(self.network_output[0])[legal_actions]
   
            for m, p in zip(legal_actions, policy):
                self.addChild(m, p)
        # Second time and beyond
        bestChild = None
        bestU = -float("inf")
        #print((len(self.children), self.outcome))
        for child in self.children:
            u = child.calcU()
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
        self.rootNode = Node(None, copy.deepcopy(env), None, board_encoding=encoding)

    def choose(self, choice):
        self.rootNode = choice
        self.rootNode.parent = None
        
    def get_action_probabilistic(self, iterations, trainingExamples=None):
        for itCount in range(iterations):
            if itCount % 20 == 0:
                print(itCount)
            self.rootNode.get_propQ()
        
        print("Q-Value: ", self.rootNode.q)

        rand = random.uniform(0,1)
        prob_cumulative = 0
        the_chosen_one = None
        probs = []
        for child in self.rootNode.children:
            prob = child.n / (self.rootNode.n - 1)
            prob_cumulative += prob
            probs.append(prob)
            print((self.rootNode.env.decode(child.last_move), prob))
            if prob_cumulative > rand and the_chosen_one is None:
                the_chosen_one = child
        
        #Add to traning examples
        if trainingExamples is not None:
            # Loss needs [network predicted move probs, tree search probs]
            # Option 1: Remember predicted move probs
            # Option 2: Pass in the board encoding and rerun at training
            example = [self.rootNode.board_encoding, probs]
            trainingExamples.append(example)

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

def addResult(examples, reward):
    examples.append(reward) #Always from white's perspective

def let_the_games_begin(numGames):
    
    for i in range(numGames):
        env = gym.make('ChessAlphaZero-v0')
        # encoding = env.reset(newBoard=chess.Board(fen="8/8/8/8/8/4k3/3p4/4K3 w - - 0 2"))
        encoding = env.reset(newBoard=chess.Board())

        alpha = AlphaChess(env, encoding)
        gameOver = False
        trainingExamples = []
        while not gameOver:
            _, reward, done, _ = env.step(alpha.get_action_probabilistic(5000, trainingExamples))
            if done:
                gameOver = True
                addResult(trainingExamples, reward)
            print(env.render(mode="unicode"))
        
let_the_games_begin()