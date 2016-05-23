__author__ = 'Kevin Mike'

from math import *
import time
import random

THINK_DURATION = 1


class Node:

    def __init__(self, move=None, parent=None, state=None):

        self.move = move
        self.parent_node = parent  # "None" for the root node
        self.children = []
        self.visits = 0
        self.untried_moves = state.get_moves()  # future child nodes
        self.who = state.get_whos_turn()
        self.score = 0.0

    def select_child(self):
        s = sorted(self.children, key=lambda c: c.score + sqrt(2 * log(self.visits) / c.visits))[-1]
        return s

    def add_child(self, m, s):
        n = Node(move=m, parent=self, state=s)
        self.untried_moves.remove(m)
        self.children.append(n)
        return n

    def update(self, result):
        self.visits += 1
        self.score += result


def think(state, quip):

    root_node = Node(state=state)

    start_time = time.time()
    end_time = start_time + THINK_DURATION

    iterations = 0

    def calculate_score(score, curr_player):
        if curr_player == 'red':
            return score['red'] - score['blue']
        else:
            return score['blue'] - score['red']

    while True:

        iterations += 1
        node = root_node
        sim_state = state.copy()

        # select
        while not node.untried_moves and node.children:
            node = node.select_child()
            sim_state.apply_move(node.move)

        # expand
        if node.untried_moves:
            rand_move = random.choice(node.untried_moves)
            sim_state.apply_move(rand_move)
            node = node.add_child(rand_move, sim_state)

        # rollout
        while sim_state.get_moves():
            sim_state.apply_move(random.choice(sim_state.get_moves()))

        # backpropagate
        while node is not None:
            if node.parent_node:
                node.update(calculate_score(sim_state.get_score(), node.parent_node.who))
            else:
                node.update(0)

            node = node.parent_node

        curr_time = time.time()
        if curr_time > end_time:
            break

    sample_rate = float(iterations)/(curr_time - start_time)
    quip(sample_rate)

    return sorted(root_node.children, key=lambda c: c.visits)[-1].move
