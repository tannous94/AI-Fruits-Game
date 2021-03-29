"""Search Algos: MiniMax, AlphaBeta
"""
from utils import ALPHA_VALUE_INIT, BETA_VALUE_INIT
#TODO: you can import more modules, if needed
import numpy as np
import copy

class SearchAlgos:
    def __init__(self, utility, succ, perform_move, goal=None, heuristic=None):
        """The constructor for all the search algos.
        You can code these functions as you like to, 
        and use them in MiniMax and AlphaBeta algos as learned in class
        :param utility: The utility function.
        :param succ: The succesor function.
        :param perform_move: The perform move function.
        :param goal: function that check if you are in a goal state.
        """
        self.utility = utility
        self.succ = succ
        self.perform_move = perform_move
        self.goal = goal
        self.heuristic = heuristic

    def search(self, state, depth, maximizing_player):
        pass


class MiniMax(SearchAlgos):

    def search(self, state, depth, maximizing_player):
        """Start the MiniMax algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """

        if self.goal(state, maximizing_player):
            #TODO: implement utility
            return self.utility(state), None, 1
        if depth == 0:
            #TODO: Add heuristic
            return self.heuristic(state), None, 0

        children = self.succ(state, maximizing_player)
        curr_num_of_leaves = len(children)
        best_move = None

        if maximizing_player:
            curr_max = -np.inf
            for child in children:
                new_state = copy.deepcopy(child)
                v, _, res_num_of_leaves = self.search(new_state, depth-1, False)
                if curr_max < v:
                    curr_max = v
                    old_pos = self.get_pos(state)
                    new_pos = self.get_pos(new_state)
                    i = new_pos[0] - old_pos[0]
                    j = new_pos[1] - old_pos[1]
                    best_move = i, j
                    curr_num_of_leaves = curr_num_of_leaves + res_num_of_leaves
                #curr_max = max(curr_max, v)
            return curr_max, best_move, curr_num_of_leaves
        else:
            curr_min = np.inf
            for child in children:
                new_state = copy.deepcopy(child)
                v, _, res_num_of_leaves = self.search(new_state, depth-1, True)
                if curr_min > v:
                    curr_min = v
                    curr_num_of_leaves = curr_num_of_leaves + res_num_of_leaves
                #curr_min = min(curr_min, v)
            return curr_min, None, curr_num_of_leaves

    def get_pos(self, state):
        pos = np.where(state.board == 1)
        return tuple(ax[0] for ax in pos)

class AlphaBeta(SearchAlgos):

    def search(self, state, depth, maximizing_player, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT):
        """Start the AlphaBeta algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param: beta: beta value
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        #TODO: erase the following line and implement this function.
        if self.goal(state, maximizing_player):
            # TODO: implement utility
            return self.utility(state), None, 1
        if depth == 0:
            # TODO: Add heuristic
            return self.heuristic(state), None, 0

        children = self.succ(state, maximizing_player)
        curr_num_of_leaves = len(children)
        best_move = None

        if maximizing_player:
            curr_max = -np.inf
            for child in children:
                new_state = copy.deepcopy(child)
                v, _, res_num_of_leaves = self.search(new_state, depth - 1, False, alpha, beta)
                if curr_max < v:
                    curr_max = v
                    old_pos = self.get_pos(state)
                    new_pos = self.get_pos(new_state)
                    i = new_pos[0] - old_pos[0]
                    j = new_pos[1] - old_pos[1]
                    best_move = i, j
                    curr_num_of_leaves = curr_num_of_leaves + res_num_of_leaves
                # curr_max = max(curr_max, v)
                alpha = max(curr_max, alpha)
                if curr_max >= beta:
                    break
            return curr_max, best_move, curr_num_of_leaves
        else:
            curr_min = np.inf
            for child in children:
                new_state = copy.deepcopy(child)
                v, _, res_num_of_leaves = self.search(new_state, depth - 1, True, alpha, beta)
                if curr_min > v:
                    curr_min = v
                    curr_num_of_leaves = curr_num_of_leaves + res_num_of_leaves
                # curr_min = min(curr_min, v)
                beta = min(curr_min, beta)
                if curr_min <= alpha:
                    break
            return curr_min, None, curr_num_of_leaves

    def get_pos(self, state):
        pos = np.where(state.board == 1)
        return tuple(ax[0] for ax in pos)
