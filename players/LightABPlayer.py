"""
MiniMax Player with AlphaBeta pruning with light heuristic
"""
from players.AbstractPlayer import AbstractPlayer
import numpy as np
import time
import copy
import SearchAlgos

#TODO: you can import more modules, if needed


class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score):
        AbstractPlayer.__init__(self, game_time, penalty_score) # keep the inheritance of the parent's (AbstractPlayer) __init__()
        self.board = None
        self.pos = None
        self.alphabeta = SearchAlgos.AlphaBeta(utility=self.utility, succ=self.succ, perform_move=None, goal=self.goal,
                                               heuristic=self.heuristic)
        self.players_score = None
        #TODO: initialize more fields, if needed, and the AlphaBeta algorithm from SearchAlgos.py


    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array, a 2D matrix of the board.
        No output is expected.
        """
        #TODO: erase the following line and implement this function.
        self.board = board
        pos = np.where(board == 1)
        # convert pos to tuple of ints
        self.pos = tuple(ax[0] for ax in pos)

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement, chosen from self.directions
        """
        #TODO: erase the following line and implement this function.
        self.players_score = players_score
        dep = 4

        board_copy = copy.deepcopy(self.board)
        scores = copy.deepcopy(self.players_score)
        fruits_on_board = self.get_fruits_on_board(self.board)
        curr_state = GameState(board_copy, fruits_on_board, scores)

        score, best_move, num_of_leaves = self.alphabeta.search(curr_state, dep, True)

        if best_move is None:
            exit(0)

        prev_pos = self.get_pos(curr_state, True)
        self.board[prev_pos] = -1
        i = prev_pos[0] + best_move[0]
        j = prev_pos[1] + best_move[1]
        best_new_pos = i, j
        self.board[best_new_pos] = 1

        return best_move


    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        #TODO: erase the following line and implement this function.
        rival_pos = np.where(self.board == 2)
        prev_pos = tuple(ax[0] for ax in rival_pos)
        self.board[prev_pos] = -1
        self.board[pos] = 2


    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        #TODO: erase the following line and implement this function. In case you choose not to use this function, 
        # use 'pass' instead of the following line.
        pass


    ########## helper functions in class ##########
    #TODO: add here helper functions in class, if needed
    def iter_time_limit(self, num_of_leaves, last_time, last_depth):
        avr_time_per_node = last_time / num_of_leaves
        next_iter_leaves = num_of_leaves + pow(3, last_depth + 1)

        res = ((avr_time_per_node * next_iter_leaves) + last_time)

        return res


    def succ_available_moves(self, state, succ_pos):
        count = 0
        for d in self.directions:
            i = succ_pos[0] + d[0]
            j = succ_pos[1] + d[1]
            # check legal move
            if 0 <= i < len(state.board) and 0 <= j < len(state.board[0]) and (state.board[i][j] not in [-1, 1, 2]):
                count = count + 1
        return count

    def state_available_moves(self, state, maximizing_player):
        count = 0
        x, y = self.get_pos(state, maximizing_player)

        for d in self.directions:
            i = x + d[0]
            j = y + d[1]
            # check legal move
            if 0 <= i < len(state.board) and 0 <= j < len(state.board[0]) and (state.board[i][j] not in [-1, 1, 2]):
                count = count + 1

        return count

    def state_available_moves_list(self, state, maximizing_player):
        moves = []
        x, y = self.get_pos(state, maximizing_player)

        for d in self.directions:
            i = x + d[0]
            j = y + d[1]
            # check legal move
            if 0 <= i < len(state.board) and 0 <= j < len(state.board[0]) and (state.board[i][j] not in [-1, 1, 2]):
                moves.append((i, j))

        return moves

    def mandistance_best_fruit(self, state):
        if bool(self.get_fruits_pos(state)) is False:
            return None

        x, y = self.get_pos(state, True)
        fruits = self.get_fruits_pos(state)
        fruits_sorted = {k: v for k, v in sorted(fruits.items(), key=lambda item: item[1], reverse=True)}
        fruits_sorted_keys = list(fruits_sorted.keys())
        md = abs(fruits_sorted_keys[0][0] - x) + abs(fruits_sorted_keys[0][1] - y)

        return md

    def state_score_for_player(self, state, maximizing_player):
        num_steps_available = 0
        x, y = self.get_pos(state, maximizing_player)
        for d in self.directions:
            i = x + d[0]
            j = y + d[1]

            # check legal move
            if 0 <= i < len(state.board) and 0 <= j < len(state.board[0]) and (state.board[i][j] not in [-1, 1, 2]):
                num_steps_available += 1

        if num_steps_available == 0:
            return -1
        else:
            return 4 - num_steps_available

    def available_cells(self, state):
        available = 0
        for i in range(len(state.board)):
            for j in range(len(state.board[0])):
                if state.board[i][j] not in [-1, 1, 2]:
                    available = available + 1
        return available

    def blocked_cells(self, state):
        blocked_count = 1
        x, y = self.get_pos(state, True)
        x_rival, y_rival = self.get_pos(state, False)

        if x_rival > x:
            row_begin = x
            row_end = x_rival
        else:
            row_begin = x_rival
            row_end = x
        if y_rival > y:
            col_begin = y
            col_end = y_rival
        else:
            col_begin = y_rival
            col_end = y

        for i in range(row_begin, row_end + 1):
            for j in range(col_begin, col_end + 1):
                if state.board[i][j] != 0 and state.board[i][j] <= 2:
                    blocked_count += 1
        return blocked_count

    def succ_mandist_from_best_fruit(self, state, succ_moves):
        succ_md_dist = {}

        fruits = self.get_fruits_pos(state)
        if bool(fruits) is False:
            return succ_md_dist

        fruits_sorted = {k: v for k, v in sorted(fruits.items(), key=lambda item: item[1], reverse=True)}
        fruits_sorted_keys = list(fruits_sorted.keys())

        for move in succ_moves:
            mandist_val = abs(move[0] - fruits_sorted_keys[0][0]) + abs(move[1] - fruits_sorted_keys[0][1])
            succ_md_dist[move] = mandist_val

        return succ_md_dist

    def most_dense_quarter(self, state):
        quarters = [0, 0, 0, 0]
        fruits = self.get_fruits_pos(state)
        if bool(fruits) is False:
            return None

        fruits_pos = list(fruits.keys())
        max_rows = len(state.board) - 1
        max_cols = len(state.board[0]) - 1
        for pos in fruits_pos:
            if pos[0] <= max_rows / 2 and pos[1] <= max_cols / 2:
                quarters[3] = quarters[3] + 1
            elif pos[0] >= max_rows / 2 and pos[1] <= max_cols / 2:
                quarters[1] = quarters[1] + 1
            elif pos[0] >= max_rows / 2 and pos[1] >= max_cols / 2:
                quarters[0] = quarters[0] + 1
            elif pos[0] <= max_rows / 2 and pos[1] >= max_cols / 2:
                quarters[2] = quarters[2] + 1

        return quarters.index(max(quarters)) + 1

    def location_in_quarter(self, state, maximizing_player):
        x, y = self.get_pos(state, maximizing_player)
        max_rows = len(state.board) - 1
        max_cols = len(state.board[0]) - 1

        if x <= max_rows / 2 and y <= max_cols / 2:
            return 4
        elif x >= max_rows / 2 and y <= max_cols / 2:
            return 2
        elif x >= max_rows / 2 and y >= max_cols / 2:
            return 1
        elif x <= max_rows / 2 and y >= max_cols / 2:
            return 3

    def heuristic(self, state):
        # number of available moves for each player
        available_moves_count = self.state_available_moves(state, True)
        rival_available_moves_count = self.state_available_moves(state, False)

        # list of available moves for each player
        available_moves_list = self.state_available_moves_list(state, True)
        rival_available_moves_list = self.state_available_moves_list(state, False)

        # list of available moves from the successor move
        succ_available_moves = [move for move in available_moves_list if self.succ_available_moves(state, move) >= 1]

        # dictionary where keys are succ available moves and values are manhattan distance from best fruit
        #succ_best_fruit_mandist = self.succ_mandist_from_best_fruit(state, succ_available_moves)
        #succ_ratio_best_fruit = 0
        #if bool(succ_best_fruit_mandist):
        #    succ_ratio_best_fruit = sum(succ_best_fruit_mandist.values()) / len(succ_best_fruit_mandist)

        # list of moves that could possibly block rival's moves
        possible_blocking_moves = [move for move in available_moves_list if move in rival_available_moves_list]

        # manhattan distance from the fruit with the best score
        #mandist_from_best_fruit = self.mandistance_best_fruit(state)

        # dictionary of fruits with keys as positions and values as fruits' scores
        fruits = self.get_fruits_pos(state)

        #score_with_fruits = 0
        # if dictionary of fruits is not empty for this state
        #if bool(fruits):
            # list of moves that could pass in fruit blocks
            #fruits_in_path = [move for move in available_moves_list if move in fruits.keys()]
            #if bool(fruits_in_path):
                # total score possible if we take fruits in path
                #score_with_fruits = sum([fruits[move] for move in fruits_in_path]) / len(fruits_in_path)

        # state score for each player (the function they provided in SimplePlayer
        #state_score = self.state_score_for_player(state, True)
        #rival_state_score = self.state_score_for_player(state, False)

        #quarter_with_dense_fruit = self.most_dense_quarter(state)
        #is_in_dense_quarter, rival_is_in_dense_quarter = False, False
        #if quarter_with_dense_fruit is not None:
        #    is_in_dense_quarter = self.location_in_quarter(state, True) == quarter_with_dense_fruit
        #    rival_is_in_dense_quarter = self.location_in_quarter(state, False) == quarter_with_dense_fruit

        # calculate board ratio according to available cells with possible blocked moves and board's total size
        board_total_size = len(state.board) * len(state.board[0])
        available_cells = self.available_cells(state)
        blocked_moves = self.blocked_cells(state)
        ratio = blocked_moves / (board_total_size - available_cells)

        heuristic_result = ((2 * available_moves_count) - (1 * rival_available_moves_count)
                            + len(possible_blocking_moves) + len(succ_available_moves))

        if ratio < 0.4:
            if bool(fruits):
                heuristic_result = heuristic_result - rival_available_moves_count + 0.3 * state.players_scores[0]
            else:
                heuristic_result = heuristic_result - rival_available_moves_count + 0.1 * state.players_scores[0]
        else:
            if bool(fruits):
                heuristic_result = heuristic_result + available_moves_count + len(succ_available_moves) \
                                   + 0.2 * state.players_scores[0]
            else:
                heuristic_result = heuristic_result + available_moves_count + len(succ_available_moves) \
                                   + 0.1 * state.players_scores[0]

        return heuristic_result


    def get_fruits_on_board(self, board):
        fruits = {}
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] > 2:
                    pos = (i, j)
                    fruits[pos] = board[i][j]
        return fruits

    '''
        returns a dictionary of fruits positions and values in current state.
    '''

    def get_fruits_pos(self, state):
        fruits = {}
        for i in range(len(state.board)):
            for j in range(len(state.board[0])):
                if state.board[i][j] > 2:
                    pos = (i, j)
                    fruits[pos] = state.board[i][j]
        return fruits

    def moves_available(self, state, maximizing_player):
        count = 0
        if maximizing_player:
            x, y = self.get_pos(state, True)
            for d in self.directions:
                i = x + d[0]
                j = y + d[1]

                if 0 <= i < len(state.board) and 0 <= j < len(state.board[0]) and (
                        state.board[i][j] not in [-1, 1, 2]):
                    count = count + 1
        else:
            x, y = self.get_pos(state, False)
            for d in self.directions:
                i = x + d[0]
                j = y + d[1]

                if 0 <= i < len(state.board) and 0 <= j < len(state.board[0]) and (
                        state.board[i][j] not in [-1, 1, 2]):
                    count = count + 1
        return count


    ########## helper functions for AlphaBeta algorithm ##########
    #TODO: add here the utility, succ, and perform_move functions used in AlphaBeta algorithm
    def goal(self, state, maximizing_player):
        available_moves = self.moves_available(state, True)
        available_moves_rival = self.moves_available(state, False)
        if maximizing_player:
            if available_moves == 0:
                return True
        else:
            if available_moves_rival == 0:
                return True
        return False

    def utility(self, state):
        available_moves = self.moves_available(state, True)
        available_moves_rival = self.moves_available(state, False)

        if available_moves == 0 or available_moves_rival == 0:
            if available_moves > 0:
                if state.players_scores[0] + self.penalty_score > state.players_scores[1]:
                    return 50000 + state.players_scores[0]
                elif state.players_scores[0] + self.penalty_score < state.players_scores[1]:
                    return -50000 - state.players_scores[1]
                else:
                    return 0
            elif available_moves_rival > 0:
                if state.players_scores[0] > state.players_scores[1] + self.penalty_score:
                    return 50000 + state.players_scores[0]
                elif state.players_scores[0] < state.players_scores[1] + self.penalty_score:
                    return -50000 - state.players_scores[1]
                else:
                    return 0
            else:
                if state.players_scores[0] > state.players_scores[1]:
                    return 50000 + state.players_scores[0]
                elif state.players_scores[0] < state.players_scores[1]:
                    return -50000 - state.players_scores[1]
                else:
                    return 0
        return 0

    '''
        returns player's current position
    '''

    def get_pos(self, state, maximizing_player):
        if maximizing_player:
            pos = np.where(state.board == 1)
        else:
            pos = np.where(state.board == 2)
        return tuple(ax[0] for ax in pos)

    '''
        this returns an array for available steps
    '''

    def succ(self, state, maximizing_player):
        children = []
        x, y = self.get_pos(state, maximizing_player)

        for d in self.directions:
            i = x + d[0]
            j = y + d[1]

            # check legal move
            if 0 <= i < len(state.board) and 0 <= j < len(state.board[0]) and (state.board[i][j] not in [-1, 1, 2]):
                pos = (i, j)
                fruits = copy.deepcopy(state.fruits)
                scores = copy.deepcopy(state.players_scores)
                board = copy.deepcopy(state.board)
                if state.board[i][j] > 2:
                    if maximizing_player:
                        scores[0] = scores[0] + state.board[i][j]
                    else:
                        scores[1] = scores[1] + state.board[i][j]
                    del fruits[pos]
                if maximizing_player:
                    board[pos] = 1
                else:
                    board[pos] = 2
                board[x][y] = -1
                new_state = GameState(board, fruits, scores)
                children.append(new_state)

        return children

class GameState:
    board = None
    fruits = {}
    players_scores = None
    '''
            board - the board state.
            fruits - a dictionary of fruit positions as 'keys' and their values as 'values'.
            players_scores - tuple of two values [0] is player 1 score, [1] is player 2 score.
    '''

    def __init__(self, board, fruits, players_scores):
        self.board = copy.deepcopy(board)
        self.fruits = copy.deepcopy(fruits)
        self.players_scores = copy.deepcopy(players_scores)
