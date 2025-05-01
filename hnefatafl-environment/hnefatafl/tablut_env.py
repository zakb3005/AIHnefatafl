import functools
from copy import deepcopy
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from itertools import cycle

class TablutEnv(AECEnv):
    metadata = {"name": "tablut_v0"}

    def __init__(self, render_mode='human'):
        super().__init__()
        self.render_mode = render_mode
        self.possible_agents = ["defender", "attacker"]
        self.agent_order = cycle(self.possible_agents)
        self.agent_selection = None

        self.agents = []
        self.board = None
        self.board_size = 9
        self.action_space_size = 200

        self.num_defenders = 8
        self.num_attackers = 16
        self.defender_pieces = []
        self.attacker_pieces = []
        self.king_position = None

        self.castle_position = None
        self.castle_value = 4

        self.max_moves = 2000
        self.move_count = 0

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.agent_order = cycle(self.agents)
        self.agent_selection = next(self.agent_order)
        self.move_count = 0

        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        center = self.board_size // 2

        self.castle_position = (center, center)
        self.castle_value = 4

        self.king_position = self.castle_position
        self.board[center, center] = 3

        self.defender_pieces = []
        defender_positions = [
            (center - 1, center),
            (center + 1, center),
            (center, center - 1),
            (center, center + 1),
            (center - 2, center),
            (center + 2, center),
            (center, center - 2),
            (center, center + 2),
        ]
        for pos in defender_positions:
            self.board[pos] = 1
            self.defender_pieces.append(pos)

        self.attacker_pieces = []
        attacker_positions = [
            (0, center - 1), (0, center), (0, center + 1), (1, center),
            (self.board_size - 1, center - 1), (self.board_size - 1, center),
            (self.board_size - 1, center + 1), (self.board_size - 2, center),
            (center - 1, 0), (center, 0), (center + 1, 0), (center, 1),
            (center - 1, self.board_size - 1), (center, self.board_size - 1),
            (center + 1, self.board_size - 1), (center, self.board_size - 2),
        ]
        for pos in attacker_positions:
            self.board[pos] = 2
            self.attacker_pieces.append(pos)

        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self.was_dead_step(action)
            return

        self.move_count += 1

        valid_moves = self.get_valid_moves(agent)
        if action is not None and action < len(valid_moves):
            move = valid_moves[action]
            self.apply_move(agent, move, actual_move=True)
        else:
            pass  # Invalid move

        self.check_captures(agent, actual_move=True)

        game_over, winner = self.check_game_over()
        if self.render_mode == 'human':
            print(f"After {agent}'s move:")
            self.render()
            if game_over:
                if winner == "defender":
                    print("Game Over: Defenders win!")
                elif winner == "attacker":
                    print("Game Over: Attackers win!")
                else:
                    print("Game Over: Draw")
        if game_over:
            self.terminations = {agent: True for agent in self.agents}
            if winner == "defender":
                self.rewards["defender"] = 1
                self.rewards["attacker"] = -1
            elif winner == "attacker":
                self.rewards["defender"] = -1
                self.rewards["attacker"] = 1
        elif self.move_count >= self.max_moves:
            self.truncations = {agent: True for agent in self.agents}
            if self.render_mode == 'human':
                print("Game Over: Move limit reached.")

        self.accumulate_rewards()
        self.agent_selection = next(self.agent_order)

    def render(self):
        if self.render_mode != 'human':
            return
        piece_symbols = {
            0: '.',
            1: 'D',
            2: 'A',
            3: 'K',
            4: 'C',
        }
        print("  " + " ".join(map(str, range(self.board_size))))
        for i in range(self.board_size):
            row_symbols = []
            for j in range(self.board_size):
                symbol = piece_symbols[self.board[i, j]]
                if (i, j) == self.castle_position and self.board[i, j] != 3:
                    symbol = piece_symbols[4]
                row_symbols.append(symbol)
            print(f"{i} " + " ".join(row_symbols))
        print()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(low=0, high=4, shape=(self.board_size, self.board_size), dtype=int)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(self.action_space_size)

    def observe(self, agent):
        return self.board.copy()

    def get_valid_moves(self, agent):
        moves = []
        if agent == "defender":
            pieces = self.defender_pieces + ([self.king_position] if self.king_position else [])
        else:
            pieces = self.attacker_pieces

        for piece_pos in pieces:
            from_row, from_col = piece_pos
            piece = self.board[from_row, from_col]
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in directions:
                steps = 1
                while True:
                    to_row = from_row + dr * steps
                    to_col = from_col + dc * steps
                    if 0 <= to_row < self.board_size and 0 <= to_col < self.board_size:
                        if self.board[to_row, to_col] == 0:
                            if (to_row, to_col) == self.castle_position:
                                if piece != 3:
                                    break
                            moves.append(((from_row, from_col), (to_row, to_col)))
                            steps += 1
                        else:
                            break
                    else:
                        break
        return moves

    def apply_move(self, agent, move, actual_move=False):
        from_pos, to_pos = move
        from_row, from_col = from_pos
        to_row, to_col = to_pos

        if self.board[to_row, to_col] != 0:
            return

        piece = self.board[from_row, from_col]
        piece_type = 'King' if piece == 3 else ('Defender' if piece == 1 else 'Attacker')
        if self.render_mode == 'human' and actual_move:
            print(f"{agent.capitalize()} moves {piece_type} from {from_pos} to {to_pos}")

        self.board[to_row, to_col] = piece
        self.board[from_row, from_col] = 0

        if piece == 3:
            if from_pos == self.castle_position:
                self.board[self.castle_position] = self.castle_value
            self.king_position = to_pos
        else:
            if to_pos == self.castle_position:
                return

        if agent == "defender":
            if piece == 1:
                idx = self.defender_pieces.index(from_pos)
                self.defender_pieces[idx] = to_pos
            elif piece == 3:
                self.king_position = to_pos
        else:
            idx = self.attacker_pieces.index(from_pos)
            self.attacker_pieces[idx] = to_pos

    def check_captures(self, agent, actual_move=False):
        if agent == 'defender':
            opponent_positions = self.attacker_pieces.copy()
            opponent_type = 'attacker'
        else:
            opponent_positions = self.defender_pieces.copy()
            opponent_type = 'defender'

        for pos in opponent_positions:
            if self.is_captured(pos, opponent_type):
                if self.render_mode == 'human' and actual_move:
                    print(f"{opponent_type.capitalize()} at {pos} captured")
                self.board[pos] = 0
                if opponent_type == 'attacker':
                    self.attacker_pieces.remove(pos)
                else:
                    self.defender_pieces.remove(pos)

        if agent == 'attacker' and self.king_position is not None:
            if self.is_king_captured(self.king_position):
                if self.render_mode == 'human' and actual_move:
                    print(f"King captured at {self.king_position}")
                self.board[self.king_position] = 0
                self.king_position = None

    def is_captured(self, pos, piece_type):
        row, col = pos
        if piece_type == "defender":
            enemy_pieces = {2}
            friendly_pieces = {1, 3}
        elif piece_type == "attacker":
            enemy_pieces = {1, 3}
            friendly_pieces = {2}
        elif piece_type == "king":
            enemy_pieces = {2}
            friendly_pieces = {1}
        else:
            enemy_pieces = set()
            friendly_pieces = set()

        opposite_pairs = [((-1, 0), (1, 0)), ((0, -1), (0, 1))]

        for dir1, dir2 in opposite_pairs:
            adj1_row, adj1_col = row + dir1[0], col + dir1[1]
            adj2_row, adj2_col = row + dir2[0], col + dir2[1]

            adj1_is_enemy = False
            adj2_is_enemy = False

            if 0 <= adj1_row < self.board_size and 0 <= adj1_col < self.board_size:
                adj1_piece = self.board[adj1_row, adj1_col]
                if (adj1_row, adj1_col) == self.castle_position:
                    adj1_is_enemy = False
                else:
                    adj1_is_enemy = adj1_piece in enemy_pieces
            else:
                adj1_is_enemy = False

            if 0 <= adj2_row < self.board_size and 0 <= adj2_col < self.board_size:
                adj2_piece = self.board[adj2_row, adj2_col]
                if (adj2_row, adj2_col) == self.castle_position:
                    adj2_is_enemy = False
                else:
                    adj2_is_enemy = adj2_piece in enemy_pieces
            else:
                adj2_is_enemy = False

            if adj1_is_enemy and adj2_is_enemy:
                return True
        return False

    def is_king_captured(self, pos):
        row, col = pos
        attacker = 2

        if (row, col) == self.castle_position:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in directions:
                adj_row, adj_col = row + dr, col + dc
                if not (0 <= adj_row < self.board_size and 0 <= adj_col < self.board_size):
                    return False
                adj_piece = self.board[adj_row, adj_col]
                if adj_piece != attacker:
                    return False
            return True
        else:
            if (abs(row - self.castle_position[0]) + abs(col - self.castle_position[1])) == 1:
                castle_direction = None
                directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                for index, (dr, dc) in enumerate(directions):
                    adj_row, adj_col = row + dr, col + dc
                    if (adj_row, adj_col) == self.castle_position:
                        castle_direction = index
                        break
                if castle_direction is not None:
                    surrounding_attackers = 0
                    for index, (dr, dc) in enumerate(directions):
                        if index == castle_direction:
                            continue
                        adj_row, adj_col = row + dr, col + dc
                        if 0 <= adj_row < self.board_size and 0 <= adj_col < self.board_size:
                            adj_piece = self.board[adj_row, adj_col]
                            if adj_piece == attacker:
                                surrounding_attackers += 1
                        else:
                            pass
                    if surrounding_attackers == 3:
                        return True
                return False
            else:
                return self.is_captured(pos, "king")

    def check_game_over(self):
        if self.king_position is not None:
            row, col = self.king_position
            if row == 0 or row == self.board_size - 1 or col == 0 or col == self.board_size - 1:
                return True, "defender"
        else:
            return True, "attacker"
        return False, None

    def accumulate_rewards(self):
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]

    def was_dead_step(self, action):
        pass