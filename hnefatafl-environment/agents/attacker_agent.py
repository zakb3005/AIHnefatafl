import random
import math

class AttackerAgent:
    def __init__(self, env, max_depth=4):
        self.env = env
        self.max_depth = max_depth

    def select_actions(self):
        action = self.select_action()
        return {self.env.agent_selection: action}

    def select_action(self):
        agent = self.env.agent_selection
        valid_moves = self.env.get_valid_moves(agent)
        if not valid_moves:
            return None

        move_scores = []
        alpha = -float('inf')
        beta = float('inf')

        for move in valid_moves:
            snapshot = self.make_move(agent, move)
            score = self.min_value(self.max_depth - 1, alpha, beta)
            self.undo_move(snapshot)
            move_scores.append(score)

        move_probabilities = self.softmax(move_scores, temperature=0.5)
        selected_move_index = self.weighted_choice(move_probabilities)
        return selected_move_index

    def softmax(self, scores, temperature=1.0):
        adjusted_scores = [score / temperature for score in scores]
        max_score = max(adjusted_scores)
        exp_scores = [math.exp(score - max_score) for score in adjusted_scores]
        sum_exp_scores = sum(exp_scores)
        probabilities = [exp_score / sum_exp_scores for exp_score in exp_scores]
        return probabilities

    def weighted_choice(self, probabilities):
        cumulative_distribution = []
        cumulative = 0.0
        for p in probabilities:
            cumulative += p
            cumulative_distribution.append(cumulative)
        r = random.random()
        for index, cumulative_prob in enumerate(cumulative_distribution):
            if r < cumulative_prob:
                return index
        return len(probabilities) - 1

    def max_value(self, depth, alpha, beta):
        if depth == 0 or self.env.terminations['attacker']:
            return self.evaluate_state()

        agent = 'attacker'
        valid_moves = self.env.get_valid_moves(agent)
        if not valid_moves:
            return self.evaluate_state()

        value = -float('inf')
        for move in valid_moves:
            snapshot = self.make_move(agent, move)
            value = max(value, self.min_value(depth - 1, alpha, beta))
            self.undo_move(snapshot)
            if value >= beta:
                return value
            alpha = max(alpha, value)
        return value

    def min_value(self, depth, alpha, beta):
        if depth == 0 or self.env.terminations['defender']:
            return self.evaluate_state()

        agent = 'defender'
        valid_moves = self.env.get_valid_moves(agent)
        if not valid_moves:
            return self.evaluate_state()

        value = float('inf')
        for move in valid_moves:
            snapshot = self.make_move(agent, move)
            value = min(value, self.max_value(depth - 1, alpha, beta))
            self.undo_move(snapshot)
            if value <= alpha:
                return value
            beta = min(beta, value)
        return value

    def make_move(self, agent, move):
        snapshot = {
            'board': self.env.board.copy(),
            'king_position': self.env.king_position,
            'defender_pieces': list(self.env.defender_pieces),
            'attacker_pieces': list(self.env.attacker_pieces),
            'agent_selection': self.env.agent_selection,
            'terminations': dict(self.env.terminations),
            'rewards': dict(self.env.rewards),
        }

        self.env.apply_move(agent, move, actual_move=False)
        self.env.check_captures(agent, actual_move=False)
        game_over, winner = self.env.check_game_over()
        if not game_over:
            self.env.agent_selection = 'defender' if agent == 'attacker' else 'attacker'

        return snapshot

    def undo_move(self, snapshot):
        self.env.board = snapshot['board']
        self.env.king_position = snapshot['king_position']
        self.env.defender_pieces = snapshot['defender_pieces']
        self.env.attacker_pieces = snapshot['attacker_pieces']
        self.env.agent_selection = snapshot['agent_selection']
        self.env.terminations = snapshot['terminations']
        self.env.rewards = snapshot['rewards']

    def evaluate_state(self):
        if self.env.king_position is None:
            return 1000

        score = 0

        # 1. Fewer Defenders Remaining is Better
        num_defenders = len(self.env.defender_pieces)
        score += (8 - num_defenders) * 20

        # 2. King's Distance to Center
        king_row, king_col = self.env.king_position
        center = self.env.board_size // 2
        distance_to_center = abs(king_row - center) + abs(king_col - center)
        score -= distance_to_center * 5

        # 3. Attackers’ Proximity to King
        attackers_near_king = self.count_attackers_near_king()
        score += attackers_near_king * 10

        # 4. Potential Captures of Defenders (Increased weight)
        potential_captures = self.count_potential_captures('attacker')
        score += potential_captures * 50

        # 5. Blocking Escape Routes
        escape_routes_blocked = self.count_blocked_escape_routes()
        score += escape_routes_blocked * 5

        # 6. King Threat Level
        king_threat = self.assess_king_threat()
        score += king_threat * 20

        # 7. King Can Be Captured Next
        if self.king_can_be_captured_next():
            score += 500

        return score

    def king_can_be_captured_next(self):
        king_pos = self.env.king_position
        if king_pos is None:
            return False

        return self.is_piece_capturable(king_pos, 'attacker')

    def count_attackers_near_king(self):
        king_row, king_col = self.env.king_position
        attackers_nearby = 0
        for attacker_pos in self.env.attacker_pieces:
            attacker_row, attacker_col = attacker_pos
            distance = abs(king_row - attacker_row) + abs(king_col - attacker_col)
            if distance <= 3:
                attackers_nearby += 1
        return attackers_nearby

    def count_potential_captures(self, agent):
        potential_captures = 0
        opponent_pieces = self.env.defender_pieces if agent == 'attacker' else self.env.attacker_pieces

        for pos in opponent_pieces:
            if self.is_piece_capturable(pos, agent):
                potential_captures += 1
        return potential_captures

    def is_piece_capturable(self, pos, agent):
        row, col = pos
        friend_value = 2 if agent == 'attacker' else 1
        enemy_value = 1 if agent == 'attacker' else 2
        castle_position = self.env.castle_position

        directions = [((-1, 0), (1, 0)), ((0, -1), (0, 1))]
        for dir1, dir2 in directions:
            adj1_row, adj1_col = row + dir1[0], col + dir1[1]
            adj2_row, adj2_col = row + dir2[0], col + dir2[1]
            adj1_friend = adj2_friend = False

            if 0 <= adj1_row < self.env.board_size and 0 <= adj1_col < self.env.board_size:
                adj1_piece = self.env.board[adj1_row, adj1_col]
                if (adj1_row, adj1_col) == castle_position:
                    adj1_friend = False
                else:
                    adj1_friend = adj1_piece == friend_value
            else:
                adj1_friend = False

            if 0 <= adj2_row < self.env.board_size and 0 <= adj2_col < self.env.board_size:
                adj2_piece = self.env.board[adj2_row, adj2_col]
                if (adj2_row, adj2_col) == castle_position:
                    adj2_friend = False
                else:
                    adj2_friend = adj2_piece == friend_value
            else:
                adj2_friend = False

            if adj1_friend and adj2_friend:
                return True
        return False

    def count_blocked_escape_routes(self):
        blocked_routes = 0
        king_row, king_col = self.env.king_position
        board_size = self.env.board_size
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in directions:
            row, col = king_row, king_col
            while 0 <= row + dr < board_size and 0 <= col + dc < board_size:
                row += dr
                col += dc
                if self.env.board[row, col] != 0:
                    if self.env.board[row, col] == 2:
                        blocked_routes += 1
                    break
                if row == 0 or row == board_size - 1 or col == 0 or col == board_size - 1:
                    break
        return blocked_routes

    def assess_king_threat(self):
        king_row, king_col = self.env.king_position
        threats = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for dr, dc in directions:
            adj_row, adj_col = king_row + dr, king_col + dc
            if 0 <= adj_row < self.env.board_size and 0 <= adj_col < self.env.board_size:
                if self.env.board[adj_row, adj_col] == 2:
                    threats += 1
        return threats