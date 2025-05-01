class DefenderAgent:
    def __init__(self, env, max_depth=4):
        self.env = env
        self.max_depth = max_depth
        self.visited_states = set()
        self.state_repetition_penalty = -50

    def select_actions(self):
        action = self.select_action()
        return {self.env.agent_selection: action}

    def select_action(self):
        agent = self.env.agent_selection
        valid_moves = self.env.get_valid_moves(agent)
        if not valid_moves:
            return None

        best_move = None
        best_score = -float('inf')
        alpha = -float('inf')
        beta = float('inf')

        for idx, move in enumerate(valid_moves):
            snapshot = self.make_move(agent, move)
            score = self.min_value(self.max_depth - 1, alpha, beta)
            self.undo_move(snapshot)

            if score > best_score:
                best_score = score
                best_move = idx
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break

        return best_move

    def max_value(self, depth, alpha, beta):
        if depth == 0 or self.env.terminations['defender']:
            return self.evaluate_state()

        agent = 'defender'
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
        if depth == 0 or self.env.terminations['attacker']:
            return self.evaluate_state()

        agent = 'attacker'
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
            self.env.agent_selection = 'attacker' if agent == 'defender' else 'defender'

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
            return -10000

        score = 0

        # 1. King's Distance to Edge (Increase weight)
        king_row, king_col = self.env.king_position
        distance_to_edges = [
            king_row,
            self.env.board_size - 1 - king_row,
            king_col,
            self.env.board_size - 1 - king_col
        ]
        min_distance = min(distance_to_edges)
        score += (self.env.board_size - 1 - min_distance) * 50

        # 2. Penalty if King is in Castle after many moves
        if (king_row, king_col) == self.env.castle_position and self.env.move_count > 10:
            score -= 100

        # 3. Defenders Near the King (Reduced weight)
        defenders_near_king = self.count_defenders_near_king()
        score += defenders_near_king * 10

        # 4. King's Mobility
        king_legal_moves = self.count_king_legal_moves()
        score += king_legal_moves * 20

        # 5. Number of Defenders Remaining
        num_defenders = len(self.env.defender_pieces)
        score += num_defenders * 10

        # 6. King Threat Level
        king_threat = self.assess_king_threat()
        score -= king_threat * 50

        # 7. Potential Captures of Attackers (Increased weight)
        potential_captures = self.count_potential_captures()
        score += potential_captures * 50

        # 8. Defender Mobility
        defender_mobility = self.calculate_defender_mobility()
        score += defender_mobility * 5

        # 9. Penalize Repetition
        board_hash = self.hash_board_state()
        if board_hash in self.visited_states:
            score += self.state_repetition_penalty
        else:
            self.visited_states.add(board_hash)

        return score

    def hash_board_state(self):
        return hash(self.env.board.tostring())

    def count_defenders_near_king(self):
        if self.env.king_position is None:
            return 0
        king_row, king_col = self.env.king_position
        defenders_nearby = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            adj_row, adj_col = king_row + dr, king_col + dc
            if 0 <= adj_row < self.env.board_size and 0 <= adj_col < self.env.board_size:
                if self.env.board[adj_row, adj_col] == 1:
                    defenders_nearby += 1
        return defenders_nearby

    def count_king_legal_moves(self):
        if self.env.king_position is None:
            return 0
        king_row, king_col = self.env.king_position
        legal_moves = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            steps = 1
            while True:
                new_row = king_row + dr * steps
                new_col = king_col + dc * steps
                if 0 <= new_row < self.env.board_size and 0 <= new_col < self.env.board_size:
                    if self.env.board[new_row, new_col] == 0:
                        if (new_row, new_col) != self.env.castle_position:
                            legal_moves += 1
                        steps += 1
                    else:
                        break
                else:
                    break
        return legal_moves

    def assess_king_threat(self):
        if self.env.king_position is None:
            return 0
        king_row, king_col = self.env.king_position
        threats = 0
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            adj_row, adj_col = king_row + dr, king_col + dc
            if 0 <= adj_row < self.env.board_size and 0 <= adj_col < self.env.board_size:
                piece = self.env.board[adj_row, adj_col]
                if piece == 2:
                    threats += 1
                elif piece == 0:
                    if self.is_square_threatened((adj_row, adj_col)):
                        threats += 0.5
        return threats

    def is_square_threatened(self, square):
        row, col = square
        for attacker_pos in self.env.attacker_pieces:
            attacker_row, attacker_col = attacker_pos
            if attacker_row == row or attacker_col == col:
                if self.is_path_clear(attacker_pos, square):
                    return True
        return False

    def is_path_clear(self, from_pos, to_pos):
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        if from_row == to_row:
            step = 1 if to_col > from_col else -1
            for col in range(from_col + step, to_col, step):
                if self.env.board[from_row, col] != 0:
                    return False
            return True
        elif from_col == to_col:
            step = 1 if to_row > from_row else -1
            for row in range(from_row + step, to_row, step):
                if self.env.board[row, from_col] != 0:
                    return False
            return True
        return False

    def count_potential_captures(self):
        potential_captures = 0
        for attacker_pos in self.env.attacker_pieces:
            if self.is_attacker_capturable(attacker_pos):
                potential_captures += 1
        return potential_captures

    def is_attacker_capturable(self, pos):
        row, col = pos
        friend_value = 1
        enemy_value = 2
        directions = [((-1, 0), (1, 0)), ((0, -1), (0, 1))]
        for dir1, dir2 in directions:
            adj1_row, adj1_col = row + dir1[0], col + dir1[1]
            adj2_row, adj2_col = row + dir2[0], col + dir2[1]
            adj1_friend = adj2_friend = False

            if 0 <= adj1_row < self.env.board_size and 0 <= adj1_col < self.env.board_size:
                adj1_piece = self.env.board[adj1_row, adj1_col]
                if adj1_piece == friend_value or adj1_piece == 3:
                    adj1_friend = True
            if 0 <= adj2_row < self.env.board_size and 0 <= adj2_col < self.env.board_size:
                adj2_piece = self.env.board[adj2_row, adj2_col]
                if adj2_piece == friend_value or adj2_piece == 3:
                    adj2_friend = True

            if adj1_friend and adj2_friend:
                return True
        return False

    def calculate_defender_mobility(self):
        total_moves = 0
        for defender_pos in self.env.defender_pieces:
            moves = self.get_piece_legal_moves(defender_pos)
            total_moves += len(moves)
        return total_moves

    def get_piece_legal_moves(self, piece_pos):
        from_row, from_col = piece_pos
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            steps = 1
            while True:
                to_row = from_row + dr * steps
                to_col = from_col + dc * steps
                if 0 <= to_row < self.env.board_size and 0 <= to_col < self.env.board_size:
                    if self.env.board[to_row, to_col] == 0:
                        if (to_row, to_col) != self.env.castle_position:
                            moves.append(((from_row, from_col), (to_row, to_col)))
                        steps += 1
                    else:
                        break
                else:
                    break
        return moves