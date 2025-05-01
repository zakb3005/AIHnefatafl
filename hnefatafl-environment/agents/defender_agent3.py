class DefenderAgent:
    def __init__(self, env, max_depth=4):
        self.env = env
        self.max_depth = max_depth
        self.visited_states = set()
        self.state_repetition_penalty = -50
        self.starting_positions = None
        self.initial_num_attackers = None

    def select_actions(self):
        action = self.select_action()
        return {self.env.agent_selection: action}

    def select_action(self):
        if self.starting_positions is None:
            self.starting_positions = self.env.defender_pieces.copy()
        if self.initial_num_attackers is None:
            self.initial_num_attackers = len(self.env.attacker_pieces)

        agent = self.env.agent_selection
        valid_moves = self.env.get_valid_moves(agent)
        if not valid_moves:
            return None

        move_scores = []
        for idx, move in enumerate(valid_moves):
            snapshot = self.make_move(agent, move)
            captures = self.check_captures_after_move(agent)
            king_can_escape = self.king_can_escape()
            king_in_threat = self.king_in_immediate_threat()
            self.undo_move(snapshot)

            if king_can_escape:
                priority = 5
            elif captures:
                priority = 4
            elif king_in_threat and self.can_defend_king(move):
                priority = 3
            elif self.is_moving_back_to_start(move):
                priority = 2
            elif self.is_move_safe(move):
                priority = 1
            else:
                priority = 0

            move_scores.append((priority, idx, move))

        move_scores.sort(reverse=True)

        best_move = None
        best_score = -float('inf')
        alpha = -float('inf')
        beta = float('inf')

        for priority, idx, move in move_scores:
            snapshot = self.make_move(agent, move)
            score = self.min_value(self.max_depth - 1, alpha, beta)
            self.undo_move(snapshot)

            if score > best_score:
                best_score = score
                best_move = idx
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break

        if best_move is not None:
            return best_move
        else:
            return 0

    def max_value(self, depth, alpha, beta):
        if depth == 0 or self.env.terminations['defender']:
            return self.evaluate_state()

        agent = 'defender'
        valid_moves = self.env.get_valid_moves(agent)
        if not valid_moves:
            return self.evaluate_state()

        move_scores = []
        for move in valid_moves:
            snapshot = self.make_move(agent, move)
            captures = self.check_captures_after_move(agent)
            king_can_escape = self.king_can_escape()
            king_in_threat = self.king_in_immediate_threat()
            self.undo_move(snapshot)

            if king_can_escape:
                priority = 5
            elif captures:
                priority = 4
            elif king_in_threat and self.can_defend_king(move):
                priority = 3
            elif self.is_moving_back_to_start(move):
                priority = 2
            elif self.is_move_safe(move):
                priority = 1
            else:
                priority = 0

            move_scores.append((priority, move))

        move_scores.sort(reverse=True)

        value = -float('inf')
        for priority, move in move_scores:
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
        num_attackers = len(self.env.attacker_pieces)
        num_defenders = len(self.env.defender_pieces)

        # King's proximity to edge
        king_row, king_col = self.env.king_position
        distance_to_edge = min(
            king_row,
            self.env.board_size - 1 - king_row,
            king_col,
            self.env.board_size - 1 - king_col
        )

        # Encourage king to escape if path is safe
        if self.king_can_escape():
            score += (self.env.board_size - 1 - distance_to_edge) * 1000
        else:
            score -= distance_to_edge * 10

        # Penalize if king is under immediate threat
        if self.king_in_immediate_threat():
            score -= 5000

        # Reward capturing attackers
        attackers_captured = self.initial_num_attackers - num_attackers
        score += attackers_captured * 1000

        # Reward defenders in starting positions
        num_defenders_in_start = sum(
            1 for pos in self.env.defender_pieces if pos in self.starting_positions
        )
        score += num_defenders_in_start * 100

        # Reward safe defenders, penalize defenders in danger
        defenders_safe = sum(
            1 for pos in self.env.defender_pieces if not self.is_piece_in_danger(pos)
        )
        defenders_in_danger = num_defenders - defenders_safe
        score += defenders_safe * 50
        score -= defenders_in_danger * 200

        # Penalize state repetitions
        board_hash = self.hash_board_state()
        if board_hash in self.visited_states:
            score += self.state_repetition_penalty
        else:
            self.visited_states.add(board_hash)

        return score

    def hash_board_state(self):
        return hash(self.env.board.tostring())

    def is_moving_back_to_start(self, move):
        from_pos, to_pos = move
        return from_pos not in self.starting_positions and to_pos in self.starting_positions

    def is_move_safe(self, move):
        agent = 'defender'
        from_pos, to_pos = move
        snapshot = self.make_move(agent, move)
        is_safe = not self.is_piece_in_danger(to_pos)
        self.undo_move(snapshot)
        return is_safe

    def is_piece_in_danger(self, pos):
        row, col = pos
        opponent = 2
        for dir1, dir2 in [((-1, 0), (1, 0)), ((0, -1), (0, 1))]:
            adj1_row, adj1_col = row + dir1[0], col + dir1[1]
            adj2_row, adj2_col = row + dir2[0], col + dir2[1]
            adj1_is_enemy = adj2_is_enemy = False

            if 0 <= adj1_row < self.env.board_size and 0 <= adj1_col < self.env.board_size:
                if self.env.board[adj1_row, adj1_col] == opponent:
                    adj1_is_enemy = True

            if 0 <= adj2_row < self.env.board_size and 0 <= adj2_col < self.env.board_size:
                if self.env.board[adj2_row, adj2_col] == opponent:
                    adj2_is_enemy = True

            if adj1_is_enemy and adj2_is_enemy:
                return True
        return False

    def king_can_escape(self):
        if self.env.king_position is None:
            return False
        return self.king_escape_path_exists(max_steps=2)

    def king_escape_path_exists(self, max_steps):
        from_row, from_col = self.env.king_position
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        board_size = self.env.board_size

        for dr, dc in directions:
            steps = 1
            while steps <= max_steps:
                to_row = from_row + dr * steps
                to_col = from_col + dc * steps
                if 0 <= to_row < board_size and 0 <= to_col < board_size:
                    if self.env.board[to_row, to_col] != 0:
                        break
                    if to_row == 0 or to_row == board_size - 1 or to_col == 0 or to_col == board_size - 1:
                        if self.is_path_safe((from_row, from_col), (to_row, to_col)):
                            return True
                    steps += 1
                else:
                    break
        return False

    def is_path_safe(self, from_pos, to_pos):
        path = self.get_straight_line_path(from_pos, to_pos)
        for pos in path:
            if self.is_square_threatened(pos):
                return False
        return True

    def get_straight_line_path(self, from_pos, to_pos):
        path = []
        if from_pos[0] == to_pos[0]:
            row = from_pos[0]
            cols = range(min(from_pos[1], to_pos[1]) + 1, max(from_pos[1], to_pos[1]))
            for col in cols:
                path.append((row, col))
        elif from_pos[1] == to_pos[1]:
            col = from_pos[1]
            rows = range(min(from_pos[0], to_pos[0]) + 1, max(from_pos[0], to_pos[0]))
            for row in rows:
                path.append((row, col))
        return path

    def is_square_threatened(self, square):
        row, col = square
        opponent = 2
        for attacker_pos in self.env.attacker_pieces:
            if attacker_pos[0] == row or attacker_pos[1] == col:
                if self.is_path_clear(attacker_pos, square):
                    return True
        return False

    def is_path_clear(self, from_pos, to_pos):
        if from_pos == to_pos:
            return True
        path = self.get_straight_line_path(from_pos, to_pos)
        for pos in path:
            if self.env.board[pos] != 0:
                return False
        return True

    def king_in_immediate_threat(self):
        if self.env.king_position is None:
            return False
        king_row, king_col = self.env.king_position
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            adj_row, adj_col = king_row + dr, king_col + dc
            if 0 <= adj_row < self.env.board_size and 0 <= adj_col < self.env.board_size:
                if self.env.board[adj_row, adj_col] == 2:
                    return True
        return False

    def can_defend_king(self, move):
        _, to_pos = move
        if self.env.king_position is None:
            return False
        return self.is_adjacent(to_pos, self.env.king_position)

    def is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def check_captures_after_move(self, agent):
        initial_attacker_pieces = set(self.env.attacker_pieces)
        self.env.check_captures(agent, actual_move=False)
        captured_attackers = initial_attacker_pieces - set(self.env.attacker_pieces)
        return len(captured_attackers) > 0