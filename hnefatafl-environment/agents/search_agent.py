class SearchAgent:
    def __init__(self, env, max_depth=3):
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
            return -1000
        king_row, king_col = self.env.king_position
        min_distance = min(king_row, self.env.board_size - 1 - king_row, king_col, self.env.board_size - 1 - king_col)
        score = (self.env.board_size - 1 - min_distance) * 10
        return score