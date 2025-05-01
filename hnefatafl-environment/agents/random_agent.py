import numpy as np

class RandomAgent:
    def __init__(self, env):
        self.env = env

    def select_actions(self):
        actions = {}
        for agent in self.env.agents:
            valid_moves = self.env.get_valid_moves(agent)
            if valid_moves:
                action = np.random.choice(len(valid_moves))
                actions[agent] = action
            else:
                actions[agent] = 0  # Placeholder
        return actions