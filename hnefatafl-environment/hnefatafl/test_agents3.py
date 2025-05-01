from hnefatafl.tablut_env import TablutEnv
from agents.defender_agent3 import DefenderAgent
from agents.attacker_agent import AttackerAgent

if __name__ == "__main__":
    env = TablutEnv(render_mode='human')
    defender_agent = DefenderAgent(env, max_depth=3)
    attacker_agent = AttackerAgent(env, max_depth=2)

    wins = {
        "defender": 0,
        "attacker": 0
    }
    total_moves = []
    num_games = 10
    for i in range(num_games):
        env.reset()
        move_count = 0
        while True:
            agent = env.agent_selection
            if env.terminations[agent]:
                env.step(None)
                continue

            if agent == "defender":
                actions = defender_agent.select_actions()
            else:
                actions = attacker_agent.select_actions()

            action = actions[agent]
            env.step(action)
            move_count += 1

            if all(env.terminations.values()):
                if env.rewards["defender"] > 0:
                    winner = "Defender"
                    wins["defender"] += 1
                else:
                    winner = "Attacker"
                    wins["attacker"] += 1
                print(f"Game {i+1} winner: {winner}, Moves: {move_count}\n")
                total_moves.append(move_count)
                break
    
    defender_win_rate = wins['defender'] / num_games * 100
    avg_game_length = sum(total_moves) / num_games

    print(f"Defender win rate: {defender_win_rate}%")
    print(f"Average game length: {avg_game_length} moves")