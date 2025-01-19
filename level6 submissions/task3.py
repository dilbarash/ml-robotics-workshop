import numpy as np
import random
import json
import matplotlib.pyplot as plt
with open('datasets/grid_environment.json') as f:
    grid_data = json.load(f)
grid_size = grid_data['grid_size']
start_pos = tuple(grid_data['start_position'])
goal_pos = tuple(grid_data['end_position'])
obstacles = set(tuple(obs) for obs in grid_data['obstacles'])
reward_structure = grid_data['reward_structure']
num_states = grid_size[0] * grid_size[1]
num_actions = 4
alpha = 0.1
gamma = 0.9
epsilon = 0.2
num_episodes = 500
Q_table = np.zeros((grid_size[0], grid_size[1], num_actions))
action_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
def is_valid_move(state, action):
    row, col = state
    action_move = action_map[action]
    new_row, new_col = row + action_move[0], col + action_move[1]
    if 0 <= new_row < grid_size[0] and 0 <= new_col < grid_size[1]:
        return (new_row, new_col) not in obstacles
    return False
def get_reward(state):
    if state == goal_pos:
        return reward_structure['goal']
    if state in obstacles:
        return reward_structure['obstacle']
    if not (0 <= state[0] < grid_size[0] and 0 <= state[1] < grid_size[1]):
        return reward_structure['out_of_bounds']
    return reward_structure['step']
def get_next_state(state, action):
    row, col = state
    action_move = action_map[action]
    new_row, new_col = row + action_move[0], col + action_move[1]
    next_state = (new_row, new_col)
    if is_valid_move(state, action):
        return next_state
    return state
learning_rates = [0.1, 0.3]
discount_factors = [0.8, 0.9]
exploration_rates = [0.2, 0.3]
best_reward = -float('inf')
best_params = {}
reward_trends = {}
for alpha in learning_rates:
    for gamma in discount_factors:
        for epsilon in exploration_rates:
            Q_table = np.zeros((grid_size[0], grid_size[1], num_actions))
            cumulative_rewards = []
            for episode in range(num_episodes):
                state = start_pos
                done = False
                episode_reward = 0
                while not done:
                    if random.uniform(0, 1) < epsilon:
                        action = random.randint(0, num_actions-1)
                    else:
                        action = np.argmax(Q_table[state[0], state[1]])
                    next_state = get_next_state(state, action)
                    reward = get_reward(next_state)
                    episode_reward += reward
                    best_next_action = np.argmax(Q_table[next_state[0], next_state[1]])
                    Q_table[state[0], state[1], action] = Q_table[state[0], state[1], action] + alpha * (
                        reward + gamma * Q_table[next_state[0], next_state[1], best_next_action] - Q_table[state[0], state[1], action]
                    )
                    state = next_state
                    if state == goal_pos:
                        done = True
                cumulative_rewards.append(episode_reward)
            reward_trends[(alpha, gamma, epsilon)] = cumulative_rewards
            total_reward = sum(cumulative_rewards)
            if total_reward > best_reward:
                best_reward = total_reward
                best_params = {'alpha': alpha, 'gamma': gamma, 'epsilon': epsilon}
plt.figure(figsize=(10, 6))
for (alpha, gamma, epsilon), rewards in reward_trends.items():
    label = f"alpha={alpha}, gamma={gamma}, epsilon={epsilon}"
    plt.plot(rewards, label=label)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Cumulative Reward', fontsize=12)
plt.title('Reward Trend Over Training Episodes', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
print(f"Best Hyperparameters: {best_params}")
