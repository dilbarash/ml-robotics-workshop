import numpy as np
import random
num_states = 5
num_actions = 3
alpha = 0.1
gamma = 0.9
epsilon = 0.2
num_episodes = 1000
Q_table = np.zeros((num_states, num_actions))
def get_reward(state, action):
    reward = random.choice([-1, 0, 1])
    return reward
def get_next_state(state, action):
    return (state + action) % num_states
for episode in range(num_episodes):
    state = random.randint(0, num_states-1)
    done = False
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, num_actions-1)
        else:
            action = np.argmax(Q_table[state])
        reward = get_reward(state, action)
        next_state = get_next_state(state, action)
        best_next_action = np.argmax(Q_table[next_state])
        Q_table[state, action] = Q_table[state, action] + alpha * (
            reward + gamma * Q_table[next_state, best_next_action] - Q_table[state, action]
        )
        state = next_state
        if state == 0:
            done = True
print("Q-table after training:")
print(Q_table)
