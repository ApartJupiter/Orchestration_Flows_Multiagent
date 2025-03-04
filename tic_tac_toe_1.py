import numpy as np
import matplotlib.pyplot as plt

class TicTacToe:
    def __init__(self):
        self.board = np.zeros(9)  # 3x3 Tic-Tac-Toe board flattened to a 1D array
        self.done = False
        self.current_player = 1  # Player 1 starts the game

    def reset(self):
        self.board = np.zeros(9)
        self.done = False
        self.current_player = 1
        return self.board

    def is_done(self):
        # Check for a win or draw
        for i in range(3):
            # Check rows, columns, and diagonals
            if np.all(self.board[i*3:(i+1)*3] == self.current_player) or \
               np.all(self.board[i::3] == self.current_player):
                return True
        if np.all(self.board[::4] == self.current_player) or np.all(self.board[2:8:2] == self.current_player):
            return True
        if np.all(self.board != 0):
            return True
        return False

    def step(self, action):
        if self.board[action] != 0 or self.done:
            return self.board, -10, True  # Invalid move penalty

        self.board[action] = self.current_player
        if self.is_done():
            if self.current_player == 1:
                reward = 10  # Win for Player 1
            else:
                reward = -10  # Loss for Player 2
            self.done = True
        else:
            reward = -1  # Small penalty for each move to encourage quicker games
            self.current_player = -self.current_player  # Switch player

        return tuple(self.board), reward, self.done

class QLearningAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}

    def get_q_value(self, state, action):
        state_tuple = tuple(state)  # Convert numpy.ndarray to tuple
        if (state_tuple, action) not in self.q_table:
            self.q_table[(state_tuple, action)] = 0  # Initialize Q-value if not present
        return self.q_table[(state_tuple, action)]

    def update_q_value(self, state, action, reward, next_state, done):
        state_tuple = tuple(state)  # Convert numpy.ndarray to tuple
        next_state_tuple = tuple(next_state)  # Convert next_state to tuple
        best_next_action = self.get_best_action(next_state)
        max_q_next = self.get_q_value(next_state_tuple, best_next_action)
        target = reward + (self.discount_factor * max_q_next * (1 - done))
        self.q_table[(state_tuple, action)] = (1 - self.learning_rate) * self.get_q_value(state_tuple, action) + \
                                               self.learning_rate * target

    def get_best_action(self, state):
        state_tuple = tuple(state)  # Convert state to tuple
        best_action = None
        max_q_value = -float('inf')
        for action in self.action_space:
            q_value = self.get_q_value(state_tuple, action)
            if q_value > max_q_value:
                max_q_value = q_value
                best_action = action
        return best_action

    def choose_action(self, state):
        state_tuple = tuple(state)  # Convert state to tuple
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        return self.get_best_action(state_tuple)

def train_agent():
    env = TicTacToe()
    agent = QLearningAgent(action_space=list(range(9)), learning_rate=0.1, epsilon=0.3)
    episodes = 5000  # Increase the number of training episodes
    epsilon_decay = 0.995  # You can tweak this value for slower or faster decay
    min_epsilon = 0.1  # Set a minimum epsilon value to allow exploration even in later episodes
    rewards = []  # List to track rewards per episode

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_value(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        rewards.append(total_reward)

        # Decay epsilon
        agent.epsilon = max(min_epsilon, agent.epsilon * epsilon_decay)

        # Print progress every 100 episodes
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

    return rewards

# Train the agent and plot the rewards
rewards = train_agent()

# Plot rewards
plt.plot(range(len(rewards)), rewards)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.title("Training Progress")
plt.show()
