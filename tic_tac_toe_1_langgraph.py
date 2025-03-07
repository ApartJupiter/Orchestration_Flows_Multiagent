import random
import numpy as np
from langgraph.graph import StateGraph
from langgraph.pregel import Pregel

# Tic-Tac-Toe Game State
class TicTacToeState:
    def __init__(self):
        self.board = [" " for _ in range(9)]
        self.gameOver = False
        self.winner = None
        self.turn = "Human"

    def checkForWin(self):
        win_conditions = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        for (i, j, k) in win_conditions:
            if self.board[i] != " " and self.board[i] == self.board[j] == self.board[k]:
                self.winner = self.board[i]
                self.gameOver = True
                return
        
        if " " not in self.board:
            self.winner = "Tie"
            self.gameOver = True

# Q-learning AI
class QLearningAI:
    def __init__(self, symbol, epsilon=0.2, alpha=0.1, gamma=0.9):
        self.symbol = symbol
        self.q_table = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.states = []

    def get_state(self, board):
        return tuple(board)

    def choose_move(self, board):
        state = self.get_state(board)
        valid_moves = [i for i in range(9) if board[i] == " "]

        if state not in self.q_table:
            self.q_table[state] = [0] * 9

        if random.random() < self.epsilon:
            move = random.choice(valid_moves)
        else:
            q_values = [self.q_table[state][i] if i in valid_moves else -np.inf for i in range(9)]
            move = int(np.argmax(q_values))

        self.states.append((state, move))
        return move

    def update_q_table(self, reward):
        for state, move in self.states:
            if state in self.q_table:
                max_future_q = max(self.q_table[state])
                self.q_table[state][move] += self.alpha * (reward + self.gamma * max_future_q - self.q_table[state][move])
        self.states.clear()

# AI and Human Moves
def ai_move(state: TicTacToeState, ai: QLearningAI):
    if state.gameOver:
        return state

    move = ai.choose_move(state.board)
    state.board[move] = "O"
    state.turn = "Human"
    state.checkForWin()
    return state

def human_move(state: TicTacToeState):
    if state.gameOver:
        return state

    print("Your valid moves:", [i for i in range(9) if state.board[i] == " "])
    
    move = -1
    while move not in range(9) or state.board[move] != " ":
        try:
            move = int(input("Where would you like to move (0-8)? "))
        except ValueError:
            print("Invalid input. Enter a number between 0-8.")

    state.board[move] = "X"
    state.turn = "AI"
    state.checkForWin()
    return state

# Reward AI after game ends
def reward_ai(state: TicTacToeState, ai: QLearningAI):
    if state.winner == "O":
        ai.update_q_table(1)
    elif state.winner == "X":
        ai.update_q_table(-1)
    elif state.winner == "Tie":
        ai.update_q_table(0.5)
    return state

# Build LangGraph
def build_graph(ai: QLearningAI):
    graph = StateGraph(TicTacToeState)
    
    graph.add_node("human_move", human_move)
    graph.add_node("ai_move", lambda s: ai_move(s, ai))
    graph.add_node("reward_ai", lambda s: reward_ai(s, ai))

    graph.add_edge("human_move", "ai_move")
    graph.add_edge("ai_move", "reward_ai")
    graph.add_edge("reward_ai", "human_move")

    graph.set_entry_point("human_move")
    return graph.compile()

# Run the Game with LangGraph
def start_game():
    ai = QLearningAI(symbol="O")
    graph = build_graph(ai)
    pregel = Pregel(
    nodes=graph.nodes,
    channels=graph.channels,
    input_channels=graph.input_channels,
    output_channels=graph.output_channels
)

     

    state = TicTacToeState()
    for step in pregel.run(state):
        if step.gameOver:
            print("Game Over! Winner:", step.winner)
            break


    play_again = input("Play again? (yes/no): ").strip().lower()
    if play_again == "yes":
        start_game()

# Start Learning Tic-Tac-Toe AI
start_game()
