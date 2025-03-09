import random
import numpy as np
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

# Define state schema for LangGraph
class GameStateSchema(TypedDict):
    board: list
    current_player: int
    game_over: bool
    winner: str

# Game class
class GameState:
    def __init__(self):
        self.board = [" " for _ in range(9)]
        self.current_player = 0  # 0 for Human, 1 for AI
        self.game_over = False
        self.winner = ""

    def to_dict(self) -> GameStateSchema:
        return {
            "board": self.board.copy(),
            "current_player": self.current_player,
            "game_over": self.game_over,
            "winner": self.winner,
        }

    @classmethod
    def from_dict(cls, state_dict: GameStateSchema) -> "GameState":
        state = cls()
        state.board = state_dict.get("board", [" " for _ in range(9)])
        state.current_player = state_dict.get("current_player", 0)
        state.game_over = state_dict.get("game_over", False)
        state.winner = state_dict.get("winner", "")
        return state

    def print_board(self):
        for x in range(3):
            print("|", end="")
            for y in range(3):
                print(self.board[x * 3 + y], end="|")
            print("\n---------")

# Player class
class Player:
    def __init__(self, name, symbol, epsilon=0.99):
        self.name = name
        self.symbol = symbol
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}
        self.states = []

    def make_move(self, board):
        valid_moves = [x for x in range(len(board)) if board[x] == " "]

        if self.name == "Human":
            print(f"Your valid moves are: {valid_moves}")
            move = -1
            while move not in valid_moves:
                print(f"Where would you like to move (0-8)?")
                try:
                    move = int(input())
                    if move not in valid_moves:
                        print("Invalid move. Please choose a valid move.")
                except ValueError:
                    print("Invalid input. Enter a number between 0 and 8.")
            board[move] = self.symbol
            return

        elif self.name == "AI":
            state = tuple(board)
            if state not in self.q_table:
                self.q_table[state] = [0 for _ in range(len(valid_moves))]  

            if random.random() < self.epsilon:
                move_index = random.randint(0, len(valid_moves) - 1)
                move = valid_moves[move_index]
            else:
                move_index = np.argmax(self.q_table[state])  
                move = valid_moves[move_index]

            self.states.append((state, move_index))
            board[move] = self.symbol
            return

    def update_q_table(self, reward):
        alpha = 0.1  
        gamma = 0.9  
        for (state, action_index) in self.states:
            current_q = self.q_table.get(state, [0] * 9)
            max_future_q = max(current_q) if current_q else 0
            if state not in self.q_table:
                self.q_table[state] = current_q
            self.q_table[state][action_index] += alpha * (reward + gamma * max_future_q - self.q_table[state][action_index])
        self.states.clear()

# LangGraph Nodes
def human_move_node(state: GameStateSchema, player: Player) -> GameStateSchema:
    game_state = GameState.from_dict(state)
    if not game_state.game_over:
        player.make_move(game_state.board)
        game_state.current_player = 1
        print("\nYour move:")
        game_state.print_board()  # Print board after human's move
    return game_state.to_dict()

def ai_move_node(state: GameStateSchema, player: Player) -> GameStateSchema:
    game_state = GameState.from_dict(state)
    if not game_state.game_over:
        player.make_move(game_state.board)
        game_state.current_player = 0
        print("\nAI's move:")
        game_state.print_board()  # Print board after AI's move
    return game_state.to_dict()

def check_win_node(state: GameStateSchema) -> GameStateSchema:
    game_state = GameState.from_dict(state)
    
    win_conditions = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    ]
    
    # Print board state for debugging
    print("\nChecking win condition for board state:")
    game_state.print_board()
    
    # Check for a win
    for (i, j, k) in win_conditions:
        if (
            game_state.board[i] != " " 
            and game_state.board[i] == game_state.board[j] 
            and game_state.board[j] == game_state.board[k]
        ):
            game_state.winner = "Human" if game_state.board[i] == "X" else "AI"
            game_state.game_over = True
            print(f"\nWinner detected: {game_state.winner}")  # Debugging output
            return game_state.to_dict()  # Exit immediately after detecting a win
    
    # Check for tie (only if no winner and board is full)
    if " " not in game_state.board:
        game_state.winner = "Tie"
        game_state.game_over = True
        print("\nGame is a tie.")  # Debugging output
    
    return game_state.to_dict()


# Create game graph
def create_game_graph(human_player: Player, ai_player: Player) -> StateGraph:
    graph = StateGraph(GameStateSchema)

    graph.add_node("human_move", lambda state: human_move_node(state, human_player))
    graph.add_node("ai_move", lambda state: ai_move_node(state, ai_player))
    graph.add_node("check_win", check_win_node)

    graph.add_edge("human_move", "check_win")
    graph.add_edge("ai_move", "check_win")

    def decide_next_node(state: GameStateSchema) -> Literal["human_move", "ai_move", END]:
        game_state = GameState.from_dict(state)
        if game_state.game_over:
            return END  # Terminate graph immediately if game is over
        else:
            return "human_move" if game_state.current_player == 0 else "ai_move"

    graph.add_conditional_edges(
        "check_win",
        decide_next_node,
        {
            END: END,
            "human_move": "human_move",
            "ai_move": "ai_move",
        }
    )

    graph.set_entry_point("human_move")
    return graph.compile()

# Main game logic
def start_game_with_langgraph():
    human_player = Player("Human", "X", 0)
    ai_player = Player("AI", "O", 0.99)

    game_graph = create_game_graph(human_player, ai_player)

    while True:
        print("\nNew game starts!")
        game_state = GameState().to_dict()
        initial_state = GameState.from_dict(game_state)
        print("Initial board:")
        initial_state.print_board()

        # Run the game until it ends
        for _ in game_graph.stream(game_state):
            game_state = _

        final_state = GameState.from_dict(game_state)

        # **Ensure we print the correct final board**
        print("\nFinal board:")
        final_state.print_board()

        # **Print correct winner message**
        if final_state.winner == "AI":
            print("AI wins!")
            human_player.update_q_table(-1)
            ai_player.update_q_table(1)
        elif final_state.winner == "Human":
            print("Human wins!")
            human_player.update_q_table(1)
            ai_player.update_q_table(-1)
        else:
            print("It's a tie!")  # This should now only appear if it's truly a tie.

        play_again = input("\nDo you want to play again? (yes/no): ").strip().lower()
        if play_again != "yes":
            print("Thank you for playing!")
            break

if __name__ == "__main__":
    start_game_with_langgraph()