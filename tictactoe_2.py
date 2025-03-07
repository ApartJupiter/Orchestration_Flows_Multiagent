import random
import numpy as np

# Game class
class Game:
    def __init__(self, players):
        self.players = players
        self.board = [" " for x in range(9)]
        self.gameOver = False
        self.winner = ""
    
    def printBoard(self):
        for x in range(3):
            print("|", end="")
            for y in range(3):
                print(self.board[x * 3 + y], end="|")
            print("\n---------")
    
    def playGame(self):
     turn = 0  # Start with Player 1 (Human)
     while not self.gameOver:
        self.players[turn].makeMove(self.board)
        self.printBoard()
        self.checkForWin()
        if self.gameOver:
            break
        turn = 1 - turn  # Switch turns between 0 (Human) and 1 (AI)

    
    def checkForWin(self):
        win_conditions = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        for (i, j, k) in win_conditions:
            if self.board[i] != " " and self.board[i] == self.board[j] == self.board[k]:
                if self.board[i] == "X":
                    self.winner = 1  # Human wins
                    self.gameOver = True
                else:
                    self.winner = 0  # AI wins
                    self.gameOver = True
                return
        if " " not in self.board:  # Tie if no more moves are left
            self.winner = 2
            self.gameOver = True

    def reset(self):
        self.board = [" " for x in range(9)]
        self.gameOver = False
        self.winner = ""


# Player class
class Player:
    def __init__(self, name, symbol, epsilon=0.99):
        self.name = name
        self.symbol = symbol
        self.epsilon = epsilon  # Epsilon for AI exploration
        self.q_table = {}
        self.states = []
    
    def makeMove(self, board):
        validMoves = [x for x in range(len(board)) if board[x] == " "]
        
        if self.name == "Human":
            print(f"Your valid moves are: {validMoves}")
            move = -1
            while move not in validMoves:
                print(f"Where would you like to move (0-8)?")
                try:
                    move = int(input())
                    if move not in validMoves:
                        print("Invalid move. Please choose one of the valid moves.")
                except ValueError:
                    print("Invalid input. Please enter a number between 0 and 8.")
            board[move] = self.symbol
            return
        
        elif self.name == "AI":
            state = tuple(board)
            move_index = 0
            if state not in self.q_table:
                self.q_table[state] = [0 for x in range(len(validMoves))]  # Initialize Q-table for this state
            
            if random.random() < self.epsilon:
                move_index = random.randint(0, len(validMoves) - 1)  # Random move (exploration)
                move = validMoves[move_index]
            else:
                move_index = np.argmax(self.q_table[state])  # Best move based on Q-table (exploitation)
                move = validMoves[move_index]
            
            self.states.append((state, move_index))
            board[move] = self.symbol
            return

    def updateQTable(self, reward):
        alpha = 0.1  # Learning rate
        gamma = 0.9  # Discount factor
        for (state, action_index) in self.states:
            max_future_q = max(self.q_table[state])  # Max Q-value of next state
            self.q_table[state][action_index] += alpha * (reward + gamma * max_future_q - self.q_table[state][action_index])
        self.states.clear()

# Main game logic
def startGame():
    # Create two players: Human vs AI
    player1 = Player("Human", "X", 0)  # Human strategy, epsilon set to 0 since it's a human
    player2 = Player("AI", "O", 0.99)  # AI strategy, epsilon set to 0.99 for exploration
    
    players = [player1, player2]
    game = Game(players)

    # Play the game and keep updating the AI's Q-table
    while True:
        print("New game starts!")
        game.playGame()

        # Handle outcome and update Q-table
        if game.winner == 0:  # AI wins
            print("AI wins!")
            player1.updateQTable(-1)
            player2.updateQTable(1)
        elif game.winner == 1:  # Human wins
            print("Human wins!")
            player1.updateQTable(1)
            player2.updateQTable(-1)
        else:  # Tie
            print("It's a tie!")
            player1.updateQTable(0.5)
            player2.updateQTable(0.5)

        # Ask if the player wants to play another round
        game.reset()
        play_again = input("Do you want to play again? (yes/no): ").strip().lower()
        if play_again != "yes":
            print("Thank you for playing!")
            break


# Start the game
startGame()