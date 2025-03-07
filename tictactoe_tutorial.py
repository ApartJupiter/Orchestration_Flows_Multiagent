import random
import numpy as np

#Game class
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
                print(self.board[x*3+y], end="")
                print("|", end="")
            print("\n---------")
    def playGame(self):
        while(True):
            move = self.players[0].makeMove(self.board)
            self.board[move] = "X"
            self.checkForWin()
            if(self.gameOver):
                break
            move = self.players[1].makeMove(self.board)
            self.board[move] = "O"
            self.checkForWin()
            if(self.gameOver):
                break
    def checkForWin(self):
        win_conditions = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        for (i, j, k) in win_conditions:
            if(self.board[i] != " " and self.board[i] == self.board[j] == self.board[k]):
                if(self.board[i] == "X"):
                    self.winner = 0
                    self.gameOver = True
                else:
                    self.winner = 1
                    self.gameOver = True
        if " " not in self.board:
            self.gameOver = True
            self.winner = 2
        return

    def reset(self):
        self.board = [" " for x in range(9)]
        self.gameOver = False
        self.winner = ""

class Player:
    def __init__(self, name, strategy, epsilon):
        self.name = name
        self.strategy = strategy
        self.epsilon = epsilon
        self.q_table = {}
        self.states = []
    def makeMove(self, board):
        validMoves = [] #board = [" ", "X", "O", "X", "O"...]
        for x in range(len(board)):
            if(board[x] == " "):
                validMoves.append(x)
        if(self.strategy == "human"):
            print("Your valid moves are:", validMoves)
            print("Where would you like to move?")
            move = int(input())
            return move
        elif(self.strategy == "random"):
            return random.choice(validMoves)
        elif(self.strategy == "AI"):
            state = tuple(board)
            move_index = 0
            if(state not in self.q_table):
                self.q_table[state] = [0 for x in range(len(validMoves))]
            if(random.random() < self.epsilon):
                move_index = random.randint(0, len(validMoves)-1)
                move = validMoves[move_index]
            else:
                move_index = np.argmax(self.q_table[state])
                move = validMoves[move_index]
            self.states.append((state, move_index))
            return move
    '''
    def updateQTable(self, reward):
        for(state, action_index) in self.states:
            self.q_table[state][action_index] += reward
        self.states.clear()
    '''
    #New Version
    def updateQTable(self, reward):
        alpha = 0.1  # Learning rate
        gamma = 0.9  # Discount factor
        for (state, action_index) in self.states:
            max_future_q = max(self.q_table[state])  # Max Q-value of next state
            self.q_table[state][action_index] += alpha * (reward + gamma * max_future_q - self.q_table[state][action_index])
        self.states.clear()

player1 = Player("Player1", "AI", 0.99)
player2 = Player("Player2", "AI", 0.99)
players = [player1, player2]
game = Game(players)

for x in range(1000000):
    game.playGame()
    if(game.winner == 0):
        if(player1.strategy == "AI"):
            player1.updateQTable(1)
        if(player2.strategy == "AI"):
            player2.updateQTable(-1)
        #print("Player 1 wins!")
    elif(game.winner == 1):
        if(player1.strategy == "AI"):
            player1.updateQTable(-1)
        if(player2.strategy == "AI"):
            player2.updateQTable(1)
        #print("Player 2 wins!")
    else:
        if(player1.strategy == "AI"):
            player1.updateQTable(0.5)
        if(player2.strategy == "AI"):
            player2.updateQTable(0.5)
        #print("It's a tie!")
    game.reset()

player1.epsilon = 0
game2 = Game(players)
wins = 0
ties = 0
losses = 0
for x in range(100000):
    game2.playGame()
    if(game2.winner == 0):
        wins += 1
    elif(game2.winner == 1):
        losses += 1
    else:
        ties += 1
    game2.reset()
print("Wins:", wins, "Losses:", losses, "Ties:", ties)

print("\n\n")
#for state, q_values in player1.q_table.items():
    #print(state, q_values)
#Wins: 9796 Losses: 0 Ties: 204