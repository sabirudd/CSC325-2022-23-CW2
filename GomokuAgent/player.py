#1907953

from misc import legalMove, winningTest
from gomokuAgent import GomokuAgent
import numpy as np

class Player(GomokuAgent):
    def __init__(self, ID, BOARD_SIZE, X_IN_A_LINE):
        self.ID = ID
        self.BOARD_SIZE = BOARD_SIZE
        self.X_IN_A_LINE = X_IN_A_LINE
        print(ID, BOARD_SIZE, X_IN_A_LINE)
        
        self.PLAYER = ID
        self.OPPONENT = self.getOpponent(self.PLAYER)
         
    
    def getOpponent(self, currentPlayer):
        """Checks the piece type for current player and returns the opposite type.

        Args:
            currentPlayer (int): Player piece type (1 or -1)

        Returns:
            int: Opponent piece type (1 or -1)
        """
        if currentPlayer == -1:
            return 1
        else:
            return -1


    def isFirst(self, board):
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                if board[i][j] != 0:
                    return False
        return True
    
    
    def terminalTest(self, board):
        """Check if there are anymore empty positions left on the board.

        Args:
            board (int[][]): 2D array of all positions on the board

        Returns:
            bool: True if there are empty positions, otherwise False
        """
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                if board[i][j] != 0:
                    return True
        return False


    def getAvailableMoves(self, board):
        """Gets all the empty positions remaining on the board.

        Args:
            board (int[][]): 2D array of all positions on the board

        Returns:
            int[tuple(int, int)]: Array containing tuples of all the empty 
                                  positions on the board
        """
        available_moves = []
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                if board[i][j] == 0:
                    available_moves.append((i, j))
        return available_moves


    def evaluate(self, board):
        """Evaluates the score of all legal lines in Gomoku based on the positions
        of the player and opponent.

        Args:
            board (int[][]): 2D array of all positions on the board

        Returns:
            int: Calculated score
        """
        if winningTest(self.PLAYER, board, self.X_IN_A_LINE):
            return np.inf
        
        if winningTest(self.OPPONENT, board, self.X_IN_A_LINE):
            return -np.inf
        
        playerScore = 0
        opponentScore = 0
        
        boardRotated = np.rot90(board)
        playerScore = self.scoreCheck(board, playerScore, self.PLAYER)
        playerScore = self.scoreCheck(boardRotated, playerScore, self.PLAYER)
        
        opponentScore = self.scoreCheck(board, opponentScore, self.OPPONENT)
        opponentScore = self.scoreCheck(boardRotated, opponentScore, self.OPPONENT)
                
        return playerScore - opponentScore
    
    
    def scoreCheck(self, board, score, currentPlayer):
        """Gets the score of the horizontal and diagonal (top-left to bottom-right)
        lines.

        Args:
            board (int[][]): 2D array of all positions on the board
            score (int): The current score
            currentPlayer (int): The current player piece type

        Returns:
            score: Cumulative score of horizontal and diagonal
        """
        #horizontal, vertical if 90 rotation
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE - self.X_IN_A_LINE + 1):
                line = board[row, col:col+5]
                score += self.lineScore(line, currentPlayer)
        
        #diagonal
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE - self.X_IN_A_LINE + 1):
                line = board[row:row+5, col:col:5]
                score += self.lineScore(line, currentPlayer)
        
        return score
    
    
    def lineScore(self, line, currentPlayer):
        """Calculate the score for a single line of 5 slots, 
        based on the positions of the pieces on the board.

        Args:
            line (int[]): A single line to evaluate
            currentPlayer (int): The current player piece type

        Returns:
            int: Score for a single line
        """
        score = 0
        
        empty = np.sum(line == 0)
        numPlayer = np.sum(line == currentPlayer)
        
        opponentID = self.getOpponent(currentPlayer)
        numOpponent = np.sum(line == opponentID)
        
        
        if numPlayer == 4:
            score += 1000
        elif numPlayer == 3 and empty == 2:
            score += 100
        elif numPlayer == 2 and empty == 3:
            score += 10
        elif numPlayer == 1 and empty == 4:
            score += 1
            
        if numOpponent == 4:
            score -= 1000
        elif numOpponent == 3 and empty == 2:
            score -= 100
        elif numOpponent == 2 and empty == 3:
            score -= 10
        elif numOpponent == 1 and empty == 4:
            score -= 1
            
        return score
        
    
    def minimax(self, board, depth, alpha, beta, isMaximising):
        """Implementation of the Minimax algorithm with Alpha-Beta pruning.

        Args:
            board (int[][]): 2D array of all positions on the board
            depth (int): The limit of recursion
            alpha (float): The minimum score
            beta (float): The maximum score
            isMaximising (bool): True if player, False if opponent

        Returns:
            int: Value of the set of moves
        """
        # Stop search and evaluate potential moves
        if not self.terminalTest(board) or depth == 0:
            return self.evaluate(board)
        
        available_moves = self.getAvailableMoves(board)

        # Player
        if isMaximising:
            value = -np.inf
            for move in available_moves:
                value = max(value, self.minimax(board, depth-1, alpha, beta, False))
                alpha = max(alpha, value)
                # Stop searching if worse 
                if value >= beta:
                    break
            return value
        
        # Opponent
        else:
            value = np.inf
            for move in available_moves:
                value = min(value, self.minimax(board, depth-1, alpha, beta, True))
                beta = min(beta, value)
                # Stop searching if worse
                if value <= alpha:
                    break
            return value
    

    def getBestMove(self, board):
        """Search through all possibilities of moves on the board to find
        the best move.

        Args:
            board (int[][]): 2D array of all positions on the board

        Returns:
            tuple(int, int): x and y position of the best found move
            int: Value of the best found move
        """
        bestValue = -1000
        bestMove = (-1, -1)
        
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                
                if (board[row][col] == 0):
                    # Get value for a move
                    board[row][col] = self.PLAYER
                    value = self.minimax(board, 3, np.inf, -np.inf, True)
                    
                    # Undo move
                    board[row][col] = 0
                    
                    # Update move if new best found
                    if value > bestValue:
                        bestMove = (row, col)
                        bestValue = value
        
        return bestMove, bestValue


    def move(self, board):
        """Send a request to place a piece on the board. Uses the Minimax
        algorithm with alpha-beta pruning to find the best move.
        Generates another move if the found move is not legal on the board.

        Args:
            board (int[][]): 2D array of all positions on the board

        Returns:
            tuple(int, int): x and y position of the best found move,
                             if the move is legal on the board
        """
        while True:
            print("looking for best move...")
            move, val = self.getBestMove(board)
            print("minimax ai (" + str(self.ID) + ") player called: " +  str(move) + \
                  " with value: " + str(val))
            if legalMove(board, move):
                return move
            else:
                print("invalid move " + str(move) + ", generating new move...")
