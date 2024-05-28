import random

class Player:
    def __init__(self, color):
        self.color = color
        self.pieces = [(0, 0), (0, 0), (0, 0), (0, 0)]  # Piece positions on the board

class Ludo:
    def __init__(self):
        self.players = {
            'red': Player('red'),
            'blue': Player('blue'),
            'green': Player('green'),
            'yellow': Player('yellow')
        }
        self.board = self.initialize_board()
        self.turn_counter = 0

    def initialize_board(self):
        # Initialize a 15x15 board with routes and safe line to home
        board = [['-' for _ in range(15)] for _ in range(15)]

        # Define routes
        for i in range(6):
            board[i][6] = 'R'  # Red route
            board[8][i] = 'B'  # Blue route
            board[14 - i][8] = 'G'  # Green route
            board[6][14 - i] = 'Y'  # Yellow route

        # Define safe line to home
        for i in range(6, 9):
            board[6][i] = '*'  # Safe line to home (red)
            board[i][8] = '*'  # Safe line to home (blue)
            board[8][14 - i] = '*'  # Safe line to home (green)
            board[14 - i][6] = '*'  # Safe line to home (yellow)

        return board

    def display_board(self):
        print("Current State of the Ludo Board:")
        for row in self.board:
            print(' '.join(row))
        print()

    def update_board(self):
        # Clear the board
        self.board = self.initialize_board()
        # Place players' pieces on the board
        for player in self.players.values():
            for piece in player.pieces:
                x, y = piece
                if 0 <= x < 15 and 0 <= y < 15:
                    self.board[x][y] = player.color[0].upper()

    def play(self):
        while not self.game_over() and self.turn_counter < 50:
            self.update_board()
            self.display_board()

            current_player = list(self.players.values())[self.turn_counter % 4]
            print(f"It's {current_player.color}'s turn.")

            # Simulate rolling a die
            roll = random.randint(1, 6)
            print(f"{current_player.color} rolled a {roll}.")

            # Implement player's move logic here

            self.turn_counter += 1

    def game_over(self):
        # Check if any player has all pieces at home
        for player in self.players.values():
            if all(piece == (7, 7) for piece in player.pieces):  # (7, 7) is the center of the board
                print(f"{player.color} wins!")
                return True
        return False

if __name__ == "__main__":
    ludo_game = Ludo()
    ludo_game.play()
