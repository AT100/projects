from collections import deque
class Player:
    def __init__(self, id = None) -> None:
        self.id = id
        self.score = 0

class Game:
    def __init__(self, player1: Player, player2: Player):
        self.player1 = player1
        self.player2 = player2
    
    def get_score(self):
        print(f"Player A: {self.player1.score}, Player B: {self.player2.score}")
    
    def add_point(self, player: str):
        try:
            self.winner()
        except:
            if player == self.player1.id:
                self.player1.score += 1
            elif player == self.player2.id:
                self.player2.score += 1
            
            # reset the player scores back to 3 if it's a tie and the score is greater than 3
            if self.player1.score == self.player2.score and self.player1.score > 3:
                self.player1.score = 3
                self.player2.score = 3

    def winner(self):
        if self.player1.score > self.player2.score:
            advantage = self.player1
        else:
            advantage = self.player2
        diff = abs(self.player1.score - self.player2.score)
        if diff >= 2 and advantage.score>=5:
            print(f"The winner is {advantage.id}")
        else:
            #print("winner?")
            raise ValueError("No winner found yet!")

def main():
    a = Player('A')
    b = Player('B')
    game = Game(a, b)
    game.get_score()
    game.add_point('A')
    game.add_point('A')
    game.add_point('B')
    game.add_point('B')
    game.get_score()
    try:
        game.winner()
    except ValueError as e:
        print(e)
    game.add_point('A')
    game.get_score()

    game.add_point('B')
    game.get_score()

    game.add_point('B')
    game.get_score()

    game.add_point('B')
    game.get_score()

    try:
        game.winner()
    except ValueError as e:
        print(e)
if __name__ == "__main__":
    main()