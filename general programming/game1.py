class Game:
    def __init__(self, player1:str = None, player2:str = None):
        self.player1 = player1
        self.player2 = player2
        self.scores = {self.player1:0, self.player2:0}
        self.winner = None
    
    def score(self, player, points: int):
        try:
            self.winner_exists()

        except:
            if self.player1 == player:
                self.scores[self.player1] += points
            elif self.player2 == player:
                self.scores[self.player2] += points
            else:
                raise Exception("Player doesnt belong to this game")
            
            if self.scores[self.player1] == self.scores[self.player2] and self.scores[self.player1] > 3:
                self.scores[self.player1] = 3
                self.scores[self.player2] = 3
                print(f"Scores: {self.scores}")
            
            # after adding points, check if theres a winner and game is over to display the winner
            try:
                self.winner_exists()
            except:
                return
                
        else:
            raise Exception("Cannot add score, game is over")
    
    def winner_exists(self):
        diff = abs(self.scores[self.player1] -  self.scores[self.player2])
        max_score = max(self.scores.values())
        if diff>=2 and max_score>=5:
            self.winner = max(self.scores, key = self.scores.get)
            print(f"Winner is {self.winner}")
            return True
        raise Exception("Game is not over")

    def __repr__(self) -> str:
        return f"{self.scores}"
    

class Match:
    def __init__(self, player1:str = None, player2:str = None) -> None:
        self.player1 = player1
        self.player2 = player2
        # comes from winning a Game
        self.points = {self.player1:0, self.player2:0}
        self.current_game = Game(self.player1, self.player2)
        self.game_count = 1
        self.match_winner = None
    
    def add_score(self, player: str, points: int):
        if self.match_winner:
            raise Exception("Match is over!")

        if player not in self.points:
            raise Exception("Player not in this match")

        self.current_game.score(player, points)
        # if this game has a winner, add points, check if match is over 
        if self.current_game.winner:
            self.points[self.current_game.winner] += 1
            print(f"Game {self.game_count} points: {self.points}")
            
            # check for winner of match
            if self.points[self.player1] >= 3 or self.points[self.player2] >= 3:
                self.match_winner = max(self.points, key=self.points.get)
                print(f"Winner of the match is {self.match_winner}")
                return

            # if match is not over, start new game
            self.game_count += 1
            print(f"\nStarting game {self.game_count}")
            self.current_game = Game(self.player1, self.player2)


    def __repr__(self) -> str:
        return f"Match Points: {self.points}, Match Winner: {self.match_winner}"

def main():
    match = Match("Player1", "Player2")

    assert match.points == {"Player1": 0, "Player2": 0}
    assert match.game_count == 1

    # Game 1
    match.add_score("Player1", 5)  # Player1 wins Game 1
    assert match.points == {"Player1": 1, "Player2": 0}
    assert match.game_count == 2

    # Game 2
    match.add_score("Player2", 5)  # Player2 wins Game 2
    assert match.points == {"Player1": 1, "Player2": 1}
    assert match.game_count == 3

    # Game 3
    match.add_score("Player1", 5)  # Player1 wins Game 3
    assert match.points == {"Player1": 2, "Player2": 1}
    assert match.game_count == 4

    # Game 4
    match.add_score("Player2", 5)  # Player2 wins Game 4
    assert match.points == {"Player1": 2, "Player2": 2}
    assert match.game_count == 5

    # Game 5
    match.add_score("Player1", 5)  # Player1 wins Game 5
    assert match.points == {"Player1": 3, "Player2": 2}
    assert match.match_winner == "Player1"

    try:
        match.add_score("Player1", 1)
    except Exception as e:
        assert str(e) == "Match is over!"

    print("Match test passed")


if __name__ == "__main__":
    main()

