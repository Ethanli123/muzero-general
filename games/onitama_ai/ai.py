from . import Game
import random

INF = 1000

class OnitamaAI:
    MAX_SEARCH_DEPTH = 4

    def __init__(self, game, ai_player=-1):
        self.game = game
        self.ai_player = ai_player
        self.state_cache = {}
    
    def minimax(self, game: Game, depth, alpha, beta):
        cached = self.state_cache.get((depth, game.serialize()))
        if cached:
            return cached
        if depth > self.MAX_SEARCH_DEPTH or game.determine_winner():
            evaluation = game.evaluate()
            self.state_cache[depth, game.serialize()] = evaluation
            return evaluation
        if game.current_player > 0:
            best_score = -INF
            for move in game.legal_moves():
                new_game = game.copy()
                new_game.apply_move(move)

                game_score = self.minimax(new_game, depth + 1, alpha, beta)
                self.state_cache[depth + 1, new_game.serialize()] = game_score

                best_score = max(best_score, game_score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_score
        else:
            best_score = INF
            for move in game.legal_moves():
                new_game = game.copy()
                new_game.apply_move(move)

                game_score = self.minimax(new_game, depth + 1, alpha, beta)
                self.state_cache[depth + 1, new_game.serialize()] = game_score

                best_score = min(best_score, game_score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            return best_score
    
    def decide_move(self):
        assert self.game.current_player == self.ai_player
        best_move = []
        best_score = -INF * self.ai_player
        for move in self.game.legal_moves():
            new_game = self.game.copy()
            new_game.apply_move(move)

            game_score = self.minimax(new_game, 1, -INF, INF)
            self.state_cache[1, new_game.serialize()] = game_score

            if self.ai_player * game_score > self.ai_player * best_score:
                best_score = game_score
                best_move = [move]
            elif game_score == best_score:
                best_move.append(move)
        ai_move = random.choice(best_move)
        self.game.apply_move(ai_move)
        return ai_move
