"""Onitama game engine. Note that (x, y) goes right and down. Blue uses positive numbers, red uses negative numbers."""

import numpy as np
import random
from typing import List, NamedTuple, Optional

BOARD_WIDTH = 5
BOARD_HEIGHT = 5

class Point(NamedTuple):
    x: int
    y: int

    def to_algebraic_notation(self):
        return chr(ord("a") + self.x) + str(5 - self.y)
    
    @classmethod
    def from_algebraic_notation(cls, location):
        x = ord(location[0]) - ord("a")
        y = 5 - int(location[1])
        return cls(x, y)

class Card:
    def __init__(self, name, starting_player, *moves: List[Point]):
        """Moves are represented as a list of Points representing movement relative to (0, 0)"""
        self.name = name
        self.starting_player = starting_player
        self.moves = moves
    
    def visualize(self):
        # Cards are displayed with board centre (2, 2) at (0, 0)
        output = [["."] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        output[BOARD_HEIGHT // 2][BOARD_WIDTH // 2] = "O"
        for move in self.moves:
            output[move.y + BOARD_HEIGHT // 2][move.x + BOARD_WIDTH // 2] = "X"
        return "\n".join("".join(row) for row in output)

    def __repr__(self):
        return f"Card({self.name!r},\n{self.visualize()})"

class Move(NamedTuple):
    start: Point
    end: Point
    card: str

    def __str__(self):
        return f"{self.card} {self.start.to_algebraic_notation()} {self.end.to_algebraic_notation()}"

class Game:
    def __init__(self, *, red_cards: Optional[List[str]] = None, blue_cards: Optional[List[str]] = None,
                 neutral_card: Optional[str] = None, board=None, starting_player=None):
        """Represents an Onitama game. Generates random cards to fill in missing red_cards, blue_cards,
        or neutral_card. If starting_player is not specified, uses neutral_card.starting_player."""
        cards = set(ONITAMA_CARDS)
        if red_cards:
            cards -= set(red_cards)
            red_cards = [ONITAMA_CARDS[card] for card in red_cards]
        if blue_cards:
            cards -= set(blue_cards)
            blue_cards = [ONITAMA_CARDS[card] for card in blue_cards]
        if neutral_card:
            cards.remove(neutral_card)
            neutral_card = ONITAMA_CARDS[neutral_card]
        if not red_cards:
            card1, card2 = random.sample(cards, k=2)
            red_cards = [ONITAMA_CARDS.get(card1), ONITAMA_CARDS.get(card2)]
            cards -= {card1, card2}
        if not blue_cards:
            card1, card2 = random.sample(cards, k=2)
            blue_cards = [ONITAMA_CARDS.get(card1), ONITAMA_CARDS.get(card2)]
            cards -= {card1, card2}
        if not neutral_card:
            card = random.sample(cards, k=1)[0]
            neutral_card = ONITAMA_CARDS.get(card)
            cards.remove(card)
        if not starting_player:
            starting_player = neutral_card.starting_player
        if board is None:
            board = np.array([
                [-1, -1, -2, -1, -1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 2, 1, 1]])
        
        self.board = board
        self.red_cards = red_cards
        self.blue_cards = blue_cards
        self.neutral_card = neutral_card
        self.current_player = starting_player
    
    def visualize_piece(self, piece):
        piece_mapping = {-2: "R", -1: "r", 0: ".", 1: "b", 2: "B"}
        return piece_mapping.get(piece)
    
    def visualize_board(self):
        return "\n".join("".join(self.visualize_piece(self.board[y][x]) for x in range(BOARD_WIDTH)) for y in range(BOARD_HEIGHT))
    
    def visualize(self):
        fancy_board = [f"{5 - i} {line}" for i, line in enumerate(self.visualize_board().split("\n"))]
        fancy_board.append("  abcde")
        fancy_board_str = "\n".join(fancy_board)
        return (f"{fancy_board_str}\n"
                f"current_player: {'blue' if self.current_player > 0 else 'red'}\n"
                f"red_cards: {' '.join(card.name for card in self.red_cards)}\n" +
                "\n".join(f"{c1}\t{c2}" for c1, c2 in zip(self.red_cards[0].visualize().split("\n"), self.red_cards[1].visualize().split("\n"))) + "\n" +
                f"blue_cards: {' '.join(card.name for card in self.blue_cards)}\n" +
                "\n".join(f"{c1}\t{c2}" for c1, c2 in zip(self.blue_cards[0].visualize().split("\n"), self.blue_cards[1].visualize().split("\n"))) + "\n" +
                f"neutral_card: {self.neutral_card.name}\n"
                f"{self.neutral_card.visualize()}")
    
    def copy(self):
        return Game(red_cards=[card.name for card in self.red_cards], blue_cards=[card.name for card in self.blue_cards],
                    neutral_card=self.neutral_card.name, board=self.board.copy(), starting_player=self.current_player)
    
    def __repr__(self):
        return f"Game(\n{self.visualize()}\n)"
    
    def serialize(self):
        sorted_red = "_".join(sorted(card.name for card in self.red_cards))
        sorted_blue = "_".join(sorted(card.name for card in self.blue_cards))
        return f"{self.visualize_board()}\n{self.current_player}\n{sorted_red}\n{sorted_blue}\n{self.neutral_card.name}"
    
    def legal_moves(self):
        moves = []
        cards = self.red_cards if self.current_player == -1 else self.blue_cards
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if self.board[y][x] * self.current_player > 0:
                    direction_mod = self.current_player
                    for card in cards:
                        for card_move in card.moves:
                            new_x = x + direction_mod * card_move.x
                            new_y = y + direction_mod * card_move.y
                            if new_x in range(BOARD_WIDTH) and new_y in range(BOARD_HEIGHT) and self.board[new_y][new_x] * self.current_player <= 0:
                                moves.append(Move(Point(x, y), Point(new_x, new_y), card.name))
        if not moves:
            # pass due to no piece moves, but have to swap a card
            return [Move(Point(0, 0), Point(0, 0), card) for card in cards]
        return moves
    
    def apply_move(self, move: Move):
        cards = self.red_cards if self.current_player == -1 else self.blue_cards
        card_idx = 0 if cards[0].name == move.card else 1
        piece = self.board[move.start.y][move.start.x]
        self.board[move.start.y][move.start.x] = 0
        self.board[move.end.y][move.end.x] = piece
        self.neutral_card, cards[card_idx] = cards[card_idx], self.neutral_card
        self.current_player *= -1
    
    def determine_winner(self):
        # Way of the Stone (capture opponent master)
        red_alive = False
        blue_alive = False
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if self.board[y][x] == -2:
                    red_alive = True
                if self.board[y][x] == 2:
                    blue_alive = True
        if not red_alive:
            return 1
        if not blue_alive:
            return -1
        # Way of the Stream (move master to opposite square)
        if self.board[0][BOARD_WIDTH // 2] == 2:
            return 1
        if self.board[-1][BOARD_WIDTH // 2] == -2:
            return -1
        return 0
    
    def evaluate(self):
        """Evaluates a given board position. Very arbitrary.
        Assigns a win to +/-50.
        Each piece is worth 2, king is worth 4.
        Shortest distance (diagonals have length 1) from king to temple is subtracted."""
        winner = self.determine_winner()
        if winner:
            return winner * 50
        evaluation = 0
        blue_king_pos = Point(0, 0)
        red_king_pos = Point(0, 0)
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                # Piece evaluation
                piece = self.board[y][x]
                evaluation += piece * 2
                if piece == 2:
                    blue_king_pos = Point(x, y)
                elif piece == -2:
                    red_king_pos = Point(x, y)
        evaluation -= max(abs(blue_king_pos.x - 2), abs(blue_king_pos.y - 4))
        evaluation += max(abs(red_king_pos.x - 2), abs(red_king_pos.y - 0))
        return evaluation




ONITAMA_CARDS = {
    # symmetrical
    "tiger": Card("tiger", 1, Point(0, -2), Point(0, 1)),
    "dragon": Card("dragon", -1, Point(-2, -1), Point(2, -1), Point(-1, 1), Point(1, 1)),
    "crab": Card("crab", 1, Point(0, -1), Point(-2, 0), Point(2, 0)),
    "elephant": Card("elephant", -1, Point(-1, -1), Point(1, -1), Point(-1, 0), Point(1, 0)),
    "monkey": Card("monkey", 1, Point(-1, -1), Point(1, -1), Point(-1, 1), Point(1, 1)),
    "mantis": Card("mantis", -1, Point(-1, -1), Point(1, -1), Point(0, 1)),
    "crane": Card("crane", 1, Point(0, -1), Point(-1, 1), Point(1, 1)),
    "boar": Card("boar", -1, Point(0, -1), Point(-1, 0), Point(1, 0)),
    # left-leaning
    "frog": Card("frog", -1, Point(-1, -1), Point(-2, 0), Point(1, 1)),
    "goose": Card("goose", 1, Point(-1, -1), Point(-1, 0), Point(1, 0), Point(1, 1)),
    "horse": Card("horse", -1, Point(0, -1), Point(-1, 0), Point(0, 1)),
    "eel": Card("eel", 1, Point(-1, -1), Point(1, 0), Point(-1, 1)),
    # right-leaning
    "rabbit": Card("rabbit", 1, Point(1, -1), Point(2, 0), Point(-1, 1)),
    "rooster": Card("rooster", -1, Point(1, -1), Point(-1, 0), Point(1, 0), Point(-1, 1)),
    "ox": Card("ox", 1, Point(0, -1), Point(1, 0), Point(0, 1)),
    "cobra": Card("cobra", -1, Point(1, -1), Point(-1, 0), Point(1, 1)),
}