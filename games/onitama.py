import datetime
import os

import numpy
import torch
from .onitama_ai import Game as WilliamGame, OnitamaAI, Point

from .abstract_game import AbstractGame

class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available

        ### Game
#        self.observation_shape = (8, 5, 5)
        self.observation_shape = (9, 5, 5)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(1250))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 8  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "expert"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class
        
        ### Self-Play
        self.num_workers = 4  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False #True if torch.cuda.is_available() else False
        self.max_moves = 1000  # Maximum number of moves if game is not finished before
        self.num_simulations = 200  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        #self.root_dirichlet_alpha = 0.3
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        
        # Residual Network
        self.downsample = "resnet"  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 6  # Number of blocks in the ResNet
        self.channels = 128  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        self.resnet_fc_reward_layers = [64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [64]  # Define the hidden layers in the policy head of the prediction network
        
        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = [64]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = [64]  # Define the hidden layers in the value network
        self.fc_policy_layers = [64]  # Define the hidden layers in the policy network



        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 100000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 256  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = True if torch.cuda.is_available() else False  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.003  # Initial learning rate
        self.lr_decay_rate = 0.9  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 100000

        ### Replay Buffer
        self.replay_buffer_size = 1000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 5  # Number of game moves to keep for every batch element
        self.td_steps = 5  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 1  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False

        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.
        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25

class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Onitama()

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward * 10, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.        

        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
    
    def accept_render(self):
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.

        Returns:
            An integer from the action space.
        """
        action = self.env.human_input_to_action()
        return action

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training

        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        return self.env.action_to_human_input(action_number)


class Card():
    def __init__(self, name, colour, deltas):
        self.name = name
        self.colour = colour
        self.deltas = deltas
CARDS = [
    Card("Tiger", 1, [(2, 0), (-1, 0)]),
    Card("Dragon", -1, [(1, -2), (1, 2), (-1, -1), (-1, 1)]),
    Card("Crab", 1, [(1, 0), (0, 2), (0, -2)]),
    Card("Elephant", -1, [(1, -1), (1, 1), (0, -1), (0, 1)]),
    Card("Monkey", 1, [(-1, -1), (1, -1), (-1, 1), (1, 1)]),
    Card("Mantis", -1, [(1, 1), (1, -1), (-1, 0)]),
    Card("Crane", 1, [(1 , 0), (-1, 1), (-1, -1)]),
    Card("Boar", -1, [(0, -1), (0, 1), (1, 0)]),
    # left-leaning
    Card("Frog", -1, [(-1, -1), (0, 2), (1, 1)]),
    Card("Goose", 1, [(1, 1), (0, -1), (0, 1), (-1, -1)]),
    Card("Horse", -1, [(0, 1), (-1, 0), (1, 0)]),
    Card("Eel", 1, [(-1, 1), (0, -1), (1, 1)]),
    # right-leaning
    Card("Rabbit", 1, [(-1, 1), (0, -2), (1, -1)]),
    Card("Rooster", -1, [(1, -1), (0, -1), (0, 1), (-1, 1)]),
    Card("Ox", 1, [(0, -1), (-1, 0), (1, 0)]),
    Card("Cobra", -1, [(-1, -1), (0, 1), (1, -1)]),
]

class Onitama:
    def __init__(self):
        self.board_size = 5
        self.board = numpy.zeros((self.board_size, self.board_size), dtype="int32")
        self.board_markers = [
            chr(x) for x in range(ord("A"), ord("A") + self.board_size)
        ]

        for i in range(self.board_size):
            self.board[0][i] = 1
            self.board[4][i] = -1
        self.board[0][2] = 2
        self.board[4][2] = -2

        drawnCards = numpy.random.choice(CARDS, 5, replace=False)
        
        self.p1Card1 = drawnCards[0]
        self.p1Card2 = drawnCards[1]

        self.p2Card1 = drawnCards[2]
        self.p2Card2 = drawnCards[3]

        self.midCard = drawnCards[4]

        self.player = self.midCard.colour

        self.minimax_ai = OnitamaAI(
            self.generate_william_game(),
            1
        )

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.board = numpy.zeros((self.board_size, self.board_size), dtype="int32")

        for i in range(self.board_size):
            self.board[0][i] = 1
            self.board[4][i] = -1
        self.board[0][2] = 2
        self.board[4][2] = -2

        drawnCards = numpy.random.choice(CARDS, 5, replace=False)
        
        self.p1Card1 = drawnCards[0]
        self.p1Card2 = drawnCards[1]

        self.p2Card1 = drawnCards[2]
        self.p2Card2 = drawnCards[3]

        self.midCard = drawnCards[4]

        self.player = self.midCard.colour

        return self.get_observation()

    def step(self, action):
        piece, move, card = decode_action(action)

        if self.player == 1:
            if card == 0:
                self.p1Card1, self.midCard = self.midCard, self.p1Card1
            elif card == 1:
                self.p1Card2, self.midCard = self.midCard, self.p1Card2
        if self.player == -1:
            if card == 0:
                self.p2Card1, self.midCard = self.midCard, self.p2Card1
            elif card == 1:
                self.p2Card2, self.midCard = self.midCard, self.p2Card2
        
        self.board[move[0]][move[1]] = self.board[piece[0]][piece[1]]
        self.board[piece[0]][piece[1]] = 0

        done = self.is_finished()

        reward = 1 if done else 0

        self.player *= -1

        return self.get_observation(), reward, done

    def get_observation(self):
        board_player1 = numpy.where(self.board == 1, 1.0, 0.0)
        board_player2 = numpy.where(self.board == -1, 1.0, 0.0)
        king_player1 = numpy.where(self.board == 2, 1.0, 0.0)
        king_player2 = numpy.where(self.board == -2, 1.0, 0.0)
        board_to_play = numpy.full((5, 5), self.player, dtype="int32")
        
        threat_board1 = numpy.zeros((5, 5), dtype="float")
        threat_next1 = numpy.zeros((5, 5), dtype="float")
        threat_board2 = numpy.zeros((5, 5), dtype="float")
        threat_next2 = numpy.zeros((5, 5), dtype="float")
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] > 0:
                    for move in self.p1Card1.deltas:
                        finalposition = (i + move[0], j + move[1])
                        if 0 <= finalposition[0] < 5 and 0 <= finalposition[1] < 5:
                            threat_board1[finalposition[0]][finalposition[1]] += 1
                    for move in self.p1Card2.deltas:
                        finalposition = (i + move[0], j + move[1])
                        if 0 <= finalposition[0] < 5 and 0 <= finalposition[1] < 5:
                            threat_board1[finalposition[0]][finalposition[1]] += 1
                    for move in self.midCard.deltas:
                        finalposition = (i + move[0], j + move[1])
                        if 0 <= finalposition[0] < 5 and 0 <= finalposition[1] < 5:
                            threat_next1[finalposition[0]][finalposition[1]] += 1
                if self.board[i][j] < 0:
                    for move in self.p2Card1.deltas:
                        finalposition = (i - move[0], j - move[1])
                        if 0 <= finalposition[0] < 5 and 0 <= finalposition[1] < 5:
                            threat_board2[finalposition[0]][finalposition[1]] += 1
                    for move in self.p2Card2.deltas:
                        finalposition = (i - move[0], j - move[1])
                        if 0 <= finalposition[0] < 5 and 0 <= finalposition[1] < 5:
                            threat_board2[finalposition[0]][finalposition[1]] += 1
                    for move in self.midCard.deltas:
                        finalposition = (i - move[0], j - move[1])
                        if 0 <= finalposition[0] < 5 and 0 <= finalposition[1] < 5:
                            threat_next2[finalposition[0]][finalposition[1]] += 1
        # print("Threat board 1:")
        # print("Cards:", self.p1Card1.name, self.p1Card2.name)
        # print(threat_board1)
        # print("Threat board 2:")
        # print("Cards:", self.p2Card1.name, self.p2Card2.name)
        # print(threat_board2)
        to_return = [board_player1, board_player2, king_player1, king_player2, board_to_play, threat_board1, threat_next1, threat_board1, threat_board2]
        return numpy.array(to_return)

    def legal_actions(self):
        legal = []
        directionMod = self.player
        pawns = []
        cards = []
        if self.player == 1:
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if self.board[i][j] > 0:
                        pawns.append((i, j))
            cards = [self.p1Card1, self.p1Card2]
        else:
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if self.board[i][j] < 0:
                        pawns.append((i, j))
            cards = [self.p2Card1, self.p2Card2]

        for i in range(len(pawns)):
            for j in range(len(cards)):
                for move in cards[j].deltas:
                    finalposition = (pawns[i][0] + move[0]*directionMod, pawns[i][1] + move[1]*directionMod)
                    if 0 <= finalposition[0] < 5 and 0 <= finalposition[1] < 5:
                        canMoveToPosition = True
                        if self.board[finalposition[0]][finalposition[1]] > 0 and self.player == 1:
                            canMoveToPosition = False
                        elif self.board[finalposition[0]][finalposition[1]] < 0 and self.player == -1:
                            canMoveToPosition = False
                        if canMoveToPosition:
                            legal.append(encode_action(pawns[i], finalposition, j))

        return legal

    def is_finished(self):
        # Way of the Stream
        if self.board[0][2] == -2 or self.board[4][2] == 2:
            return True
        
        # Way of the Stone
        master_captured_neg = True
        master_captured_pos = True
        for i in range (self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == -2:
                    master_captured_neg = False
                elif self.board[i][j] == 2:
                    master_captured_pos = False

        return master_captured_neg or master_captured_pos

    def render(self):
        printTwoCards(self.p1Card1, self.p1Card2)
        print()
        print("Middle Card:", self.midCard.name)
        array = [["."]*5 for _ in range(5)]
        array[2][2] = 'O'
        for delta in self.midCard.deltas:
            array[2+delta[0]][2+delta[1]] = "X"
        marker = "  "
        for i in range(self.board_size):
            marker = marker + self.board_markers[i] + " "
        print(marker)
        for row in range(self.board_size):
            print(chr(ord("A") + row), end=" ")
            for col in range(self.board_size):
                ch = self.board[row][col]
                if ch == 0:
                    print(".", end=" ")
                elif ch == 1:
                    print("r", end=" ")
                elif ch == 2:
                    print("R", end=" ")
                elif ch == -1:
                    print("b", end=" ")
                elif ch == -2:
                    print("B", end=" ")
            print(end = " ")
            if self.player == -1:
                for col in range(self.board_size):
                    print(array[5-row-1][5-col-1], end=" ")
            else:
                for col in range(self.board_size):
                    print(array[row][col], end=" ")
            print()
                  
        printTwoCards(self.p2Card1, self.p2Card2, True)

    def human_input_to_action(self):
        human_input = input("Choose a card [0, 1]: ")
        while human_input != "0" and human_input != "1":
            human_input = input("Sorry, that's not valid. Choose a card [0, 1]: ")
        impactCard = self.p2Card1 if human_input == "0" else self.p2Card2
        square = input("Choose a square by typing in first the row, then the column of the piece you'd like to move. For example, type 'BC' if you want to move in the 2nd row and the 3rd column.")

        while not self.validateSquare(square, "piece"):
            print("Sorry, that's not valid.")
            square = input("Choose a square by typing in first the row, then the column of the piece you'd like to move.")
        x = ord(square[0]) - 65
        y = ord(square[1]) - 65
        valid_destination = False
        des_x = 0
        des_y = 0
        while not valid_destination:
            destination = input("Choose a square that is a legal move according to the card you just picked, using the same format as above.")
            while not self.validateSquare(destination, "destination"):
                destination = input("Sorry, that's not valid. Choose a square that is a legal move by typing in first the row, then the column of the destination.")
            des_x = ord(destination[0]) - 65
            des_y = ord(destination[1]) - 65
            for delta in impactCard.deltas:
                if des_x == x - delta[0] and des_y == y - delta[1] and self.board[des_x][des_y] >= 0:
                    valid_destination = True

        print("Made Action:", (x, y), (des_x, des_y), ord(human_input) - ord("0"))
        print("Encoded:", encode_action((x, y), (des_x, des_y), ord(human_input) - ord("0")))

        return encode_action((x, y), (des_x, des_y), ord(human_input) - ord("0"))

    def validateSquare(self, square, target):
        if len(square) != 2:
            return False
        x = ord(square[0]) - 65
        y = ord(square[1]) - 65
        if not(0 <= x < 5 and 0 <= y < 5):
            return False
        if target == "piece":
            return self.board[x][y] < 0
        elif target == "destination":
            self.board[x][y] >= 0
        return True

    def action_to_human_input(self, action):
        return decode_action(action)

    def generate_william_game(self):
        will_board = numpy.zeros((self.board_size, self.board_size), dtype="int32")
        for i in range(self.board_size):
            for j in range(self.board_size):
                will_board[i][j] = - self.board[i][j]
        return WilliamGame(
            red_cards = [self.p1Card1.name.lower(), self.p1Card2.name.lower()],
            blue_cards = [self.p2Card1.name.lower(), self.p2Card2.name.lower()],
            neutral_card = self.midCard.name.lower(),
            board = will_board,
            starting_player = -self.player
        )
    def expert_action(self):
        self.minimax_ai.game = self.generate_william_game()
        ai_move = self.minimax_ai.decide_move()
        blue_cards = [self.p2Card1.name.lower(), self.p2Card2.name.lower()]
        
        card = blue_cards.index(ai_move.card)
        print((ai_move.start.y, ai_move.start.x), (ai_move.end.y, ai_move.end.x), ai_move.card)
        return encode_action((ai_move.start.y, ai_move.start.x), (ai_move.end.y, ai_move.end.x), card)

def printTwoCards(card1, card2, reverse = False):
    print(card1.name, card2.name)
    array = [["."]*5 for _ in range(5)]
    array2 = [["."]*5 for _ in range(5)]
    array[2][2] = 'O'
    array2[2][2] = 'O'
    for delta in card1.deltas:
        array[2+delta[0]][2+delta[1]] = "X"
    for delta in card2.deltas:
        array2[2+delta[0]][2+delta[1]] = "X"

    if reverse:
        for i in range(5):
            for j in range(5):
                print(array[5-i-1][5-j-1], end=" ")
            print(end = " ")
            for j in range(5):
                print(array2[5-i-1][5-j-1], end=" ")
            print()
    else:
        for i in range(5):
            for j in range(5):
                print(array[i][j], end=" ")
            print(end = " ")
            for j in range(5):
                print(array2[i][j], end=" ")
            print()

def encode_action(piece, destination, card):
    pieceCode = space_to_number(piece[0], piece[1])
    destinationCode = space_to_number(destination[0], destination[1])

    moveCode = pieceCode* 25 + destinationCode

    return int(moveCode*2 + card)

def decode_action(number):
    card = int(number) % 2
    moveCode = number // 2

    pieceCode = moveCode // 25
    destinationCode = moveCode % 25

    return (number_to_space(pieceCode), number_to_space(destinationCode), card)

def space_to_number(row, col):
    return row*5 + col

def number_to_space(rawNum):
    num = int(rawNum)
    return (num//5, num % 5)
