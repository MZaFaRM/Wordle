import contextlib
import random
import re
import colorama
from colorama import Fore
from string import ascii_lowercase
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum

colorama.init(autoreset=True)


class Letter(Enum):
    BLACK = "wrong letter wrong index"
    YELLOW = "correct letter wrong index"
    GREEN = "correct letter correct index"


class InvalidWord(Exception):
    pass


with open("words.txt") as f:
    ALL_WORDS = list(map(lambda s: s.strip(), f.readlines()))


class Wordle:
    def __init__(self, verbose=True) -> None:

        self.terminal = False
        self.win = False
        self.chance_remaining = 6
        self.verbose = verbose

        self.words = ALL_WORDS
        if self.verbose:
            print(Fore.YELLOW + "Dictionary loaded")
        self.word = random.choice(self.words)
        # self.word = "abhor"

        if self.verbose:
            print(Fore.YELLOW + "Welcome to Wordle!")
            self.display()

    def result(self, string):
        letter_status = []
        for i in range(5):
            if string[i] == self.word[i]:
                letter_status.append(Letter.GREEN)
            elif string[i] in self.word:
                letter_status.append(Letter.YELLOW)
            else:
                letter_status.append(Letter.BLACK)

        return letter_status

    def display(self, string=""):
        if self.verbose:
            output = []

            if not string:
                output.append(Fore.MAGENTA + "_" * 5 + Fore.RESET)
            else:
                for i in range(5):
                    if string[i] == self.word[i]:
                        output.append(Fore.GREEN + string[i])
                    elif string[i] in self.word:
                        output.append(Fore.YELLOW + string[i])
                    else:
                        output.append(Fore.BLACK + string[i])
                output.append(Fore.RESET)
                output.append("\n")

                output.append(
                    Fore.YELLOW + f"{self.chance_remaining} guess(es) remaining"
                )

            output.append("\n")
            # for i in ascii_lowercase:
            #     if i in self.correct_guesses:
            #         output.append(Fore.GREEN + i)
            #     elif i in self.wrong_index_guesses:
            #         output.append(Fore.YELLOW + i)
            #     elif i in self.wrong_guesses:
            #         output.append(Fore.BLACK + i)
            #     else:
            #         output.append(Fore.WHITE + i)

            print("".join(output))

    def play(self, string):
        if self.terminal:
            return

        string = string.lower()

        if len(string) != 5:
            print(Fore.RED + "Word is not of length 5")
            return None
        elif string not in self.words:
            print(Fore.RED + "Word not found in dictionary")
            return None
        else:
            self.chance_remaining -= 1
            self.check_terminal(string)
            self.display(string)
            return self.result(string)

    def check_terminal(self, string):
        if string == self.word and self.terminal != True:
            self.terminal = True
            self.win = True
            print(Fore.GREEN + "Congrats you won!")
            return True
        elif self.chance_remaining == 0 and self.terminal != True:
            self.terminal = True
            self.win = False

            print(Fore.RED + "You've lost :/ the word was " + Fore.CYAN + self.word)
            return True
        return False


class WordleAI:
    def __init__(
        self, alpha=0.5, epsilon=0.3, epsilon_decay=0.99, min_epsilon=0.1
    ) -> None:
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.possible_words = ALL_WORDS

        # 0: correct guesses        GREEN
        # 1: wrong index guesses    YELLOW
        # 2: wrong guesses          BLACK
        self.game_state = (("", "", "", "", ""), (), ())

    def update(self, old_state, action, new_state, reward):
        old = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old, reward, best_future)
        return

    def get_q_value(self, state, action):
        return self.q.get((state, action), 0)

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        self.q[(state, action)] = old_q + self.alpha * (
            (future_rewards + reward) - self.get_q_value(state, action)
        )

    def best_future_reward(self, state):
        best_reward = 0
        for (curr_state, _), q_value in self.q.items():
            if curr_state == state:
                if q_value > best_reward:
                    best_reward = q_value
        return best_reward

    def choose_action(self, state, epsilon=True):

        global ALL_WORDS

        best_action = ""
        best_value = -float("inf")
        for curr_state, curr_action in self.q.keys():
            if curr_state == state:
                if best_value < self.q[(curr_state, curr_action)]:
                    best_value = self.q[(curr_state, curr_action)]
                    best_action = curr_action

        # random_action = random.choice(ALL_WORDS)
        random_action = random.choice(ALL_WORDS)

        if best_action == "":
            return random_action

        if epsilon:
            chosen_action = random.choices(
                [best_action, random_action], weights=[1 - self.epsilon, self.epsilon]
            )[0]
        else:
            chosen_action = best_action

        return chosen_action

    def update_game_state(self, action, result, wordle):
        green_states = list(self.game_state[0])

        for i in range(len(action)):
            if result[i] == Letter.GREEN:
                green_states[i] = action[i]
            elif result[i] == Letter.YELLOW:
                yellow_states.add(action[i])
            elif result[i] == Letter.BLACK:
                black_states.add(action[i])

        self.game_state = (
            tuple(green_states),
            tuple(sorted(yellow_states)),
            tuple(sorted(black_states)),
        )

        old_possible_words = self.possible_words
        wrong_letters = "".join(list(self.game_state[2]))
        regex = "".join(
            [
                rf"[^{wrong_letters}]" if letter == "" else letter
                for letter in list(self.game_state[0])
            ]
        )
        new_possible_words = [
            word for word in old_possible_words if re.match(regex, word)
        ]

        self.possible_words = new_possible_words

    def get_game_state(self):
        return self.possible_words

    def reset_game_state(self):
        self.game_state = (("", "", "", "", ""), (), ())
        self.possible_words = ALL_WORDS

    def get_reward(self, old_state, new_state, terminal, win=False):
        reward = 0
        # 0: correct guesses        GREEN
        # 1: wrong index guesses    YELLOW
        # 2: wrong guesses          BLACK

        # reward is calculated based on knowledge gained
        # new_green_states = new_state[0]
        # new_yellow_states = new_state[1]
        # new_black_states = new_state[2]

        # old_green_states = old_state[0]
        # old_yellow_states = old_state[1]
        # old_black_states = old_state[2]

        # # Set is used to not let green states cancel each other out with their '' values
        # reward += (len(set(new_green_states)) - len(set(old_green_states))) * 20
        # reward += (len(new_yellow_states) - len(old_yellow_states)) * 10
        # reward += (len(new_black_states) - len(old_black_states)) * 1

        reward = len(old_possible_words) - len(new_possible_words)

        self.possible_words = new_possible_words

        if terminal:
            if win:
                reward += 100
            else:
                reward -= 100
        return reward


class DQNGraph:
    def __init__(self) -> None:
        self.memory = []
        self.games_won = 0
        self.games_lost = 0

    def draw_graph(self, games_won, games_lost):

        self.memory.append((games_won, games_lost))
        self.games_won += games_won
        self.games_lost += games_lost

        N = len(self.memory)
        ind = np.arange(N)
        width = 0.25

        xvals = [item[0] for item in self.memory]
        bar1 = plt.bar(ind, xvals, width, color="g")

        yvals = [item[1] for item in self.memory]
        bar2 = plt.bar(ind + width, yvals, width, color="r")

        plt.xlabel("Batch Number")
        plt.ylabel("Scores")
        plt.title("Wordle Score")

        plt.xticks(ind + width, range(N))
        plt.legend(
            (bar1, bar2),
            (f"Games Won: {self.games_won}", f"Games Lost: {self.games_lost}"),
        )
        plt.savefig(f"DQN-Batch_{len(self.memory)}.png")
        print(f"DQN-Batch_{len(self.memory)}.png Saved!")


def train_ai(n):
    ai = WordleAI()
    graph = DQNGraph()

    games_won = 0
    games_lost = 0

    words = {}

    f = open("words_tried.txt", "w")

    for i in range(n):
        print("Playing game " + str(i))
        wordle = Wordle(verbose=False)
        ai.reset_game_state()

        while not wordle.terminal:
            old_state = ai.get_game_state()
            guess = ai.choose_action(old_state, epsilon=True)

            result = wordle.play(guess)
            f.write(guess + "\t")
            ai.update_game_state(guess, result, wordle.word)
            new_state = ai.get_game_state()
            reward = ai.get_reward(
                old_state,
                new_state,
                terminal=wordle.terminal,
                win=wordle.win,
            )
            f.write(f"Old: {str(old_state)} New: {str(new_state)} Reward: {reward}\n")

            words[(guess, wordle.chance_remaining)] = (
                words.get((guess, wordle.chance_remaining), 0) + 1
            )
            ai.update(old_state, guess, new_state, reward)

        if ai.epsilon > ai.min_epsilon:
            ai.epsilon *= ai.epsilon_decay

        if i % 100 == 0:
            graph.draw_graph(games_won, games_lost)
            games_won = 0
            games_lost = 0

        if wordle.win:
            games_won += 1
        else:
            games_lost += 1

        f.write("Game Ended.\n")
    f.close()

    with open("used_words.txt", "w") as t:
        words_ordered = sorted(words.keys(), key=lambda word: words[word], reverse=True)
        for i in range(5, -1, -1):
            t.write(f"{i}:\n")
            data = [
                word[0] for word in filter(lambda word: word[1] == i, words_ordered)
            ]

            h = ""
            for j in data:
                t.write(j + " : " + str(words[(j, i)]) + "\n")
    return ai


def main():
    ai = train_ai(10000)

    wordle = Wordle()

    while not wordle.terminal:
        old_state = ai.get_game_state()
        guess = ai.choose_action(old_state)

        print("Guess:", guess)
        # guess = input("Guess:")
        wordle.play(guess)


main()
