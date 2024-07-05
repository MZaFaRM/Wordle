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

        self.correct_guesses = set()
        self.wrong_index_guesses = set()
        self.wrong_guesses = set()

        self.words = ALL_WORDS
        if self.verbose:
            print(Fore.YELLOW + "Dictionary loaded")
        self.word = random.choice(self.words)

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
            for i in ascii_lowercase:
                if i in self.correct_guesses:
                    output.append(Fore.GREEN + i)
                elif i in self.wrong_index_guesses:
                    output.append(Fore.YELLOW + i)
                elif i in self.wrong_guesses:
                    output.append(Fore.BLACK + i)
                else:
                    output.append(Fore.WHITE + i)

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
    def __init__(self) -> None:
        self.sorted_words = self.sort_words(ALL_WORDS)
        self.possible_words = self.sorted_words
        self.game_state = [[list(ascii_lowercase) for _ in range(5)], []]

    def sort_words(self, words, known_letters=None):
        letter_distribution = self.get_letter_distribution(
            words, known_letters=known_letters
        )
        words = {word: 0 for word in words}
        for word in words:
            word_score = 0
            normalized_word = list(set(word))
            for letter in normalized_word:
                word_score += letter_distribution.index(letter) + 1
            words[word] = word_score

        return list(sorted(words.keys(), key=lambda word: words[word], reverse=True))

    def get_letter_distribution(self, words, known_letters=None):
        letter_distribution = {}
        if not known_letters:
            known_letters = []
        for word in words:
            for letter in word:
                if letter not in known_letters:
                    letter_distribution[letter] = letter_distribution.get(letter, 0) + 1
                else:
                    letter_distribution[letter] = float("inf")

        return sorted(
            letter_distribution.keys(),
            key=lambda letter: letter_distribution[letter],
        )

    def choose_action(self):
        if len(self.possible_words) >= 1:
            return self.possible_words[0]
        else:
            return -1

    def update_game_state(self, action, result):
        for i in range(5):
            if result[i] == Letter.GREEN:
                self.game_state[0][i] = [action[i]]
            elif result[i] == Letter.YELLOW:
                if action[i] not in self.game_state[1]:
                    self.game_state[1].append(action[i])
                with contextlib.suppress(ValueError):
                    self.game_state[0][i].remove(action[i])
            elif result[i] == Letter.BLACK:
                for j in range(5):
                    with contextlib.suppress(ValueError):
                        self.game_state[0][j].remove(action[i])

        regex = r""
        for possible_letters in self.game_state[0]:
            regex += "["
            for letter in possible_letters:
                regex += letter
            regex += "]"

        self.possible_words = self.sort_words(
            [word for word in self.possible_words if re.match(regex, word)],
            known_letters=self.game_state[1],
        )

    def reset_game_state(self):
        self.possible_words = self.sorted_words
        self.game_state = [[list(ascii_lowercase) for _ in range(5)], []]


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

    for i in range(1, n + 1):
        print("Playing game " + str(i))
        wordle = Wordle(verbose=False)
        ai.reset_game_state()

        while not wordle.terminal:
            guess = ai.choose_action()

            result = wordle.play(guess)
            ai.update_game_state(guess, result)

            words[(guess, wordle.chance_remaining)] = (
                words.get((guess, wordle.chance_remaining), 0) + 1
            )

        if i % 100 == 0:
            graph.draw_graph(games_won, games_lost)
            games_won = 0
            games_lost = 0

        if wordle.win:
            games_won += 1
        else:
            games_lost += 1

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


def wordle_input(string):
    print(
        "B : BLACK",
        "G : GREEN",
        "Y : YELLOW",
        "S : New Game",
        "R : Word not accepted",
        "eg: BGBYB",
        sep="\n",
    )
    response = input("Response: ").upper()
    formatted_response = []
    for letter in response:
        if letter == "B":
            formatted_response.append(Letter.BLACK)
        elif letter == "Y":
            formatted_response.append(Letter.YELLOW)
        elif letter == "G":
            formatted_response.append(Letter.GREEN)
        elif letter == "R":
            return -1
        elif letter == "S":
            return 1
        else:
            raise ValueError("Invalid Symbol")
    return formatted_response


def solve():
    ai = WordleAI()
    ai.reset_game_state()

    while True:
        ai.reset_game_state()
        while True:
            guess = ai.choose_action()

            if guess == -1:
                print(Fore.RED + "Can't find such a word from the dictionary :/")
                print(
                    Fore.GREEN
                    + "Perhaps updating my dictionary with this word would help"
                )
                break

            print("Guess:", guess)
            try:
                result = wordle_input(guess)
            except ValueError:
                return
            if result == -1:
                ai.possible_words.remove(guess)
                continue
            elif result == 1:
                print("Starting new game")
                break

            ai.update_game_state(guess, result)


def main():
    # ai = WordleAI()
    ai = train_ai(1000)

    wordle = Wordle()
    ai.reset_game_state()
    i = 1
    while True:
        while not wordle.terminal:
            guess = ai.choose_action()

            if guess == -1:
                print(Fore.RED + "Can't find such a word from the dictionary :/")
                print(
                    Fore.GREEN
                    + "Perhaps updating my dictionary with this word would help"
                )

            print("Guess:", guess)

            # guess = input("Guess:")
            result = wordle_input(guess)
            if result == -1:
                ai.possible_words.remove(guess)

            # result = wordle.play(guess)
            ai.update_game_state(guess, result)
        if not wordle.win:
            break
        else:
            wordle = Wordle()
            ai.reset_game_state()

        i -= 1

mode = input('Solve (s) or Train (t):').lower()
if mode == 's':
    solve()
else:
    main()
