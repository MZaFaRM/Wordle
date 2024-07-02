import random
import colorama
from colorama import Fore
from string import ascii_lowercase


class InvalidWord(Exception):
    pass


class Wordle:
    def __init__(self, verbose=True) -> None:

        self.terminal = False
        self.win = False
        self.chance_remaining = 6
        self.verbose = verbose

        if verbose:
            self.guessed_correct = set()
            self.guessed_wrong = set()
            self.guessed_wrong_index = set()

        colorama.init(autoreset=True)

        with open("words.txt", "r") as f:
            self.words = list(map(lambda s: s.strip(), f.readlines()))
            if self.verbose:
                print(Fore.YELLOW + "Dictionary loaded")
            self.word = random.choice(self.words)
            self.word = "allow"

        if self.verbose:
            print(Fore.YELLOW + "Welcome to Wordle!")
            self.display()

    def update(self, string):
        self.chance_remaining -= 1
        if self.verbose:
            for i in range(5):
                if string[i] == self.word[i]:
                    self.guessed_correct.add(string[i])
                elif string[i] in self.word:
                    self.guessed_wrong_index.add(string[i])
                else:
                    self.guessed_wrong.add(string[i])

    def display(self, string="", terminal=False):
        if not string:
            print(Fore.MAGENTA + "_" * 5)
        else:
            for i in range(5):
                if string[i] == self.word[i]:
                    print(Fore.GREEN + string[i], end="")
                elif string[i] in self.word:
                    print(Fore.YELLOW + string[i], end="")
                else:
                    print(Fore.BLACK + string[i], end="")
            print()

        if self.verbose:
            print(Fore.YELLOW + f"{self.chance_remaining} guess(es) remaining")

            for i in ascii_lowercase:
                if i in self.guessed_correct:
                    print(Fore.GREEN + i, end="")
                elif i in self.guessed_wrong_index:
                    print(Fore.YELLOW + i, end="")
                elif i in self.guessed_wrong:
                    print(Fore.BLACK + i, end="")
                else:
                    print(Fore.WHITE + i, end="")
            print()

    def play(self, string):
        if self.terminal:
            return

        string = string.lower()

        if len(string) != 5:
            print(Fore.RED + "Word is not of length 5")
        elif string not in self.words:
            print(Fore.RED + "Word not found in dictionary")
        else:
            self.update(string)
            self.check_terminal(string)
            self.display(string)

    def check_terminal(self, string):
        if self.chance_remaining == 0 and self.terminal != True:
            self.terminal = True
            self.win = False
            print(Fore.RED + "You've lost :/ the word was " + Fore.CYAN + self.word)
            return True
        elif string == self.word and self.terminal != True:
            self.terminal = True
            self.win = True
            print(Fore.GREEN + "Congrats you won!")
            return True
        return False


def main():
    wordle = Wordle()
    while not wordle.terminal:
        guess = input("Guess: ")
        wordle.play(guess)


main()
