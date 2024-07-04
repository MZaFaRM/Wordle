import contextlib
import random
import colorama
from colorama import Fore
from string import ascii_lowercase
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np

colorama.init(autoreset=True)


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
        self.wrong_guesses = set()
        self.wrong_index_guesses = set()

        self.words = ALL_WORDS
        if self.verbose:
            print(Fore.YELLOW + "Dictionary loaded")
        self.word = random.choice(self.words)

        if self.verbose:
            print(Fore.YELLOW + "Welcome to Wordle!")
            self.display()

    def update(self, string):
        self.chance_remaining -= 1
        for i in range(5):
            if string[i] == self.word[i]:
                self.correct_guesses.add(string[i])
                with contextlib.suppress(KeyError):
                    self.wrong_index_guesses.remove(string[i])

            elif string[i] in self.word:
                if string[i] not in self.correct_guesses:
                    self.wrong_index_guesses.add(string[i])
            else:
                self.wrong_guesses.add(string[i])

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


class WordleAI:
    def __init__(
        self, alpha=0.5, epsilon=0.3, epsilon_decay=0.99, min_epsilon=0.1
    ) -> None:
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def update(self, old_state, action, new_state, reward):
        """
        Updates the Q-learning model, given an old state, an action taken
        in that state, a new resulting state, and the reward received
        from taking that action.
        """
        old = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old, reward, best_future)

    def get_q_value(self, state, action):
        """
        Return the Q-value for the state `state` and the action `action`.
        If no Q-value exists yet in `self.q`, return 0.
        """
        return self.q.get((state, action), 0)

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        Update the Q-value for the state `state` and the action `action`
        given the previous Q-value `old_q`, a current reward `reward`,
        and an estimate of future rewards `future_rewards`.

        Use the formula:

        Q(s, a) <- old value estimate
                   + alpha * (new value estimate - old value estimate)

        where `old value estimate` is the previous Q-value,
        `alpha` is the learning rate, and `new value estimate`
        is the sum of the current reward and estimated future rewards.
        """
        self.q[(state, action)] = old_q + self.alpha * (
            (future_rewards + reward) - self.get_q_value(state, action)
        )

    def best_future_reward(self, state):
        """
        Given a state `state`, consider all possible `(state, action)`
        pairs available in that state and return the maximum of all
        of their Q-values.

        Use 0 as the Q-value if a `(state, action)` pair has no
        Q-value in `self.q`. If there are no available actions in
        `state`, return 0.
        """
        best_reward = 0
        for (curr_state, _), q_value in self.q.items():
            if curr_state == state:
                if q_value > best_reward:
                    best_reward = q_value
        return best_reward

    def choose_action(self, state, epsilon=True):
        """
        Given a state `state`, return an action `(i, j)` to take.

        If `epsilon` is `False`, then return the best action
        available in the state (the one with the highest Q-value,
        using 0 for pairs that have no Q-values).

        If `epsilon` is `True`, then with probability
        `self.epsilon` choose a random available action,
        otherwise choose the best action available.

        If multiple actions have the same Q-value, any of those
        options is an acceptable return value.
        """

        global ALL_WORDS

        best_action = ""
        best_value = -float("inf")
        for curr_state, curr_action in self.q.keys():
            if curr_state == state:
                if best_value < self.q[(curr_state, curr_action)]:
                    best_value = self.q[(curr_state, curr_action)]
                    best_action = curr_action

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

    def get_state(self, wordle):
        return (
            tuple(sorted(wordle.correct_guesses)),
            tuple(sorted(wordle.wrong_index_guesses)),
            tuple(sorted(wordle.wrong_guesses)),
        )

    def get_reward(self, old_state, new_state, terminal, win=False):
        reward = 0
        reward += (len(new_state[0]) - len(old_state[0])) * 50
        reward += (len(new_state[1]) - len(old_state[1])) * 5
        reward -= (len(new_state[2]) - len(old_state[2])) * 5
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

        while not wordle.terminal:
            old_state = ai.get_state(wordle)
            guess = ai.choose_action(old_state, epsilon=True)

            wordle.play(guess)
            f.write(guess + "\t")
            new_state = ai.get_state(wordle)
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
    print("Most used words: ")

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

    print(words)
    return ai


def main():
    ai = train_ai(1000)

    wordle = Wordle()

    while True:
        while not wordle.terminal:
            old_state = ai.get_state(wordle)
            guess = ai.choose_action(old_state)

            print("Guess:", guess)
            wordle.play(guess)
        if not wordle.win:
            break
        else:
            wordle = Wordle()


main()
