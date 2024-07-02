import contextlib
import random
import colorama
from colorama import Fore
from string import ascii_lowercase
from functools import lru_cache

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

        self.guessed_correct = set()
        self.guessed_wrong = set()
        self.guessed_wrong_index = set()

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
                self.guessed_correct.add(string[i])
                with contextlib.suppress(KeyError):
                    self.guessed_wrong_index.remove(string[i])

            elif string[i] in self.word:
                if string[i] not in self.guessed_correct:
                    self.guessed_wrong_index.add(string[i])
            else:
                self.guessed_wrong.add(string[i])

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
                if i in self.guessed_correct:
                    output.append(Fore.GREEN + i)
                elif i in self.guessed_wrong_index:
                    output.append(Fore.YELLOW + i)
                elif i in self.guessed_wrong:
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


class State:
    def __init__(
        self, correct_guesses, wrong_index_guesses, wrong_guesses, chances_left
    ):
        self.correct_guesses = tuple(correct_guesses)
        self.wrong_index_guesses = tuple(wrong_index_guesses)
        self.wrong_guesses = tuple(wrong_guesses)
        self.chances_left = chances_left

    def __eq__(self, other: object) -> bool:
        return (
            self.correct_guesses == other.correct_guesses
            and self.wrong_index_guesses == other.wrong_index_guesses
            and self.wrong_guesses == other.wrong_guesses
            and self.chances_left == other.chances_left
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.correct_guesses,
                self.wrong_index_guesses,
                self.wrong_guesses,
                self.chances_left,
            )
        )

    def __str__(self) -> str:
        return (
            f"CG: {''.join(self.correct_guesses)}; WI: {''.join(self.wrong_index_guesses)}; "
            f"WG: {''.join(self.wrong_guesses)}; Chances Left: {self.chances_left}"
        )


class WordleAI:
    def __init__(self, alpha=0.5, epsilon=0.1) -> None:
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

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
        data = State(
            correct_guesses=wordle.guessed_correct,
            wrong_index_guesses=wordle.guessed_wrong_index,
            wrong_guesses=wordle.guessed_wrong,
            chances_left=wordle.chance_remaining,
        )
        return data

    def get_reward(self, old_state, new_state, win=False, lose=False):
        reward = 0
        reward += (len(new_state.correct_guesses) - len(old_state.correct_guesses)) * 10
        reward += (
            len(new_state.wrong_index_guesses) - len(old_state.wrong_index_guesses)
        ) * 5
        reward -= (len(new_state.wrong_guesses) - len(old_state.wrong_guesses)) * 5
        if win:
            reward += 100
        if lose:
            reward -= 100
        return reward


def train_ai(n):
    ai = WordleAI()
    for i in range(n):
        print("Playing game " + str(i))
        wordle = Wordle(verbose=False)

        while not wordle.terminal:
            old_state = ai.get_state(wordle)
            guess = ai.choose_action(old_state, epsilon=True)

            wordle.play(guess)
            new_state = ai.get_state(wordle)
            reward = ai.get_reward(
                old_state,
                new_state,
                win=wordle.win,
                lose=wordle.terminal and not wordle.win,
            )

            ai.update(old_state, guess, new_state, reward)

    return ai


def main():
    ai = train_ai(99999999)
    wordle = Wordle()
    while not wordle.terminal:
        old_state = ai.get_state(wordle)
        guess = ai.choose_action(old_state)

        print("Guess:", guess)
        wordle.play(guess)


main()
