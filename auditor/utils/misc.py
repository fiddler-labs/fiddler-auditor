"""For miscellaneous utilities"""
import math
from typing import List
import random

# nearby keys on a QWERTY keyboard
KEYS_MAP = {
    'a': ['q', 'w', 's', 'x', 'z'],
    'b': ['v', 'g', 'h', 'n'],
    'c': ['x', 'd', 'f', 'v'],
    'd': ['s', 'e', 'r', 'f', 'c', 'x'],
    'e': ['w', 's', 'd', 'r'],
    'f': ['d', 'r', 't', 'g', 'v', 'c'],
    'g': ['f', 't', 'y', 'h', 'b', 'v'],
    'h': ['g', 'y', 'u', 'j', 'n', 'b'],
    'i': ['u', 'j', 'k', 'o'],
    'j': ['h', 'u', 'i', 'k', 'n', 'm'],
    'k': ['j', 'i', 'o', 'l', 'm'],
    'l': ['k', 'o', 'p'],
    'm': ['n', 'j', 'k', 'l'],
    'n': ['b', 'h', 'j', 'm'],
    'o': ['i', 'k', 'l', 'p'],
    'p': ['o', 'l'],
    'q': ['w', 'a', 's'],
    'r': ['e', 'd', 'f', 't'],
    's': ['w', 'e', 'd', 'x', 'z', 'a'],
    't': ['r', 'f', 'g', 'y'],
    'u': ['y', 'h', 'j', 'i'],
    'v': ['c', 'f', 'g', 'v', 'b'],
    'w': ['q', 'a', 's', 'e'],
    'x': ['z', 's', 'd', 'c'],
    'y': ['t', 'g', 'h', 'u'],
    'z': ['a', 's', 'x'],
}


def round_list(nums: List[float], precision=2):
    return [round(n, precision) for n in nums]


def simulate_typos(sentence: str, typo_probability: float) -> str:

    # convert the message to a list of characters
    charList = list(sentence)
    outputList = []

    # the positions of eligible of characters that can have typos
    eligible_positions = []
    for pos in range(0, len(charList)):
        if charList[pos] in KEYS_MAP:
            eligible_positions.append(pos)

    n_chars_to_flip = math.ceil(len(charList) * typo_probability)

    # list of characters that will be flipped
    pos_to_flip = []
    for i in range(n_chars_to_flip):
        rand_pos = random.choice(eligible_positions)
        pos_to_flip.append(rand_pos)
        eligible_positions.remove(rand_pos)

    # insert typos
    for pos in range(0, len(charList)):
        char = charList[pos]
        if pos in pos_to_flip and char in KEYS_MAP:
            typo_array = KEYS_MAP[char] + [char, None]
            typo_selection = random.choice(typo_array)
            if typo_selection is not None:
                outputList.append(typo_selection)
        else:
            outputList.append(char)

    return ''.join(outputList)
