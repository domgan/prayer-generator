import os
from pathlib import Path
from typing import Tuple

import numpy as np


def load_data(path: [str, Path] = 'data') -> str:
    path = Path(path)
    text = ''
    for file in os.listdir(path):
        with open(path / file) as f:
            text += f.read().replace('\n', ' ').replace('(', ' ').replace(')', ' ') + ' '
    return text


def word_embedding(text: str) -> Tuple[np.ndarray, np.ndarray]:
    vocab = np.array(sorted(set(text)))
    char2idx = {u: i for i, u in enumerate(vocab)}
    text_int = np.array([char2idx[c] for c in text])
    return text_int, vocab


if __name__ == '__main__':
    text = load_data()
    data = word_embedding(text)
    seq_length = 100
    examples_per_epoch = len(text)//(seq_length+1)
