import numpy as np


TRAIN_SAMPLES_PER_WORLD = 420
VAL_SAMPLES_PER_WORLD = 120
TEST_SAMPLES_PER_WORLD = 60


def sum_parity_labels(c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    return ((c1 + c2) % 2).astype(np.int64)


def default_dataset_dimensions(n_digits: int) -> dict[str, int]:
    worlds = n_digits * n_digits
    return {
        "train": TRAIN_SAMPLES_PER_WORLD * worlds,
        "val": VAL_SAMPLES_PER_WORLD * worlds,
        "test": TEST_SAMPLES_PER_WORLD * worlds,
    }


def in_distribution_mask(real_concepts: np.ndarray) -> np.ndarray:
    left_even = (real_concepts[:, 0] % 2) == 0
    right_even = (real_concepts[:, 1] % 2) == 0
    return (left_even & right_even) | (~left_even & ~right_even) | (~left_even & right_even)


def ood_mask(real_concepts: np.ndarray) -> np.ndarray:
    left_even = (real_concepts[:, 0] % 2) == 0
    right_even = (real_concepts[:, 1] % 2) == 0
    return left_even & (~right_even)
