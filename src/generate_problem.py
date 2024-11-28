import numpy as np


def true_f(x: np.ndarray) -> np.ndarray:
    return x[0] + np.sin(x[1]) / 5


TEST_SIZE = 10_000
TRAIN_SIZE = 1000

x_validation = np.vstack(
    [
        np.random.random_sample(size=TEST_SIZE) * 2 * np.pi - np.pi,
        np.random.random_sample(size=TEST_SIZE) * 2 - 1,
    ]
)
y_validation = true_f(x_validation)
train_indexes = np.random.choice(TEST_SIZE, size=TRAIN_SIZE, replace=False)
x_train = x_validation[:, train_indexes]
y_train = y_validation[train_indexes]
assert np.all(y_train == true_f(x_train)), "D'ho"

np.savez("problem_0.npz", x=x_train, y=y_train)
