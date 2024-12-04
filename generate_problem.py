import numpy as np


def true_f0(x: np.ndarray) -> np.ndarray:
    return x[0] + np.sin(x[1]) / 5


def true_f1(x: np.ndarray) -> np.ndarray:
    return np.sin(x[0]) + np.cos(x[1]) / 5


def true_f2(x: np.ndarray) -> np.ndarray:
    return np.sin(x[0]) + np.cos(x[1]) / 5 + x[0] * x[1]


TEST_SIZE = 10_000
TRAIN_SIZE = 1000


def gen_problem(filename: str, true_f: callable):
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

    np.savez(filename, x=x_train, y=y_train)


if __name__ == "__main__":
    gen_problem("problem_0.npz", true_f0)
    gen_problem("problem_1.npz", true_f1)
    gen_problem("problem_2.npz", true_f2)
