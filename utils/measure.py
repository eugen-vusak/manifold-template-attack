import numpy as np


def make_ge_scoring(number_of_experiments):

    def ge_scoring(clf, X, y):
        scores = clf.predict_proba(X)

        # compute measures (GE and SR)
        np.random.seed(43)
        ge, sr = guessing_entropy_and_success_rate(
            scores,
            y[0],
            number_of_experiments=number_of_experiments)

        return ge[-1]

    return ge_scoring


def guessing_entropy_and_success_rate(y_log_proba, secret_key, number_of_experiments=50):

    N, L = y_log_proba.shape

    GE = np.zeros(N)
    SR = np.zeros(N)

    for _ in range(number_of_experiments):

        np.random.shuffle(y_log_proba)

        # compute cumulative sum of log_probas (log of cumulative product)
        cumsum = y_log_proba.cumsum(axis=0)

        # argsort to find rank of key guesses (first is worst, last is best)
        ranked_key_guesses = cumsum.argsort(axis=1)

        # find key_rank of secret_key
        key_rank = np.argmax(ranked_key_guesses == secret_key, axis=1)
        key_rank = L - key_rank - 1  # adjust to reverse order
        # print(key_rank)

        key_is_best_guess = (key_rank == 0).astype(int)

        GE += key_rank
        SR += key_is_best_guess

    GE /= number_of_experiments
    SR /= number_of_experiments

    return GE, SR
