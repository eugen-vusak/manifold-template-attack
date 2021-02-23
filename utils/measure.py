import numpy as np

def guessing_entropy_and_success_rate(y_log_proba, secret_key, number_of_traces, number_of_experiments=50):

    N, L = y_log_proba.shape

    GE = np.zeros(number_of_traces)
    SR = np.zeros(number_of_traces)

    for _ in range(number_of_experiments):

        # generate random sample indexes
        random_indexes = np.random.choice(np.arange(N), number_of_traces)

        y_log_proba_sample = y_log_proba[random_indexes]

        # compute cumulative sum of log_probas (log of cumulative product)
        cumsum = y_log_proba_sample.cumsum(axis=0)

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
