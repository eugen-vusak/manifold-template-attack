import numpy as np
from scipy.stats import multivariate_normal


class TemplateAttack():

    def __init__(self, output_fn):
        self.output_fn = output_fn
        self.distributions = None
        self._logpdfs = None
        self.subkeys = range(256)

    def _create_template_distrubutions(self, traces, y):
        output_classes = np.unique(y)

        # calculate means and covariances for every output class
        meanVectors = {}
        covMatrices = {}
        for output_class in output_classes:

            traces_for_class = traces[y == output_class]

            # mean for selected features
            mean = np.average(traces_for_class, axis=0)
            meanVectors[output_class] = mean

            # cov for selected features
            # print(traces_for_class)
            cov = np.cov(traces_for_class, rowvar=False)
            covMatrices[output_class] = cov

        # pooled cov matrix
        pooledCov = np.average(list(covMatrices.values()), axis=0)

        # generate distribution for every output class
        distributions = {
            output_class: multivariate_normal(
                mean=meanVectors[output_class],
                cov=pooledCov,                        # use this for pooled TA
                # cov=covMatrices[output_class],        # use this for regular TA
                # allow_singular=True                   # use this if needed
            )
            for output_class in output_classes
        }

        return distributions

    def logpdfs(self, traces, plain_text, use_cache=False):

        if use_cache and self._logpdfs is not None:
            return self._logpdfs

        # calculate logpdfs for every key guess
        self._logpdfs = np.zeros((len(traces), len(self.subkeys)))
        for example_index, (trace, plain_text) in enumerate(zip(traces, plain_text)):
            for subkey_guess in self.subkeys:
                output = self.output_fn(plain_text, subkey_guess)

                rv = self.distributions[output]
                self._logpdfs[example_index, subkey_guess] += rv.logpdf(trace)

        return self._logpdfs

    def create_template(self, traces, plain_text=None, key=None, output=None):

        if output is None:
            if plain_text is None or key is None:
                raise RuntimeError(
                    "You must provide either 'plain_text' and 'key' or 'output' parameters")

            output = self.output_fn(plain_text, key)

        else:
            if plain_text is not None or key is not None:
                raise Warning(
                    "'plain_text' and 'key' are ignored when using 'oputput'")

        self._logpdfs = None  # reset logpdfs
        self.distributions = self._create_template_distrubutions(
            traces, output)

    def guess_key(self, traces, plain_text, number_of_traces=None, use_cache=False):
        N, _ = traces.shape
        logpdfs = self.logpdfs(traces, plain_text, use_cache)

        if number_of_traces:
            random_indexes = np.random.choice(np.arange(N), number_of_traces)
            logpdfs = logpdfs[random_indexes]

        return logpdfs.sum(axis=0).argmax()
