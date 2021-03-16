from itertools import combinations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SumOfDifferenceFeatureSelector(TransformerMixin, BaseEstimator):

    def __init__(self, n_components=50, feature_spacing=5):
        self.n_components = n_components
        self.feature_spacing = feature_spacing
        self.features = []
        self.classes = None

    def fit(self, traces, y):
        self.classes = np.unique(y)

        trace_len = traces.shape[1]

        # calculate mean of every group
        means = [
            np.average(traces[y == y_i], axis=0)
            for y_i in self.classes
        ]

        # calculate sum of pairwise absolute differences of means
        sumDiffs = np.zeros(trace_len)
        for p_i, p_j in combinations(means, 2):
            sumDiffs += np.abs(p_i - p_j)

        # select POIs
        Nf = len(sumDiffs)
        for _ in range(self.n_components):
            # select best feature (largest diff)
            feature_i = np.nanargmax(sumDiffs)
            self.features.append(feature_i)

            # ignore neighbourhood around selected feature
            ignore_start = max(feature_i - self.feature_spacing, 0)
            ignore_end = min(feature_i + self.feature_spacing + 1, Nf)
            sumDiffs[ignore_start:ignore_end] = np.nan

        return self

    def transform(self, traces, y=None):
        return traces[:, self.features]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
