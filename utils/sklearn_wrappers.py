from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np


# class ColumnTransformer(TransformerMixin, BaseEstimator):

#     def __init__(self, transformer, X_column):
#         self.transformer = transformer
#         self.X_column = X_column

#     def fit(self, X, y, **fit_params):
#         self.transformer.fit(X[:, self.X_column], y, **fit_params)
#         return self

#     def transform(self, X):
#         X[:, self.X_column] = self.transformer.transform(X[:, self.X_column])

#         return X

#     # def get_params(self, deep=True):
#     #     return self.transformer.get_params(deep)

#     def set_params(self, **kwargs):
#         return self.transformer.set_params(**kwargs)


class TransformedTargetTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, transformer, func):
        self.transformer = transformer
        self.func = func

    def fit(self, X, y, **fit_params):
        y = self.func(X, y)

        self.transformer.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        return self.transformer.transform(X)

    # def get_params(self, deep=True):
    #     return self.transformer.get_params(deep)

    def set_params(self, **kwargs):
        return self.transformer.set_params(**kwargs)
