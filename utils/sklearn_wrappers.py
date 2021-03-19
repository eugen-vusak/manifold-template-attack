from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np


class ColumnTransformer(TransformerMixin, BaseEstimator):

    def __init__(self, transformer, X_column, y_column=None):
        self.transformer = transformer
        self.X_column = X_column
        self.y_column = y_column

    def __X(self, X):
        return X[:, :self.X_column]

    def __y(self, y):
        return y
        if self.y_column is None:
            return y

        return y[self.y_column]

    def fit(self, X, y, **fit_params):
        print(X.shape)
        Xt, _ = np.hsplit(X, [self.X_column])

        self.transformer.fit(Xt, y, **fit_params)
        return self

    def transform(self, X):
        Xt, Xr = np.hsplit(X, [self.X_column])
        Xt = self.transformer.transform(Xt)

        return np.hstack((Xt, Xr))

    def get_params(self, deep=True):
        return self.transformer.get_params(deep)

    def set_params(self, **kwargs):
        return self.transformer.set_params(**kwargs)


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

    def get_params(self, deep=True):
        return self.transformer.get_params(deep)

    def set_params(self, **kwargs):
        return self.transformer.set_params(**kwargs)
