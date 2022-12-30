from sklearn.base import BaseEstimator, TransformerMixin

class Remover(BaseEstimator, TransformerMixin):

    def __init__(self, useless):
        self.useless = useless

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        X_copy = X_copy.drop(self.useless, axis=1)

        return X_copy
