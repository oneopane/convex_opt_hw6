import numpy as np
from scipy.linalg import norm

class BaseModel(object):

    def __init__(self, params, name):

        self.params = params
        self.name = name


    def forward(self, X):
        raise NotImplementedError

    def gradient(self, X, y):
        raise NotImplementedError

    def __repr__(self):
        return self.name


class OLS(BaseModel):

    def __init__(self, data_dim, params=None):
        if params is None:
            params = np.random.rand(data_dim)
        else:
            assert len(params) == data_dim
        super(OLS, self).__init__(params, "OLS")

    def forward(self, X):
        return X @ self.params

    def loss(self, X, y):
        return 0.5 * norm(list([X @ self.params - y]), 2) / len(y)

    def gradient(self, X, y):
        return X.T * (X @ self.params - y)

    def update(self, gradient, learning_rate):
        self.params -= learning_rate * gradient


class Ridge(BaseModel):

    def __init__(self, data_dim, lambda_, params=None):

        if params is None:
            params = np.random.rand(data_dim)
        else:
            assert len(params) == data_dim
        self.lambda_ = lambda_
        super(Ridge, self).__init__(params, 'Ridge({})'.format(self.lambda_))

    def forward(self, X):
        return X @ self.params

    def loss(self, X, y):
        return norm(list([X @ self.params - y]), 2) / 2 + self.lambda_/2 * norm(self.params, 2) / len(y)

    def gradient(self, X, y):
        return X.T * (X @ self.params - y) + self.lambda_ * self.params

    def update(self, gradient, learning_rate):
        self.params -= learning_rate * gradient



def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression(BaseModel):

    def __init__(self, data_dim, params=None):
        if params is None:
            params = np.random.rand(data_dim)
        else:
            assert len(params) == data_dim

        super(LogisticRegression, self).__init__(params, 'LogisticRegression')

    def forward(self, X):

        return sigmoid(X @ self.params)

    def loss(self, X, y):
        return - sum(y * self.forward(X) + (1 - y) * (1 - self.forward(X))) / len(y)

    def gradient(self, X, y):
        return X.T * (sigmoid(X) - y)

    def update(self, gradient, learning_rate):
        self.params -= learning_rate * gradient

# could be wrong, need to make sure it is point wise
def ReLU(x):
    return np.max(0, x)

class FeedForwardNeuralNetwork(BaseModel):

    def __init__(self, layers, activation=ReLU):

        self.layers = layers

        params = OrderedDict()
        for i, in_dim, out_dim in enumerate(zip(self.layers[:1], self.layers[1:])):
            layer = np.random.normal(0, 1, size=(out_dim, in_dim))
            params['Fully_Connected_{}'.format(i)] = layer

        super(FeedForwardNeuralNetwork, self).__init__(params, "FeedForwardNeuralNetwork")

        self.activation = activation

    def forward(self, X):

        hidden = X
        for layer in self.params:
            hidden = self.activation(layer @ hidden)

        return hidden

    def loss(self, X, y):
        pass

    def gradient(self, X, y):
        pass

    def update(self, gradient, learning_rate):
        pass

