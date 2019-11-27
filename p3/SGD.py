import numpy as np


class SGD(object):

    def __init__(self, x, y, model, learning_rate):
        """
        :param x: nxd matrix, n is num data points, d is data dimension
        :param y: nx1 vector, n is num data points
        :param objective: function which takes in x, y and outputs a loss
        :param gradient: function which takes in x, y and outputs a gradient of loss
        :param learning_rate: iterator which outputs learning rate for during round t
        """
        self.x = x
        self.y = y

        self.num_data, self.data_dim = x.shape
        assert self.num_data == self.y.shape[0]

        self.model = model
        self.learning_rate = learning_rate


    def optim_step(self):
        i = np.random.choice(self.num_data)
        x_i, y_i = self.x[i,:], self.y[i]
        gradient = self.model.gradient(x_i, y_i)
        # print(gradient)
        self.model.update(gradient, self.learning_rate)

    def run(self, num_iterations=10000):
        for iteration in range(num_iterations):
            loss = self.model.loss(self.x, self.y)
            print(loss)
            self.optim_step()

