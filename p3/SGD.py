import numpy as np

class Optimizer(object):

    def __init__(self, name, x, y, model, learning_rate):
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
        self.losses = []

    def run(self, num_iterations=10000):
        loss = self.model.loss(self.x, self.y)
        self.losses.append(loss)
        for iteration in range(num_iterations):
            self.optim_step()
            loss = self.model.loss(self.x, self.y)
            self.losses.append(loss)

    def optim_step(self):
        raise NotImplementedError

class SGD(Optimizer):

    def __init__(self, x, y, model, learning_rate):
        """
        :param x: nxd matrix, n is num data points, d is data dimension
        :param y: nx1 vector, n is num data points
        :param objective: function which takes in x, y and outputs a loss
        :param gradient: function which takes in x, y and outputs a gradient of loss
        :param learning_rate: iterator which outputs learning rate for during round t
        """
        super(SGD, self).__init__('SGD', x, y, model, learning_rate)

    def optim_step(self):
        i = np.random.choice(self.num_data)
        x_i, y_i = self.x[i,:], self.y[i]
        gradient = self.model.gradient(x_i, y_i)
        # print(gradient)
        self.model.update(gradient, self.learning_rate)

    def run(self, num_iterations=10000):
        loss = self.model.loss(self.x, self.y)
        self.losses.append(loss)
        for iteration in range(num_iterations):
            self.optim_step()
            loss = self.model.loss(self.x, self.y)
            self.losses.append(loss)


class SVRG(Optimizer):

    def __init__(self, x, y, model, learning_rate):

        super(SVRG, self).__init__(x, y, model, learning_rate)

        self.average_gradient = 0
        self.gradient_list = []
        for i in range(self.num_data):
            gradient = self.model.gradient(self.x[i,:], self.y[i])
            self.gradient_list.append(gradient)
            self.average_gradient += gradient
        self.average_gradient /= self.num_data

    def optim_step(self):
        i = np.random.choice(self.num_data)
        new_gradient = self.model.gradient(self.x[i,:], self.y[i])
        self.average_gradient += new_gradient - self.gradient_list[i]
        self.gradient_list[i] = new_gradient
        self.model.update(gradient, self.learning_rate)

    def run(self, num_iterations=10000):
        loss = self.model.loss(self.x, self.y)
        self.losses.append(loss)
        for iteration in range(num_iterations):
            self.optim_step()
            loss = self.model.loss(self.x, self.y)
            self.losses.append(loss)
