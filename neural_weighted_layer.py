import numpy as np

#input is assumed to be column vector
class neural_weighted_layer:

    def __sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def __relu(self, x):
        return np.maximum(x, 0)

    def __softmax(self, x):
        exps = np.exp(x - np.max(x))
        return exps/np.sum(exps, axis=0)

    def __init__(self, n_neurons, input_features, activation='relu'):
        self.__n_neurons = n_neurons
        self.__W = np.random.randn(input_features, n_neurons)/100
        # self.__B = np.random.randn(n_neurons, 1)/100
        self.__B = np.zeros((n_neurons, 1))
        self.__activation_function = activation

    def update_weights(self, W, B):
        self.__W = self.__W - W
        self.__B = self.__B - B

    def get_W(self):
        return self.__W

    def get_activation_type(self):
        return self.__activation_function

    def activation(self, X):
        net_out = np.matmul(self.__W.T, X) + self.__B
        if self.__activation_function == 'relu':
            return self.__relu(net_out)
        elif self.__activation_function == 'sigmoid':
            return self.__sigmoid(net_out)
        elif self.__activation_function == 'softmax':
            return self.__softmax(net_out)
        # return self.__sigmoid()
