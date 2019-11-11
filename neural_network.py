import numpy as np

from neural_weighted_layer import neural_weighted_layer

class neural_network:

    def __init__(self, input_features, output_features, n_hidden_neuron_list):
        self.__n_layers = len(n_hidden_neuron_list) + 2
        self.__n_input = input_features
        self.__n_output = output_features
        self.__n_hidden_neuron_list = n_hidden_neuron_list

        #make the network
        self.__layers = []
        prev_output_length = input_features
        for neurons in n_hidden_neuron_list:
            self.__layers.append(neural_weighted_layer(neurons, prev_output_length, 'relu'))
            prev_output_length = neurons
        self.__layers.append(neural_weighted_layer(output_features, prev_output_length, 'softmax'))

    def __cross_entropy(self, X, y):
        return -1/np.shape(y)[1] * np.sum(y*np.log(X))

    def __one_hot(self, y, num_classes):
        return  np.squeeze(np.eye(num_classes)[y.reshape(-1)]).T

    def feedforward(self, X):
        layer_input = X
        self.__layer_output = [layer_input]
        for layer in self.__layers:
            layer_input = layer.activation(layer_input)
            # print("Output shape: " + str(np.shape(layer_input)))
            self.__layer_output.append(layer_input)
        return layer_input

    def train(self, X, y, num_iters=100, learning_rate=0.05, tolerance = 10**-6, reg_parameter = 1):
        #convert the output into one-hot format
        num_classes = self.__n_output
        true_y = self.__one_hot(y.astype(int), num_classes)
        alpha = learning_rate
        loss = 0
        prev_loss = tolerance + 1
        j = 0
        m = np.shape(y)[0]
        while j < num_iters and abs(loss - prev_loss) > tolerance:
            learning_rate = alpha / (1+5*j/num_iters)

            #forward propagation
            pred_y = self.feedforward(X)
            prev_loss = loss
            loss = self.__cross_entropy(pred_y, true_y)
            print("Iteration number: " + str(j) + ", loss is " + str(loss))


            #backward propagation

            #for the softmax layer
            dLdZ = pred_y - true_y
            dLdW = 1/m * np.matmul(self.__layer_output[-2], dLdZ.T) + reg_parameter/m * np.linalg.norm(self.__layers[-1].get_W())
            dLdB = 1/m * np.sum(dLdZ, axis=1, keepdims=True)
            self.__layers[-1].update_weights(learning_rate * dLdW, learning_rate * dLdB)
            prevW = self.__layers[-1].get_W()
            #for the previous layers

            for i in range(len(self.__layers)-2, -1, -1):  #until the penultimate hidden layer
                curr_layer = self.__layers[i]
                act_type = curr_layer.get_activation_type()
                act_der = 0
                if act_type == 'relu':
                    act_der = (self.__layer_output[i+1] > 0)
                    act_der = act_der*1 #converting boolean array into int array
                elif act_type == 'sigmoid':
                    a = self.__layer_output[i+1]
                    act_der = a - a*a
                dLdZ = np.matmul(prevW, dLdZ) * act_der
                dLdW = 1/m * np.matmul(self.__layer_output[i], dLdZ.T) + reg_parameter/m * np.linalg.norm(curr_layer.get_W())
                dLdB = 1/m * np.sum(dLdZ, axis = 1, keepdims = True)
                self.__layers[i].update_weights(learning_rate * dLdW, learning_rate * dLdB)
                prevW = curr_layer.get_W()

            j = j+1
