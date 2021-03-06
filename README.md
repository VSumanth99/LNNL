This is an implementation of back propagation in Python. The neural_network
class implements a fully connected neural network with a scalable architecture.
Both the number of hidden layers, and the number of neurons can be varied.
The output layer is a softmax layer, while the hidden layers can either be relu
or sigmoid.

The instance of the neural_network class has three important methods:

1. The constructor: It takes as it's parameters, the hyperparameters of the
neural network, such as number of input features, number of output classes and
the network architecture. Syntax:

```network = neural_network(INPUT_FEATURES, NUM_CLASSES, [array of neurons in every hidden layer])```

2. The train function: It is used to train the model.
Syntax:
```network.train(TRAINING_DATA, TRAINING_LABELS, NUMBER_OF_ITERATIONS, LEARNING_RATE, ERROR_TOLERANCE, LAMBDA)```
where `LAMBDA` is the L2 regularisation parameter.
This function prints the loss after every iteration of training

3. The feedforward function: It uses the learnt parameters and gives an output.
Syntax: ```network.feedforward(TEST_DATA)```

The `neural_weighted_layer` is a helper class, and it implements a fully connected
layer with either relu or sigmoid activation functions.
