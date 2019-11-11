from neural_network import neural_network
import numpy as np
import csv

X=0
y=0
np.set_printoptions(threshold=np.inf)

#import the training dataset
with open('mnist_train.csv', 'r') as train_file:
    csv_reader = csv.reader(train_file, delimiter=',')
    row_count = 7000
    X = np.zeros((row_count, 784))
    y = np.zeros((row_count, 1))
    i = 0
    for row in csv_reader:
        y[i] = int(row[0])
        X[i] = row[1:]
        i = i+1
X = X.T

#setup a fully connected neural network 
#the parameters are the number of training examples, the number of classes in the output, and an array with the number of neurons in each hidden layer.
#both the number of hidden layers and the number of neurons can be changed
network = neural_network(np.shape(X)[0], 10, [256, 512, 256])

#train the neural network
#the parameters are the training examples, the labels, the learning rate, error tolerance (default value of 10**-6 is used here)
#and the regularisation parameter lambda (L2 regularisation)
network.train(X, y, 1000, 0.005, reg_parameter = 0.005)

#the network output is obtained by network.feedforward()
scores = network.feedforward(X)
predicted_class = np.argmax(scores, axis=0)

#calculate training accuracy
correct = 0
k = 0
for i in y:
    if predicted_class[k] == i:
        correct = correct + 1
    k = k+1
print("Training accuracy: " + str(correct/7000))


#open the test data
with open('mnist_test.csv', 'r') as train_file:
    csv_reader = csv.reader(train_file, delimiter=',')
    row_count = 3000
    X_test = np.zeros((row_count, 784))
    i = 0
    for row in csv_reader:
        X_test[i] = row
        i = i+1
    #X = X.T

#run the model on test data
pred_y = network.feedforward(X_test.T)
predicted_class = np.argmax(pred_y, axis = 0)

#write to submission file
with open('submission.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['id', 'label'])
    k=1
    for i in predicted_class:
        writer.writerow([k, i])
        k = k + 1

csv_file.close()
