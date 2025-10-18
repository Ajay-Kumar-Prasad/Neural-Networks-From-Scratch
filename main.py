import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from utils import relu, relu_derivative, softmax, compute_loss, accuracy

# preparing the MNIST dataset
#load the dataset
(x_train,y_train),(x_test,y_test) = mnist.load_data()

#preprocess the data
x_train = x_train.reshape(-1,28*28) / 255.0 # flatten the 28*28 images to 784-dim vectors + normalize
x_test = x_test.reshape(-1,28*28) / 255.0
encoder = OneHotEncoder(sparse_output=False)
y_train = encoder.fit_transform(y_train.reshape(-1,1))
y_test = encoder.transform(y_test.reshape(-1,1))

#Hyperparameters
input_size = 784 # 28*28
hidden_size = 128
output_size = 10 # number of classes 0-9
learning_rate = 0.01
batch_size = 64
epochs = 10

# initialize the weights and biases between input layer & hidden layer
W1 = np.random.randn(input_size, hidden_size) * 0.01 #random weight initialization
b1 = np.zeros((1, hidden_size)) # biases 0 intially
# initialize the weights and biases between hidden layer & output layer
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1,output_size))

'''
A two layer neural network:-
    Input Layer (784) --> Hidden Layer (128) --> Output Layer (10)
    Input layer: 784 neurons (for 28*28 image pixels)
    Hidden layer: 128 neurons (with ReLU activation)
    Output layer: 10 neurons (for digits 0-9, with Softmax)
'''
# forward pass function
def forward_pass(X):
    #pre-activation
    a1 = np.dot(X, W1) + b1  # relu activation (hidden layer)
    #activation
    h1 = relu(a1)
    
    #pre-activation
    a2 = np.dot(h1, W2) + b2 #softmax activation (output layer)
    #activation
    h2 = softmax(a2)

    return a1, h1, a2, h2

# backward pass function (Backpropagation)
def backward_pass(X,Y,a1,h1,a2,h2):
    m = X.shape[0]
    #output layer gradients
    da2 = (h2 - Y)  # h2 - Y  = y_hat - y_true

    #compute gradients for output weights and biases
    dW2 = np.dot(h1.T, da2) / m   
    db2 = np.sum(da2, axis=0, keepdims=True) / m

    #compute gradients for hidden layer
    dh1 = np.dot(da2, W2.T)
    da1 = dh1 * relu_derivative(h1)
    dW1 = np.dot(X.T, da1) / m
    db1 = np.sum(da1, axis=0, keepdims=True) / m

    return dW1,db1, dW2, db2

# update parameters function
def update_parameters(dW1, db1, dW2, db2, learning_rate):
    global W1, b1, W2, b2
    lr = learning_rate
    W1 -= lr*dW1
    b1 -= lr*db1
    W2 -= lr*dW2
    b2 -= lr*db2

# Training loop
loss_history = [] # keep tracks of losses at each step

# mini-batch gradient descent
for epoch in range(epochs):
    p = np.random.permutation(x_train.shape[0]) # shuffle the data
    x_train_new = x_train[p]
    y_train_new = y_train[p]

    for i in range(0, x_train.shape[0], batch_size):
        # create batches
        x_batch = x_train_new[i:i+batch_size]
        y_batch = y_train_new[i:i+batch_size]

        # calculate activations
        a1, h1, a2, h2 = forward_pass(x_batch)
        # calculate gradients
        dW1, db1, dW2, db2 = backward_pass(x_batch, y_batch, a1, h1, a2, h2)
        # update weights and biases
        update_parameters(dW1,db1,dW2,db2,learning_rate)

        _,_,_, y_pred = forward_pass(x_train)
        #compute loss only on current mini-batch (faster)
        loss = compute_loss(y_batch, h2)
        # store loss
        loss_history.append(loss)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

np.save('loss_history.npy', np.array(loss_history))

# Evaluate the model on test data
def predict(X):
    _,_,_, h2 = forward_pass(X)
    return np.argmax(h2, axis=1) # choose the class with highest probability

y_pred = predict(x_test)
y_true = np.argmax(y_test, axis=1)
test_accuracy = accuracy(y_true, y_pred)

print("Test Accuracy:", test_accuracy)

#plotting the loss curve
import matplotlib.pyplot as plt
loss_history = np.load('loss_history.npy')
plt.plot(loss_history)
plt.show()






