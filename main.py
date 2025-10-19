import numpy as np
from sklearn.preprocessing import OneHotEncoder
from utils import relu, relu_derivative, softmax, compute_loss, accuracy
from optimizer import Optimizer

# preparing the MNIST dataset
#load the dataset
# Load MNIST data
mnist_path = './data/mnist.npz' 
with np.load(mnist_path) as data:
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']

# Flatten and normalize images
x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0

#preprocess the data
encoder = OneHotEncoder(sparse_output=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

#Hyperparameters
input_size = 784 # 28*28
hidden_size = 128
output_size = 10 # number of classes 0-9
learning_rate = 0.01
batch_size = 64
epochs = 5 # for faster comparison  or 10

def init_params():
    # initialize the weights and biases between input layer & hidden layer
    W1 = np.random.randn(input_size, hidden_size) * 0.01 #random weight initialization
    b1 = np.zeros((1, hidden_size)) # biases 0 intially
    # initialize the weights and biases between hidden layer & output layer
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1,output_size))
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

'''
A two layer neural network:-
    Input Layer (784) --> Hidden Layer (128) --> Output Layer (10)
    Input layer: 784 neurons (for 28*28 image pixels)
    Hidden layer: 128 neurons (with ReLU activation)
    Output layer: 10 neurons (for digits 0-9, with Softmax)
'''
# forward pass function
def forward_pass(X, params):
    #pre-activation
    a1 = np.dot(X, params['W1']) + params['b1']  # relu activation (hidden layer)
    #activation
    h1 = relu(a1)
    
    #pre-activation
    a2 = np.dot(h1, params['W2']) + params['b2'] #softmax activation (output layer)
    #activation
    h2 = softmax(a2)

    return a1, h1, a2, h2

# backward pass function (Backpropagation)
def backward_pass(X,Y,a1,h1,a2,h2, params):
    m = X.shape[0]
    #output layer gradients
    da2 = (h2 - Y)  # h2 - Y  = y_hat - y_true

    #compute gradients for output weights and biases
    dW2 = np.dot(h1.T, da2) / m   
    db2 = np.sum(da2, axis=0, keepdims=True) / m

    #compute gradients for hidden layer
    dh1 = np.dot(da2, params['W2'].T)
    da1 = dh1 * relu_derivative(h1)
    dW1 = np.dot(X.T, da1) / m
    db1 = np.sum(da1, axis=0, keepdims=True) / m

    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

# Training function
def train_model(optimizer_type):
    # params
    params = init_params()
    if optimizer_type in ['rmsprop', 'adam']:
        lr = 0.001
    else:
        lr = learning_rate
    optimizer = Optimizer(optimizer=optimizer_type, lr=lr)

    loss_history = []

    for epoch in range(epochs):
        p = np.random.permutation(x_train.shape[0])
        x_train_new = x_train[p]
        y_train_new = y_train[p]

        for i in range(0, x_train.shape[0], batch_size):
            x_batch = x_train_new[i:i+batch_size]
            y_batch = y_train_new[i:i+batch_size]

            a1, h1, a2, h2 = forward_pass(x_batch, params)
            grads = backward_pass(x_batch, y_batch, a1, h1, a2, h2, params)
            #update parameters
            params = optimizer.update_parameters(params, grads)

            #compute loss only for current batch (faster)
            loss = compute_loss(y_batch, h2)
            loss_history.append(loss)
        print(f"{optimizer_type.upper()} | Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    
    return loss_history, params

# Train the model using different optimizers
optimizers = ['sgd', 'momentum', 'nag', 'adagrad', 'rmsprop', 'adam']
loss_dict = {} # stores loss history of all optimizers
params_dict = {} # stores trained params for all optimizers

for opt in optimizers:
    print(f"\nTraining with {opt.upper()} optimizer...")
    loss_history, trained_params = train_model(opt)
    loss_dict[opt] = loss_history
    params_dict[opt] = trained_params

# Save all loss histories to file
np.save('loss_histories.npy', loss_dict)

# Evaluate the model on test data
def predict(X, params):
    _, _, _, h2 = forward_pass(X, params)
    return np.argmax(h2, axis=1)  # choose the class with highest probability

for opt in optimizers:
    y_pred = predict(x_test, params_dict[opt])  # use the full parameter dict
    y_true = np.argmax(y_test, axis=1)
    test_accuracy = accuracy(y_true, y_pred)
    print(f"{opt.upper()} Test Accuracy: {test_accuracy:.4f}")

#plotting the loss curves
import matplotlib.pyplot as plt

# Load the saved loss histories
loaded_losses = np.load('loss_histories.npy', allow_pickle=True).item()

optimizers = list(loaded_losses.keys())
num_opts = len(optimizers)

plt.figure(figsize=(15, 8))

for idx, opt in enumerate(optimizers):
    plt.subplot(2, 3, idx + 1)
    plt.plot(loaded_losses[opt])
    plt.title(opt.upper())
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid(True)

plt.tight_layout()
plt.show()



