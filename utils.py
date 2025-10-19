# Helper functions
import numpy as np

def relu(z):
    # ReLU actvation function
    return np.maximum(0,z)

def relu_derivative(z):
    return (z>0).astype(float) # if z > 0 return 1 else return 0

def softmax(z):
    #softmax activation function
    z = z - np.max(z, axis=1, keepdims=True)  # shift to avoid overflow
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def compute_loss(y_true, y_pred):
    #cross-entropy loss
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
    return loss

def accuracy(y_true, y_pred):
    return np.mean(y_true==y_pred)

