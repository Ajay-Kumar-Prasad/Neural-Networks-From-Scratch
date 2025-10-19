import numpy as np
# define a class Optimizer to optimize the parameters
class Optimizer:
    #constructor
    def __init__(self, optimizer='sgd',lr = 0.01, beta1 = 0.9, beta2 = 0.999, momentum=0.9):
        self.optimizer = optimizer.lower()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.momentum = momentum
        self.t = 0 # time step for Adam bias correction
        self.v, self.s = {}, {} # past velocity and squared gradients

    # update parameters function based on optimizer
    def update_parameters(self, params, grads):
        # params = dict(W1, b1, W2, b2)
        # grads = dict(dW1, db1, dW2, db2)
        
        #initialize s and v
        if not self.v:
            for key in params.keys():
                self.v[key] = np.zeros_like(params[key]) 
                self.s[key] = np.zeros_like(params[key])
        
        # SGD / Batch gradient descent
        if self.optimizer == 'sgd':
            for key in params.keys(): # key = W1 or W2 or b1 or b2
                params[key] -= self.lr * grads['d' + key] # dW1, db1, dW2, db2
        
        # Momentum based gradient descent
        elif self.optimizer == 'momentum':
            for key in params.keys():
                self.v[key] = self.momentum * self.v[key] + self.lr * grads["d" + key]
                params[key] -= self.v[key]
            
        # NAG
        elif self.optimizer == 'nag':
            for key in params.keys():
                #look ahead step
                prev_v = self.v[key]
                self.v[key] = self.momentum * self.v[key] + self.lr * grads["d" + key]
                params[key] -= self.momentum * prev_v + (1 + self.momentum)* (self.lr * grads["d" + key])
        
        # Adagrad
        elif self.optimizer == 'adagrad':
            for key in params.keys():
                self.s[key] += grads["d" + key] **2 # accumulate sqrd gradients
                params[key] -= (self.lr * grads["d"+key])/(np.sqrt(self.s[key])+1e-8) # update params


        # RMSProp
        elif self.optimizer == 'rmsprop':
            for key in params.keys():
                self.s[key] = self.beta2 * self.s[key] + (1-self.beta2) * (grads["d"+key]**2)
                params[key] -= (self.lr * grads["d"+key])/ ( np.sqrt(self.s[key]) + 1e-8)
        
        # Adam
        elif self.optimizer == 'adam':
            self.t += 1
            for key in params.keys():
                # first moment estimate -- mean of gradients (momentum)
                self.v[key] = self.beta1 * self.v[key] + (1-self.beta1) * grads["d" + key] 
                # second moment estimate -- mean of squared gradients (RMSProp)
                self.s[key] = self.beta2 * self.s[key] + (1-self.beta2) * (grads["d" + key]**2)

                # bias correction term
                v_hat = self.v[key] / (1-(self.beta1**self.t))
                s_hat = self.s[key] / (1-(self.beta2**self.t))

                # update parameters
                params[key] -= (self.lr * v_hat)/ (np.sqrt(s_hat) + 1e-8)

        return params # updated parameters
