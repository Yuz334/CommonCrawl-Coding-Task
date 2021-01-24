################################################################################
# CSE 251B: Programming Assignment 2
# Winter 2021
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import os, gzip
import yaml
import numpy as np


def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    return yaml.load(open('config.yaml', 'r'), Loader=yaml.SafeLoader)


def normalize_data(inp):
    """
    TODO: Normalize your inputs here to have 0 mean and unit variance.
    """
    # unit variance means each row
    msd = inp - np.average(inp, axis = 1).reshape((-1, 1))
    stds = np.std(msd, axis = 1)
    return msd / stds.reshape((-1, 1))



def one_hot_encoding(labels, num_classes=10):
    """
    TODO: Encode labels using one hot encoding and return them.
    """
    new_labels = np.zeros((len(labels), num_classes))
    for i in range(len(labels)):
        new_labels[i][labels[i]] = 1
    return new_labels


def load_data(path, mode='train'):
    """
    Load Fashion MNIST data.
    Use mode='train' for train and mode='t10k' for test.
    """

    labels_path = os.path.join(path, f'{mode}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{mode}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    normalized_images = normalize_data(images)
    one_hot_labels    = one_hot_encoding(labels, num_classes=10)

    return normalized_images, one_hot_labels


def softmax(x):
    # x should be a row vector
    expx = np.exp(x)
    expx[np.isinf(expx)] = 1000000.0 # set an upper bound
    sumExpx = np.sum(expx, axis = 1).reshape((-1, 1))
    return expx / sumExpx


class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type = "sigmoid"):
        """
        TODO: Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU", "leakyReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

        elif self.activation_type == "leakyReLU":
            return self.leakyReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        TODO: Implement the sigmoid activation here.
        """
        self.x = x
        return 1.0 / (1.0 + np.exp(-x))


    def tanh(self, x):
        """
        TODO: Implement tanh here.
        """
        self.x = x
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def ReLU(self, x):
        """
        TODO: Implement ReLU here.
        """
        self.x = x
        return np.maximum(0.0, x)

    def leakyReLU(self, x):
        """
        TODO: Implement leaky ReLU here.
        """
        self.x = x
        return np.maximum(0.1 * x, x)

    def grad_sigmoid(self):
        """
        TODO: Compute the gradient for sigmoid here.
        """
        a = np.exp(-self.x)
        return a / ((1 + a)*(1 + a))

    def grad_tanh(self):
        """
        TODO: Compute the gradient for tanh here.
        """
        a = np.cosh(self.x)
        return 1.0 / (a * a)

    def grad_ReLU(self):
        """
        TODO: Compute the gradient for ReLU here.
        """
        return 1.0 * (self.x >= 0)

    def grad_leakyReLU(self):
        """
        TODO: Compute the gradient for leaky ReLU here.
        """
        return 1.0 * (self.x >= 0) + 0.1 * (self.x < 0)

class Layer():
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        # Input is a row vector, W is a matrix given by [in_units, out_units], b is a row vector
        np.random.seed(42)
        self.w = normalize_data(np.random.normal(size = (in_units, out_units)))    # Declare the Weight matrix
        self.b = np.zeros(shape = out_units)   # Create a placeholder for Bias
        self.x = np.zeros(shape = in_units)    # Save the input to forward in this
        self.a = np.zeros(shape = out_units)    # Save the output of forward pass in this (without activation)

        self.d_x = np.zeros(shape = in_units)  # Save the gradient w.r.t x in this
        self.d_w = np.zeros(shape = (in_units, out_units))  # Save the gradient w.r.t w in this
        self.d_b = np.zeros(shape = out_units)  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        TODO: Compute the forward pass through the layer here.
        DO NOT apply activation here.
        Return self.a
        """
        # x is a row
        self.a = np.matmul(x, self.w) + self.b
        self.x = x
        return self.a

    def backward(self, delta):
        """
        TODO: Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        # Note: self.dx is delta, the procedure is: use input delta to update w/b, and then calculate the new propagate delta, delta should be a row vector
        # 1. update
        modDelta = delta.reshape(-1)
        self.d_w = np.matmul(self.x.reshape((-1, 1)), modDelta.reshape(1, -1))
        self.d_b = delta.reshape(-1)
        # 2. propagate vector
        self.d_x = np.matmul(modDelta, self.w.T)
        return self.d_x


class Neuralnetwork():
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []     # Store all layers in this list.
        self.x = None        # Save the input to forward in this
        self.y = None        # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets = None):
        """
        TODO: Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        # x is a [size, dimension] matrix, one row one sample
        # y is a column vector
        self.x = x
        self.targets = targets # it becomes a matrix
        outputs = np.zeros_like(self.targets)
        size, dimen = x.shape
        for i in range(size):
            output = x[i]
            for layer in self.layers:
                output = layer.forward(output)
            outputs[i] = output
        self.y = softmax(outputs) #stored vector should be the 'category vector'
        loss = None
        if self.targets is not None:
            loss = self.loss(self.y, self.targets)
        return self.y, loss

    def loss(self, logits, targets):
        logLogits = np.log(logits) # it calculates element-wise logits
        return -np.mean(logLogits * targets) / self.y.shape[1] # product in numpy calculates point-wise product

    def backward(self):
        '''
        TODO: Implement backpropagation here.
        Call backward methods of individual layers.
        '''
        delta = (np.mean(self.y, axis = 0) - np.mean(self.targets, axis = 0)) / self.y.shape[1]
        for i in range(len(self.layers) - 1, -1, -1):
            delta = self.layers[i].backward(delta)
        # up to here we have acquired all the gradients


def batchSGD(model, x_train, y_train, config):
    # usual batchSGD
    # x_train, y_train should be a minibatch
    numEpoch = config['epochs']
    learnRate = config['learning_rate']
    losses = []
    for i in range(numEpoch):
        y, loss = model.forward(x_train, y_train) # forward train to get output
        losses.append(loss)
        model.backward() # backward train to get gradient
        for layer in model.layers:
            if isinstance(layer, Layer):
                layer.w -= learnRate * layer.d_w
                layer.b -= learnRate * layer.d_b


def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    TODO: Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """
    batch_size = config['batch_size']  # for readability
    previous_weights = None  # for early stopping
    previous_b = None
    previous_val_loss = None  # for early stopping

    for m in range(config['epochs']):
        shuffle = np.random.choice(x_train.shape[0], size = x_train.shape[0], replace = False)  # shuffle training examples for batch SGD
        if config['early_stop'] and m % config['early_stop_epoch'] == 0 and m > 0:
            # This several is used to update the weights
            val_outputs, val_loss = model.forward(x_valid, y_valid)
            if previous_val_loss is not None and val_loss > previous_val_loss:
                m = 0
                for layer in range(model.layers): # early stop
                    if isinstance(layer, Layer):
                        layer.w = previous_weights[m]
                        layer.b = previous_b[m]
                        m += 1
                break
            previous_val_loss = val_loss
            previous_weights = []
            previous_b = []
            for layer in model.layers: # update previous_value
                if isinstance(layer, Layer):
                    previous_weights.append(np.copy(layer.w))
                    previous_b.append(np.copy(layer.b))
            print(previous_val_loss)
        # one epoch of batch SGD
        for i in range(int(np.ceil(x_train.shape[0] / batch_size))):
            # make batch
            batch = x_train[shuffle[i * batch_size: min((i + 1) * batch_size, len(x_train))]]
            batch_targets = y_train[shuffle[i * batch_size: min((i + 1) * batch_size, len(y_train))]]
            # run forward and backwark propagation on batch and update weights
            batchSGD(model, batch, batch_targets, config)


def test(model, X_test, y_test):
    """
    TODO: Calculate and return the accuracy on the test set.
    """
    y_train = model.forward(X_test)
    y_implement = np.argmax(y_train, axis = 1)
    return np.mean(np.equal(y_test.reshape(-1), y_implement.reshape(-1)))


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    # Create the model
    model = Neuralnetwork(config)

    # Load the data
    x_train, y_train = load_data(path="./", mode="train")
    x_test,  y_test = load_data(path="./", mode="t10k")
    # TODO: Create splits for validation data here.
    x_valid, x_Train = x_train[:5000], x_train[5000:]
    y_valid, y_Train = y_train[:5000], y_train[5000:]

    # Normalize input
    x_Train = normalize_data(x_Train)
    x_valid = normalize_data(x_valid)
    x_test = normalize_data(x_test)
    # TODO: train the model
    train(model, x_Train, y_Train, x_valid, y_valid, config)

    test_acc = test(model, x_test, y_test)

    # TODO: Plots
    # plt.plot(...)
