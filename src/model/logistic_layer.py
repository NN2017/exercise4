from __future__ import print_function
import time
import numpy as np

from util.activation_functions import Activation


class LogisticLayer():
    """
    A layer of neural

    Parameters
    ----------
    nIn: int: number of units from the previous layer (or input data)
    nOut: int: number of units of the current layer (or output)
    activation: string: activation function of every units in the layer
    isClassifierLayer: bool:  to do classification or regression

    Attributes
    ----------
    nIn : positive int:
        number of units from the previous layer
    nOut : positive int:
        number of units of the current layer
    weights : ndarray
        weight matrix
    activation : functional
        activation function
    activationString : string
        the name of the activation function
    isClassifierLayer: bool
        to do classification or regression
    deltas : ndarray
        partial derivatives
    size : positive int
        number of units in the current layer
    shape : tuple
        shape of the layer, is also shape of the weight matrix
    """

    def __init__(self, nIn, nOut, weights=None,
                 activation='sigmoid', isClassifierLayer=False, name=""):

        # Get activation function from string
        self.activationString = activation
        self.activation = Activation.getActivation(self.activationString)
        self.activationDerivative = Activation.getDerivative(
                                    self.activationString)

        self.nIn = nIn
        self.nOut = nOut

        self.inp = np.ndarray((nIn+1, 1))
        self.inp[0] = 1
        self.preactivation = np.ndarray((nOut, 1))
        self.outp = np.ndarray((nOut, 1))
        self.deltas = np.zeros((nOut, 1))

        # You can have better initialization here
        if weights is None:
            rns = np.random.RandomState(int(time.time()))
            self.weights = rns.uniform(size=(nIn + 1, nOut))-0.5
        else:
            assert(weights.shape == (nIn + 1, nOut))
            self.weights = weights

        self.isClassifierLayer = isClassifierLayer

        # Some handy properties of the layers
        self.size = self.nOut
        self.shape = self.weights.shape
        self.name = name
        print("Created", self.name, "layer with weights", self.shape, "delta:", self.deltas.shape, )

    def forward(self, inp):
        """
        Compute forward step over the input using its weights

        Parameters
        ----------
        inp : ndarray
            a numpy array (nIn + 1,1) containing the input of the layer

        Change outp
        -------
        outp: ndarray
            a numpy array (nOut,1) containing the output of the layer
        """

        # Here you have to implement the forward pass
        self.inp[:] = np.reshape(inp, (self.nIn+1, 1))
        self.preactivation[:] = np.reshape(np.dot(inp, self.weights), (self.nOut,1))
        self.outp[:] = np.reshape(self.activation(self.preactivation), (self.nOut,1))
        return self.outp

    def computeDerivative(self, next_derivatives, next_weights):
        """
        Compute the derivatives (backward)

        Parameters
        ----------
        next_derivatives: ndarray
            a numpy array containing the derivatives from next layer
            for the last layer this is a (10,1) ndarray (multivariate case) or a (1,1) binary classification
            otherwise dimension (nNeurons_next_layer, 1)
        next_weights : ndarray
            a numpy array containing the weights from next layer
            for the last layer this is just a single value 1.0
            otherwise dimension (nOut_this_layer==nNeurons_this_layer, nNeurons_next_layer)
            !!! the biases from the weights array are already removed (if there were any)

        Change deltas
        -------
        deltas: ndarray
            a numpy array containing the partial derivatives on this layer
        """

        # Here the implementation of partial derivative calculation

        # In case of the output layer, next_weights is array of 1
        # and next_derivatives - the derivative of the error will be the errors
        # Please see the call of this method in LogisticRegression.
        # self.deltas = (self.outp *
        #              (1 - self.outp) *
        #               np.dot(next_derivatives, next_weights))

        # Or more general: output*(1-output) is the derivatives of sigmoid
        # (sigmoid_prime)
        # self.deltas = (Activation.sigmoid_prime(self.outp) *
        #                np.dot(next_derivatives, next_weights))

        # Or even more general: doesn't care which activation function is used
        # dado: derivative of activation function w.r.t the output


        dado = self.activationDerivative(self.outp)
        self.deltas = (dado * np.dot(next_weights, next_derivatives))
        # since the dimensions have changed, the order in the dot product is changed, too.

        # Or you can explicitly calculate the derivatives for two cases
        # Page 40 Back-propagation slides
        # if self.isClassifierLayer:
        #     self.deltas = (next_derivatives - self.outp) * self.outp * \
        #                   (1 - self.outp)
        # else:
        #     self.deltas = self.outp * (1 - self.outp) * \
        #                   np.dot(next_derivatives, next_weights)
        # Or you can have two computeDerivative methods, feel free to call
        # the other is computeOutputLayerDerivative or such.
        return self.deltas

    def updateWeights(self, learningRate):
        """
        Update the weights of the layer
        """

        # weight updating as gradient descent principle
        for neuron in range(0, self.nOut):
            for feature in range(self.nIn +1):
                update = learningRate * self.deltas[neuron] * self.inp[feature]
                if update > 1:
                    raise(ValueError, "I don't want to have to high updates to the weights")
                self.weights[feature, neuron] -= update