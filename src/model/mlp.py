from model.logistic_layer import LogisticLayer
from model.classifier import Classifier
from sklearn.metrics import accuracy_score
from util.loss_functions import *

import numpy as np


class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, hypes=[30], inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='bce', learningRate=0.01, epochs=50, cost=None, name=""):

        """
        A MNIST recognizer based on multi-layer perceptron algorithm

        Parameters
        ----------
        train : data_set
        valid : data_set
        test : data_set
        learningRate : float
        epochs : positive int

        Attributes
        ----------
        trainingSet : data_set
        validationSet : data_set
        testSet : data_set
        learningRate : float
        epochs : positive int
        performances: array of floats
        """

        self.learningRate = learningRate
        self.epochs = epochs
        self.outputTask = outputTask  # Either classification or regression
        self.outputActivation = outputActivation
        self.cost = cost

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test
        
        if loss == 'bce':
            self.loss = BinaryCrossEntropyError()
        elif loss == 'sse':
            self.loss = SumSquaredError()
        elif loss == 'mse':
            self.loss = MeanSquaredError()
        elif loss == 'different':
            self.loss = DifferentError()
        elif loss == 'absolute':
            self.loss = AbsoluteError()
        else:
            raise ValueError('There is no predefined loss function ' +
                             'named ' + loss)

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []
        self.performances_trainset = []
        self.losses = []
        self.loss_value = None
        self.name = name

        # Build up the network from specific layers
        self.layers = []

        # Input layer (a hidden layer)
        assert len(hypes) >= 1, "there has to be >1 hidden layer"
        inputActivation = "sigmoid"
        self.layers.append(LogisticLayer(self.trainingSet.input.shape[1], hypes[0],
                           None, inputActivation, False, name="layer0"))
        # Inner layers (a hidden layer)
        for l in range(1, len(hypes)):
            self.layers.append(LogisticLayer(hypes[l-1], hypes[l], None,
                           inputActivation, False, name="layer"+str(l)))

        # Output layer
        try:
            outlayer_size = len(self.trainingSet.label[0])
        except TypeError:
            outlayer_size = 1
        self.layers.append(LogisticLayer(hypes[-1], outlayer_size,
                           None, outputActivation, True, name="final"))

        self.inputWeights = inputWeights
        if inputWeights is not None:
            assert len(inputWeights) == len(self.layers), "# Layers != #input weights"
            for l in range(len(self.layers)):
                assert self.layers[l].shape == inputWeights[l].shape, "Input Weights on Layer "+str(l)+" don't match"
                self.layers[l].weights = inputWeights[l]


        # add bias values ("1"s) at the beginning of all data sets
        self.trainingSet.input = np.insert(self.trainingSet.input, 0, 1,
                                            axis=1)
        self.validationSet.input = np.insert(self.validationSet.input, 0, 1,
                                              axis=1)
        self.testSet.input = np.insert(self.testSet.input, 0, 1, axis=1)


    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self._get_layer(0)

    def _get_output_layer(self):
        return self._get_layer(-1)

    def _feed_forward(self, inp, idx=None):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        temp_outp = None
        temp_inp = inp
        # if idx == 1:
        #     print("input:", inp)
        for layer in self.layers:
            temp_outp = layer.forward(temp_inp)
            temp_inp = np.insert(temp_outp, 0, 1)
            # if idx == 1:
            #     print("@ weights: \n",layer.weights)
            #     print("x:\n", layer.preactivation)
            #     print("y:\n", layer.outp)
        return temp_outp
        
    def _compute_error(self, target, outp, idx=None):
        """
        Compute the total error of the network (error terms from the output layer)

        Parameters
        -------
        target: ndarray
            a numpy array (1, nNeurons_final_layer)
        outp: ndarray
            a numpy array (1, nNeurons_final_layer)
        Returns
        -------
        ndarray :
            #a numpy array (1,nOut) containing the output of the layer
            now returns ndarray (nNeurons_final_layer,1) loss value
        """
        self.loss_value = self.loss.calculateDerivative(target, outp)
        if np.max(np.abs(self.loss_value)) > 10.0:
            print("idx:", idx)
            print("target:\n",target)
            print("output:\n",outp)
            print("loss value:\n", self.loss_value)
            raise ValueError
        deltas = self.layers[-1].computeDerivative(self.loss_value, np.ones(()))
        for l in range(len(self.layers)-2, -1, -1):
            next_weights_biasless = self.layers[l+1].weights[1:, :]
            deltas = self.layers[l].computeDerivative(deltas, next_weights_biasless)
        return self.loss_value

    def _update_weights(self, learningRate):
        """
        Update the weights of the layers by propagating back the error
        """
        for layer in self.layers:
            layer.updateWeights(learningRate)
        
    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))
                #print("feed_forward", self._feed_forward(self.testSet.input[0]).T, "->",
                #      self.classify(self.testSet.input[0]), "=?=", self.testSet.label[0].T)

            self._train_one_epoch()

            if verbose:
                valset_evaluated = self.evaluate(self.validationSet)
                trainset_evaluated = self.evaluate(self.trainingSet)
                try:
                    valset_remove_onehot = np.argmax(self.validationSet.label, axis=1)
                    trainset_remove_onehot = np.argmax(self.trainingSet.label, axis=1)
                except ValueError:
                    trainset_remove_onehot = self.trainingSet.label
                    valset_remove_onehot = self.validationSet.label
                valset_accuracy = accuracy_score(valset_remove_onehot, valset_evaluated)
                trainset_accuracy = accuracy_score(trainset_remove_onehot, trainset_evaluated)
                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                self.performances.append(valset_accuracy)
                self.performances_trainset.append(trainset_accuracy)
                print("Accuracy on validation: {0:.2f}% on training: {1:.2f}%; loss: {2:.3f}"
                      .format(valset_accuracy * 100, trainset_accuracy*100, self.losses[-1]))
                print("-----------------------------")

    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """
        import time
        start_time = time.time()
        epoch_losses = []
        # img has already a 1 added for the bias.
        for img, label, idx in zip(
                self.trainingSet.input,
                self.trainingSet.label,
                range(len(self.trainingSet.input))):

            if idx % 100 == 0 and idx != 0:
                print("Datapoint", idx, "/", len(self.trainingSet.input),
                      "Mean loss:", np.mean(epoch_losses),
                      "Elapsed Time: {0:.2f}s".format(time.time()-start_time)
                      )

            # Do a forward pass to calculate the output and the error
            outp = self._feed_forward(img, idx=idx)

            # Compute the derivatives w.r.t to the error
            # Please note the treatment of nextDerivatives and nextWeights
            # in case of an output layer
            #self.layer.computeDerivative(self.loss.calculateDerivative(
            #                             label,self.layer.outp), 1.0)
            try:
                epoch_losses.append(self._compute_error(label, outp, idx=idx))
                self._update_weights(self.learningRate)
            except FloatingPointError:
                print("encountered overflow on index", idx, "output is", outp)
                continue

            if idx % 100 == 0:
                pass
        self.losses.append(np.mean(epoch_losses))


    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        outp = self._feed_forward(test_instance)

        # check if there is a one-hot encoding used
        if len(outp) > 1:
            return np.argmax(outp)
        else:
            return 1 if outp > 0.5 else 0

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
