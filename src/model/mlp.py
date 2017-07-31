from __future__ import print_function
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier
from sklearn.metrics import accuracy_score
from util.loss_functions import *
from report.evaluator import Evaluator
import time

import numpy as np


class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, hypes=[30], inputWeights=None,
                 outputTask='classification', outputActivation='softmax',
                 loss='bce', learningRate=0.01, epochs=50, cost=None, name="", clip=None):

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
        self.clip = clip

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
        self.loss_deriv = None
        self.name = name
        self.evaluator = Evaluator()

        # Build up the network from specific layers
        self.layers = []
        self.hypes = hypes

        # Input layer (a hidden layer)
        assert len(hypes) >= 1, "there has to be >1 hidden layer"
        inputActivation = "sigmoid"
        self.layers.append(LogisticLayer(self.trainingSet.input.shape[1], hypes[0],
                           None, inputActivation, False, name="layer0"))
        # Hidden Layer
        hiddenActivation = "sigmoid"

        #it works if we subtract one from the next dimension.
        for l in range(1, len(hypes)):
            self.layers.append(LogisticLayer(hypes[l-1]-1, hypes[l], None,
                           hiddenActivation, False, name="layer"+str(l)))

        # Output layer
        try:
            outlayer_size = len(self.trainingSet.label[0])
        except TypeError:
            outlayer_size = 1
        assert outlayer_size == 10
        self.layers.append(LogisticLayer(hypes[-1]-1, outlayer_size,
                           None, outputActivation, True, name="final"))

        self.inputWeights = inputWeights
        if inputWeights is not None:
            print("Loading in Input weights")
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
        #self.testSet.label = np.argmax(self.testSet.label, axis=1)


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
        # """

        temp_outp = inp
        for layer in self.layers:
            layer.forward(temp_outp)
            temp_outp = layer.outp
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
        self.loss_deriv = self.loss.calculateDerivative(target, outp, clip=self.clip)
        self.loss_value = self.loss.calculateError(target, outp)
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

            start_time = time.time()
            epoch_losses = []
            # img has already a 1 added for the bias.
            for img, label, idx in zip(
                    self.trainingSet.input,
                    self.trainingSet.label,
                    range(len(self.trainingSet.input))):

                if idx % 1000 == 0 and idx > 1:
                    #   self.learningRate *= 0.9
                    test_accuracy = self._accuracy(self.testSet)
                    print("Datapoint", idx, "/", len(self.trainingSet.input),
                          "Elapsed Time: {0:.2f}s".format(time.time() - start_time)
                          # ,"Learning Rate: {0:.4f}".format(self.learningRate)
                          ,"Testset Acc: {0:.2f}%".format(test_accuracy*100)
                          ,"Mean Epoch loss: {0:.3f}".format(np.mean(epoch_losses))
                          )
                # Do a forward pass to calculate the output and the error
                outp = self._feed_forward(img, idx=idx)

                # Compute the derivatives w.r.t to the error
                # Please note the treatment of nextDerivatives and nextWeights
                # in case of an output layer
                # self.layer.computeDerivative(self.loss.calculateDerivative(
                #                             label,self.layer.outp), 1.0)
                try:
                    loss_value =self._compute_error(label, outp, idx=idx)
                    epoch_losses.append(loss_value)

                    deltas = self._get_output_layer().computeDerivative(self.loss_deriv, 1.0)

                    for l in range(len(self.layers)-2, -1, -1):
                         next_weights= self.layers[l+1].weights.T
                         deltas = self.layers[l].computeDerivative(deltas, next_weights)

                    self._update_weights(self.learningRate)
                except FloatingPointError:
                    print("encountered overflow on index", idx, "output is", outp)
                    continue

                if idx % 100 == 0:
                    pass
            self.losses.append(np.mean(epoch_losses))
            if verbose:
                valset_accuracy = self._accuracy(self.validationSet)
                trainset_accuracy = self._accuracy(self.trainingSet)

                # # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                self.performances.append(valset_accuracy)
                self.performances_trainset.append(trainset_accuracy)
                print("Accuracy on validation: {0:.2f}% on training: {1:.2f}%; loss: {2:.3f}"
                       .format(valset_accuracy * 100, trainset_accuracy * 100, self.losses[-1]))
                print("-----------------------------")

    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        outp = self._feed_forward(test_instance)
        assert len(outp) == 10

        # check if there is a one-hot encoding used
        if len(outp) > 1:
            return [1 if i == np.argmax(outp) else 0 for i in range(len(outp))]
        else:
            return 1 if outp > 0.5 else 0

    def evaluate(self, test=None, oneHot=True):
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
        result = map(self.classify, test)
        if not oneHot:
            result = np.argmax(result, axis=1)
        return list(result)

    def _accuracy(self, testSet):
        return accuracy_score(
            list(map(lambda x:np.argmax(x), testSet.label)),
            list(map(lambda x:np.argmax(x), self.evaluate(testSet)))
        )
    def __del__(self):
        # Remove the bias from input data
        self.trainingSet.input = np.delete(self.trainingSet.input, 0, axis=1)
        self.validationSet.input = np.delete(self.validationSet.input, 0,
                                              axis=1)
        self.testSet.input = np.delete(self.testSet.input, 0, axis=1)
