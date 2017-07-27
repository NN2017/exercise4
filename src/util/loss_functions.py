# -*- coding: utf-8 -*-


"""
Loss functions.
"""

import numpy as np

from abc import ABCMeta, abstractmethod, abstractproperty


class Error:
    """
    Abstract class of an Error
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def errorString(self):
        pass

    @abstractmethod
    def calculateError(self, target, output):
        # calculate the error between target and output
        pass
        
    @abstractmethod
    def calculateDerivative(self, target, output):
        # calculate the error between target and output
        pass


class AbsoluteError(Error):
    """
    The Loss calculated by the number of differences between target and output
    """
    def errorString(self):
        self.errorString = 'absolute'

    def calculateError(self, target, output):
        # It is the numbers of differences between target and output
        return abs(target - output)
        
    def calculateDerivative(self, target, output):
        pass


class DifferentError(Error):
    """
    The Loss calculated by the number of differences between target and output
    """
    def errorString(self):
        self.errorString = 'different'

    def calculateError(self, target, output):
        # It is the numbers of differences between target and output
        return target - output
    
    def calculateDerivative(self, target, output):
        return -1


class MeanSquaredError(Error):
    """
    The Loss calculated by the mean of the total squares of differences between
    target and output.
    """
    def errorString(self):
        self.errorString = 'mse'

    def calculateError(self, target, output):
        # MSE = 1/n*sum (i=1 to n) of (target_i - output_i)^2)
        n = np.asarray(target).size
        return (1.0/n) * np.sum((target - output)**2)
    
    def calculateDerivative(self, target, output):
        # MSEPrime = -n/2*(target - output)
        n = np.asarray(target).size
        return (2.0/n) * (output - target)


class SumSquaredError(Error):
    """
    The Loss calculated by the sum of the total squares of differences between
    target and output.
    """
    def errorString(self):
        self.errorString = 'sse'

    def calculateError(self, target, output):
        # SSE = 1/2*sum (i=1 to n) of (target_i - output_i)^2)
        return 0.5*np.sum((target - output)**2)
        
    def calculateDerivative(self, target, output):
        # SSEPrime = -(target - output)
        return output - target


class BinaryCrossEntropyError(Error):
    """
    The Loss calculated by the Cross Entropy between binary target and
    probabilistic output (BCE)
    """
    def errorString(self):
        self.errorString = 'bce'

    def calculateError(self, target, output):
        # type: (np.ndarray, np.ndarray) -> np.ndarray
        """
        :param target: ndarray (nNeurons_final_layer, 1)
        :param output: ndarray (nNeurons_final_layer, 1)
        :return: ndarray (1,1)

        BCE always positive, tends towards 0 for better match
        """
        assert max(output) <= 1.0 and min(output) > 0.0, "output not in (0,1]"+str(output)
        n = np.asarray(target).size
        return -np.sum(target*np.log(output) + (1-target)*np.log(1-output))/n
        
    def calculateDerivative(self, target, output, debug=False, clip=1):
        # type: (np.ndarray, np.ndarray, bool, int) -> np.ndarray
        """
        :param target: ndarray (nNeurons_final_layer, 1)
        :param output: ndarray (nNeurons_final_layer, 1)
        :return: ndarray (nNeurons_final_layer,1)
            returns the derivative of the loss for each of the outputs
            from the network's final layer dE/do_j

        BCE always positive, tends towards 0 for better match
        """

        assert max(output) <= 1.0 and min(output) > 0.0, "output not in (0,1]"+str(output)
        # BCEPrime = -target/output + (1-target)/(1-output)
        if debug:
            debug = 0
        np.seterr(over="raise", divide="raise", invalid='raise')

        real_bce = -target/output + (1-target)/(1-output)
        return np.clip(real_bce, -clip, clip)
 

class CrossEntropyError(Error):
    """
    The Loss calculated by the more general Cross Entropy between two
    probabilistic distributions.
    """
    def errorString(self):
        self.errorString = 'crossentropy'

    def calculateError(self, target, output):
        pass
        
    def calculateDerivative(self, target, output):
        pass
