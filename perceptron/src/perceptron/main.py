import random
from random import choice
from typing import List

import numpy
import numpy as np
from matplotlib import pyplot as plt
from pylab import ylim, plot

class perceptron_model:
    def __init__(self):
        self.trainig_data = None
        self.error_in_every_epoch = []
        self.weights = None
    @staticmethod
    def __step_function(x : numpy.array) -> int:
        return -1 if x < 0 else 1

    def plot(self):
        ylim([-2.5, 2.5])
        plot(self.error_in_every_epoch)
        plt.ylabel('error')
        plt.xlabel('epoch')
        plt.show()

    def evaluate(self, training_data) -> float:
        count = 0
        for data in training_data:
            test = data[0]
            target = data[1]
            print(">input{} target:{} predicted:{}".format(test, target, self.__step_function(np.dot(test, self.weights))))
            if target == self.__step_function(np.dot(test, self.weights)):
                count += 1
        return count/len(training_data)

    def fit(self, training_data: List[List[int]], epochs: int = 100, lr : float =0.1) -> object:
        self.trainig_data = training_data
        weights = np.random.rand(3)
        for _ in range(epochs):
            feature, expected = choice(training_data)
            result = np.dot(weights, feature)
            error = expected - self.__step_function(result)
            self.error_in_every_epoch.append(error)
            weights += lr * np.dot(error, feature)
        self.weights = weights
        print(f"Final Weight: {weights}")
        self.plot()


