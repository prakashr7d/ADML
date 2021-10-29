from typing import List

import numpy
import numpy as np


class FeedForwardNetwork:
    def __init__(self):
        self.input = None
        self.bias = None
        self.weights = None
        self.output = None
        self.epoch_format = {"weights": [], "inputs_for_the_network": []}
        self.Training_manager = {}

    def fit(self, input_var: List[List[int]], output: List[int], bias: int, weights: List[int]):
        self.input = np.array(input_var)
        self.weights = np.array(weights)
        self.output = np.array(output)
        self.bias = np.array(bias)

    @staticmethod
    def __add_array(array: numpy.array):
        total = np.zeros(array.shape[0])
        for single_list in array:
            total += single_list
        return total

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def predict(self, feature):
        total = 0
        for value, weight in zip(feature, self.weights[1:]):
            total += (value * weight)
        return 1 if self.sigmoid(total) >= 0.5 else 0

    def train(self, epochs: int, lr: float):
        for epoch in range(epochs):
            self.Training_manager[str(epoch)] = self.epoch_format.copy()
            overall_output = numpy.array([])
            for feature, ground_truth in zip(self.input, self.output):
                feature = numpy.insert(feature, 0, self.bias)
                for weight in self.weights:
                    output = np.array([0, 0, 0])
                    output = feature.dot(weight) + output
                prediction = self.predict(output)
                error = prediction - ground_truth
                self.weights = self.weights + (lr * error**2)
                np.append(overall_output, output)
            print(f"epoch: {epoch+1} error: {error}")
            self.input = self.__add_array(overall_output)[1:]
            self.Training_manager[str(epoch)]["input"] = self.input
