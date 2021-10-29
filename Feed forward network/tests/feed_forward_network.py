from dataclasses import dataclass
from typing import List

from pytest import fixture
from data.test_datas import EPOCHS, input_features, WEIGHTS, BIAS, output
from feedforwardnetwork import FeedForwardNetwork


@dataclass
class TrainingData:
    epochs: int
    input_features: List[List[int]]
    weights: List[int]
    bias: int
    output: List[int]


@fixture
def training_data():
    return TrainingData(epochs=EPOCHS, input_features=input_features, weights=WEIGHTS, bias=BIAS, output=output)


def test_feed_forward(training_data: TrainingData):
    network = FeedForwardNetwork()
    network.fit(input_var=training_data.input_features, output=training_data.output,
                bias=training_data.bias, weights=training_data.weights)
    network.train(epochs=training_data.epochs, lr=0.1)
    assert network.predict([0.1, 0.2]) == 1

