from perceptron.main import perceptron_model
from src.perceptron import __version__
import pytest
from data.training_data import Q1, Q2


def test_version():
    assert __version__ == '0.1.0'


@pytest.mark.parametrize(
    "training_data",
    [Q1, Q2]
)
def test_perceptron(training_data):
    perceptron = perceptron_model()
    perceptron.fit(Q1)
    accuracy = perceptron.evaluate(training_data)
    print(f"Accuracy of the data is: {accuracy}")

