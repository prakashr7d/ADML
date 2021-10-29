from pathlib import Path

import pandas as pd
import pytest

from LDA.ldatraining import LdaTraining
from src import __version__
from pytest import fixture
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


def test_version():
    assert __version__ == '0.1.0'


@fixture
def training_data():
    return pd.read_csv(Path('data/iris.csv'))


@pytest.mark.parametrize(
    "no_of_component,accuracy",
    [
        (1, 98.0),
        (2, 96.0)
    ]
)
def test_lda_iris_with_logistic(accuracy: float, no_of_component: int, training_data: pd.DataFrame):
    ld = LdaTraining()
    feature = training_data[training_data.columns[1:-1]]
    target = training_data[training_data.coumns[-1]]
    ld.fit(feature=feature, target=target, no_of_component=no_of_component)
    ld.train(model=LogisticRegression(max_iter=50))
    assert float("{:.0%}".format(ld.evaluate())[:-1]) == accuracy
    print(f"Accuracy of the logistic regression with accuracy: {ld.accuracy} and no_of_component for the LDA: {no_of_component}")


@pytest.mark.parametrize(
    "no_of_component,accuracy",
    [
        (1, 98.0),
        (2, 96.0)
    ]
)
def test_lda_iris_with_naive(accuracy: float, no_of_component: int, training_data: pd.DataFrame):
    ld = LdaTraining()
    feature = training_data[training_data.columns[1:-1]]
    target = training_data[training_data.columns[-1]]
    ld.fit(feature=feature, target=target, no_of_component=no_of_component)
    ld.train(model=GaussianNB())
    assert float("{:.0%}".format(ld.evaluate())[:-1]) == accuracy
    print(f"Accuracy of the naive byes with accuracy: {ld.accuracy}"
          f"and no_of_component for the LDA: {no_of_component}")


@pytest.mark.parametrize(
    "no_of_component,accuracy",
    [
        (1, 98.0),
        (2, 96.0)
    ]
)
def test_lda_iris_with_random_forest(accuracy: float, no_of_component: int, training_data: pd.DataFrame):
    ld = LdaTraining()
    feature = training_data[training_data.columns[1:-1]]
    target = training_data[training_data.columns[-1]]
    ld.fit(feature=feature, target=target, no_of_component=no_of_component)
    ld.train(model=RandomForestClassifier())
    assert float("{:.0%}".format(ld.evaluate())[:-1]) == accuracy
    print(f"Accuracy of the random forest with accuracy: {ld.accuracy}"
          f"and no_of_component for the LDA: {no_of_component}")


