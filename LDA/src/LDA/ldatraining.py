from dataclasses import dataclass
from typing import Optional

import numpy
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@dataclass
class TrainingSet:
    x_train: numpy.array
    y_train: numpy.array
    x_test: numpy.array
    y_test: numpy.array


class LdaTraining:
    def __init__(self):
        self.training_set: Optional[TrainingSet] = None
        self.model = None
        self.accuracy: float = 1.0

    @staticmethod
    def __dataset_split(x: pd.DataFrame, y: pd.Series) -> TrainingSet:
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=30, test_size=0.3)
        set_of_data = TrainingSet(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        return set_of_data

    def fit(self, feature: pd.DataFrame, target: pd.Series, no_of_component: int) -> object:
        ld = LinearDiscriminantAnalysis(n_components=no_of_component)
        training_set = self.__dataset_split(feature, target)
        ld.fit(training_set.x_train, training_set.y_train)
        training_set.x_train = ld.transform(training_set.x_train)
        training_set.x_test = ld.transform(training_set.x_test)
        self.training_set = training_set
        return self

    def __performance_metrics(self) -> float:
        prediction = self.model.predict(self.training_set.x_test)
        self.accuracy = accuracy_score(y_true=self.training_set.y_test, y_pred=prediction, )
        return self.accuracy

    def evaluate(self):
        return self.accuracy

    def train(self, model: object) -> object:
        model.fit(self.training_set.x_train, self.training_set.y_train)
        self.model = model
        self.__performance_metrics()
        return self

