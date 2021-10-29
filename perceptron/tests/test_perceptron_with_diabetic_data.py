from pathlib import Path

import pandas as pd
from pytest import fixture
from sklearn.preprocessing import StandardScaler


@fixture
def data():
    data = pd.read_csv(Path('data') / "diabetes.csv")
    return data


def test_diabetics_data(data: pd.DataFrame):
    data = dat

