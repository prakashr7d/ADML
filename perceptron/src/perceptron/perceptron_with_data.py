
class perceptron_with_data:
    def __init__(self):
        pass

    def fit(self):
        cleaned_data = data.dropna()
        standard_scaler = StandardScaler()
        standard_scaler.fit(data)