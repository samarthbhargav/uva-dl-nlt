from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=90)
    
    def fit(self, data):
        self.classifier.fit(*data)
    
    def predict(self, data):
        return self.classifier.predict(data)
