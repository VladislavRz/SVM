import numpy as np


def add_bias_feature(a):
    a_extended = np.zeros((a.shape[0],a.shape[1]+1))
    a_extended[:, :-1] = a
    a_extended[:, -1] = int(1)
    return a_extended


class SVM:
    def __init__(self, step=0.01, epochs=200):
        self.epochs = epochs
        self.step = step
        self.weights = None

    def fit(self, data, target):

        data = add_bias_feature(data)
        self.weights = np.random.normal(loc=0, scale=0.05, size=data.shape[1])

        for epoch in range(self.epochs):
            for i, x in enumerate(data):
                margin = target[i]*np.dot(self.weights, data[i])

                if margin >= 1:
                    self.weights = self.weights - self.step * self.weights / self.epochs
                else:
                    self.weights = self.weights + self.step * (target[i] * data[i] - self.weights / self.epochs)

    def predict(self, x: np.array) -> np.array:
        y_pred = []
        x_extended = add_bias_feature(x)
        for i in range(len(x_extended)):
            y_pred.append(np.sign(np.dot(self.weights, x_extended[i])))
        return np.array(y_pred)
