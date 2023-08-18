import numpy as np

class SVM():
    def __init__(self, learning_rate = 0.001, regularisation_lambda = 1, n_iters = 1000):
        self.learning_rate = learning_rate
        self.regularisation_lambda = regularisation_lambda
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.J_history = []

    def predict(self, X_i):
        return (np.dot(self.w, X_i.T) - self.b)

    def cost(self, X, y):
        prediction = self.predict(X)
        loss = y*prediction
        return ((self.regularisation_lambda * ((self.w)**2)) + max(0, 1 - loss))

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        """"
          Training the Linear SVM Algorithm with given data
        """

        for i in range(self.n_iters):
            prediction  = self.predict(X)
            self.J_history.append(self.cost(X,y))
            if y*prediction >= 1:
                dJW = 2 * self.regularisation_lambda * self.w
                dJB = 0
            else:
                dJW = 2 * (self.regularisation_lambda * self.w) + np.dot(y, X.T)
                dJB = y

            self.w = self.w - self.learning_rate * dJW
            self.b = self.b - self.learning_rate * dJB

          