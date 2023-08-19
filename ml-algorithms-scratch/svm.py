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
        approx = np.dot(self.w, X_i.T) - self.b
        return np.sign(approx)


    def cost(self, X, y):
        """"
          Caculating the loss of the function.

          Input :
            X - np.array() - Input Feature Vectors
            y - np.array() - Output Target Vector

          Returns :
            total_loss - int - Total lost of the function for the given data
        """
        prediction = np.dot(self.w, X.T) - self.b
        prediction = y*prediction
        hinge_loss = np.maximum(0, 1 - prediction)
        loss = np.mean(hinge_loss)  
        regularised_loss = np.sum(self.regularisation_lambda * ((abs(self.w))**2))
        return (loss+regularised_loss)

    def fit(self, X, y):
        """"
          Training the Linear SVM Algorithm with given data.

          Input :
            X - np.array() - Input Feature Vectors
            y - np.array() - Output Target Vector

          Returns :
            J_History - list - Collection of Cost during each iterations
        """
        self.w = np.zeros(X.shape[1])  #setting initial weights for the features as zeros
        self.b = 0      #setting the bias value to zero

        for i in range(self.n_iters):
            loss = self.cost(X,y)
            print(f"{i} Epochs : {loss:.2f} Loss")
            self.J_history.append(loss)

            for idx,x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b)
                if condition >=1:
                    dJW = 2 * self.regularisation_lambda * self.w
                    dJB = 0
                else:
                    dJW = 2 * (self.regularisation_lambda * self.w) - np.dot(y[idx], x_i)
                    dJB = y[idx]

            self.w = self.w - self.learning_rate * dJW
            self.b = self.b - self.learning_rate * dJB

        return self.J_history

          