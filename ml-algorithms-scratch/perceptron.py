import numpy as np

#single layer perceptron
class Perceptron:

  def __init__(self, lr, epochs) -> None:
    self.lr = lr
    self.epoch = epochs

  def fit(self, X, y):
    n_samples, n_features = X.shape

    # init parameters
    self.weights = np.zeros(n_features)
    self.bias = 0

    y_ = np.array([1 if i > 0 else 0 for i in y])

    for _ in range(self.epoch):
      for idx, x_i in enumerate(X):
        linear_output = np.dot(x_i, self.weights) + self.bias
        prediction = self.activation_function(linear_output)

        # Perceptron update rule
        update = self.lr * (y_[idx] - prediction)

        self.weights += update * x_i
        self.bias += update

      print(self.weights, self.bias)

  def predict(self, X):
    linear_output = np.dot(X, self.weights) + self.bias
    prediction = self.activation_function(linear_output)
    return prediction
  
  def activation_function(self, x):
    return np.where(x >= 0, 1, 0)
  

# testing

if __name__ == "__main__":
  def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

  X = np.array([[0, 0],  # Input: [0, 0], Output: 0
                [0, 1],  # Input: [0, 1], Output: 0
                [1, 0],  # Input: [1, 0], Output: 0
                [1, 1]]) # Input: [1, 1], Output: 1

  y = np.array([0, 0, 0, 1])

  p = Perceptron(lr=0.01, epochs=1000)
  p.fit(X, y)
  predictions = p.predict(X)

  print("Perceptron classification accuracy", accuracy(y, predictions))