import numpy as np

class LogisticRegression:
  def __init__(self, alpha=0.01, epochs=100):
    self.w = 0
    self.b = 0
    self.alpha = alpha
    self.epochs = epochs

  def predict(self, X: np.ndarray):
    probabilities = X.dot(self.w) + self.b
    return self.sigmoid(probabilities)

  def sigmoid(self,prediction: np.ndarray):
    z = 1/(1+np.exp(-prediction))
    return z

  def computeCost(self, X: np.ndarray, y:np.ndarray):
    m = X.shape[0] #no of rows of X
    prediction = X.dot(self.w) + self.b #making prediction with weights
    prediction = self.sigmoid(prediction) #maping the prediction to 0 or 1
    cost = np.sum((y*np.log(prediction))+ ((1-y)*np.log(1-prediction))) #computing the losses
    cost = (1/m) * -cost  #computing the total cost
    return cost

  def fit(self, X: np.ndarray, y:np.ndarray):
    y = y.reshape(-1,1)
    m = X.shape[1]
    self.w = np.zeros((m,1))

    for i in range(self.epochs):
      cost = self.computeCost(X,y)
      if i%100 ==0:
        print(f"Epoch: {i+1}, Loss: {cost}")
      prediction = self.predict(X)
      error = prediction - y
      gradient = np.dot(X.transpose(),error)
      self.w -= self.alpha * 1/m * gradient
      self.b -= self.alpha * 1/m * np.sum(error)

    return (self.w,self.b)
  
if __name__ == "__main__":
  from sklearn.metrics import accuracy_score

  X = np.array([[0, 0],  # Input: [0, 0], Output: 0
                [0, 1],  # Input: [0, 1], Output: 0
                [1, 0],  # Input: [1, 0], Output: 0
                [1, 1]]) # Input: [1, 1], Output: 1

  y = np.array([0, 0, 0, 1])

  lr = LogisticRegression(alpha=1, epochs=10)
  lr.fit(X, y)
  predictions = lr.predict(X)
  print(predictions) #probabilities of being 1
  # Apply a threshold of 0.5 to get binary class labels
  binary_predictions = (predictions >= 0.5).astype(int)
  print(binary_predictions)

  print("Logistic Regression classification accuracy", accuracy_score(y, binary_predictions))