import numpy as np

class DecisionStump:
  def __init__(self) -> None:
    self.polarity = 1
    self.threshold = None
    self.alpha = None
    self.feature_idx = None

  def predict(self, X):
    n_samples = X.shape[0]
    X_column = X[:, self.feature_idx]
    predictions = np.ones(n_samples)

    if self.polarity == 1:
      predictions[X_column < self.threshold] = -1
    else:
      predictions[X_column > self.threshold] = -1
    
    return predictions
  
class AdaBoost:
  def __init__(self, n_clf=5) -> None:
    self.n_clf = n_clf

  def fit(self, X, y):
    n_samples, n_features = X.shape

    # Initialize weights to 1/N
    w = np.full(n_samples, (1/n_samples))

    self.clfs = []

    #iterate through the classifiers
    for _ in range(self.n_clf):

      clf = DecisionStump()
      min_error = float('inf')

      for feature_idx in range(n_features):
        X_column = X[:, feature_idx]
        thresholds = np.unique(X_column)

        for threshold in thresholds:
          #predict with polarity 1
          p = 1
          predictions = np.ones(n_samples)
          predictions[X_column < threshold] = -1

          misclassified = w[y != predictions]
          error = sum(misclassified)

          if error > 0.5:
            error = 1 - error
            p = -1

          if error < min_error:
            min_error = error
            clf.polarity = p
            clf.threshold = threshold
            clf.feature_idx = feature_idx

      #calculate alpha
      EPSILON = 1e-10
      clf.alpha = 0.5 * np.log((1.0 - min_error + EPSILON) / (min_error + EPSILON))

      #calculate predictions
      predictions = clf.predict(X)
      w = w * np.exp(-clf.alpha * y * predictions)
      w = w / np.sum(w)

      self.clfs.append(clf)

  def predict(self, X):
    clf_predictions = [clf.alpha * clf.predict(X) for clf in self.clfs]
    y_pred = np.sum(clf_predictions, axis=0)
    return np.sign(y_pred)


if __name__ == "__main__":
    # Imports
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import matplotlib.pyplot as plt

    X, y = datasets.make_blobs(
        n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
    )
    y = np.where(y == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(y)

    clf = AdaBoost()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_train_pred = clf.predict(X_train)
    print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred)}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Confusion Matrix {confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report {classification_report(y_test, y_pred)}")