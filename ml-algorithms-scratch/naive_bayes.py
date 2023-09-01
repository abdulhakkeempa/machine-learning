import numpy as np

class NaiveBayes:
    
    def fit(self,X, y):
        """
          Fit the training data
          
          Args: 
              X (np.array): Input array
              y (np.array): Target array

          Returns:
              None

        """
        num_samples, num_features = X.shape
        self._classes = np.unique(y)
        num_classes = len(self._classes)

        self._mean = np.zeros((num_classes, num_features), dtype=np.float64)
        self._var = np.zeros((num_classes, num_features), dtype=np.float64)
        self._priors = np.zeros(num_classes, dtype=np.float64)

        for idx, cols in enumerate(self._classes):
            X_rows = X[y==cols]
            self._mean[idx, :] = np.mean(X_rows, axis=0, keepdims=True)
            self._var[idx, :] = np.var(X_rows, axis=0, keepdims=True)
            self._priors[idx] = X_rows.shape[0] / num_samples


    def predict(self, X):
        """
        Predict the class labels
        
        Args:
            X (np.array): Input array

        Returns:
            [np.array]: Class labels
        """

        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    def _predict(self, x):
        """"
        Predict the class label
        
        Args:
            x (np.array): Input array

        Returns:
            [np.array]: Class label
        """
        posteriors = []
        for idx, cols in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posteriror = np.log(self._pdf(idx, x)).sum()
            posterior = prior + posteriror
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, index, X):
        """"
          Calculate the probability density function with gaussian distribution
        
        Args:
            index (int): Index of the row in the mean and variance array
            X (np.array): Input array

        Returns:
            [np.float64]: Probability density function
        """
        
        mean = self._mean[index]
        var = self._var[index]
        numerator = np.exp(-(X-mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator/denominator
    



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

    clf = NaiveBayes()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Confusion Matrix {confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report {classification_report(y_test, y_pred)}")
