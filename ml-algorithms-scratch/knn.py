import numpy as np

class KNN():
  def __init__(self,X = None, y= None,k = 5):
    self.X = X
    self.y = y
    self.k = k

  def fit(self, X, y):
    self.X = X
    self.y = y

  def predict(self, X_i):
    distances = np.sqrt(np.sum((self.X - X_i)**2, axis=1))
    sorted_indices = np.argsort(distances)
    y_sorted = self.y[sorted_indices]
    k_nearest = y_sorted[:self.k]
    values, count = np.unique(k_nearest, return_counts=True)
    return values[np.argmax(count)]
    



if __name__ == "__main__":
    # Imports
    from sklearn import datasets
    import matplotlib.pyplot as plt

    X, y = datasets.make_blobs(
        n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
    )
    y = np.where(y == 0, -1, 1)
    print(y)
    clf = KNN(k=5)
    clf.fit(X, y)


    # Define the input data point for prediction
    input_data_point = np.array([1, 2])

    # Make a prediction
    prediction = clf.predict(input_data_point)
    
    # Create a colormap for the classes
    colormap = np.array(['b', 'r'])

    # Plot the training data with different colors for different labels
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', s=50, label='Training Data')

    # Plot the input data point
    plt.scatter(input_data_point[0], input_data_point[1], c='black', marker='*', s=100, label='Input Data Point')

    # Annotate the prediction
    plt.annotate(f'Predicted Class: {prediction}', xy=(input_data_point[0], input_data_point[1]), xytext=(2, 2), textcoords='offset points', color='red')

    # Set plot labels and legend
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='best')
    plt.title("K-Nearest Neighbour Algorithm")

    # Show the plot
    plt.axis('equal')  # Ensure aspect ratio is equal for a circular KNN plot
    plt.show()

