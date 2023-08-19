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

            # print(self.w, self.b)

        return self.J_history


# Testing
if __name__ == "__main__":
    # Imports
    from sklearn import datasets
    import matplotlib.pyplot as plt

    X, y = datasets.make_blobs(
        n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=40
    )
    y = np.where(y == 0, -1, 1)

    clf = SVM(regularisation_lambda=0.01)
    history = clf.fit(X, y)



    print(clf.w, clf.b)

    def visualise_loss(history):
        plt.plot(history)
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.show()


    def visualize_svm():
        def get_hyperplane_value(x, w, b, offset):
            return (-w[0] * x + b + offset) / w[1]

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)

        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        x1_1 = get_hyperplane_value(x0_1, clf.w, clf.b, 0)
        x1_2 = get_hyperplane_value(x0_2, clf.w, clf.b, 0)

        x1_1_m = get_hyperplane_value(x0_1, clf.w, clf.b, -1)
        x1_2_m = get_hyperplane_value(x0_2, clf.w, clf.b, -1)

        x1_1_p = get_hyperplane_value(x0_1, clf.w, clf.b, 1)
        x1_2_p = get_hyperplane_value(x0_2, clf.w, clf.b, 1)

        ax.plot([x0_1, x0_2], [x1_1, x1_2], "y--")
        ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k")
        ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k")

        x1_min = np.amin(X[:, 1])
        x1_max = np.amax(X[:, 1])
        ax.set_ylim([x1_min - 3, x1_max + 3])

        plt.show()

    visualize_svm()
    visualise_loss(history)