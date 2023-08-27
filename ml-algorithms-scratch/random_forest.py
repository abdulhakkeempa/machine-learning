import numpy as np
from collections import Counter
from decision_tree import DecisionTree

def bootstrap_sample(X,y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    print(idxs)
    return X[idxs], y[idxs]

def most_common_label(y):
    counter = Counter(y)
    most_common_label = counter.most_common(1)[0][0]
    return most_common_label

class RandomForest:
    def __init__(self, n_trees=2, min_sample_split=2, max_depth=100, n_features=None) -> None:
        self.n_trees = n_trees
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.trees = []
        for _ in range(self.n_features):
            tree = DecisionTree(min_sample=self.min_sample_split, max_depth=self.max_depth)
            X_sample, y_sample = bootstrap_sample(X,y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)


    def predict(self,X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)
    


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

    tree = RandomForest(max_depth=10)
    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Confusion Matrix {confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report {classification_report(y_test, y_pred)}")