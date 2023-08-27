from collections import Counter
import numpy as np

def entropy(y):
    """"
      Caculating the entropy of the function.

      Input :
        y - np.array() - Output Target Vector

      Returns :
        entropy - int - Entropy of the function for the given data
    """
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
    return entropy


class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, *, value = None) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_sample=2, max_depth=100, n_feats=None) -> None:
        self.min_sample = min_sample
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None


    def fit(self,X, y):
        """"
          Training the Decision Tree Algorithm with given data.

          Input :
            X - np.array() - Input Feature Vectors
            y - np.array() - Output Target Vector

          Returns :
            None
        """
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)


    def _grow_tree(self, X,y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        #stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_sample):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        

        feature_idx = np.random.choice(n_features, self.n_feats, replace=False)
        print(feature_idx)

        # greedy search
        best_feat, best_threshold = self._best_criteria(X, y, feature_idx)
        left_idx, right_idx = self._split(X[:, best_feat], best_threshold)
        left = self._grow_tree(X[left_idx,:], y[left_idx], depth+1)
        right = self._grow_tree(X[right_idx,:], y[right_idx], depth+1)
        return Node(best_feat, best_threshold, left, right)

    def _best_criteria(self, X,y, feature_idx):
        best_gain = -1
        split_idx, split_threshold = None, None
        for feature in feature_idx:
            X_column = X[:, feature]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_threshold = threshold

        return split_idx, split_threshold
    
    def _information_gain(self, y, X_column, split_threshold):
        #parent entropy
        parent_entropy = entropy(y)

        #generate split
        left_idx, right_idx = self._split(X_column, split_threshold)

        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0
        
        #weighted average child
        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        e_l, e_r = entropy(y[left_idx]), entropy(y[right_idx])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r
        ig = parent_entropy - child_entropy
        return ig


    def _split(self, X_column, split_threshold):
        left_indx = np.argwhere(X_column <= split_threshold).flatten()
        right_indx = np.argwhere(X_column > split_threshold).flatten()
        return left_indx, right_indx
        
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        elif x[node.feature] > node.threshold:
            return self._traverse_tree(x, node.right)
        



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

    tree = DecisionTree(max_depth=10)
    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Confusion Matrix {confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report {classification_report(y_test, y_pred)}")