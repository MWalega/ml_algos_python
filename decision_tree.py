import numpy as np
from collections import Counter


def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:

    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feature_idxs = np.random.choice(n_features, self.n_features, replace=False)

        best_feat, best_thresh = self._best_criteria(X, y, feature_idxs)
        left_idxs, right_idxs = self._split(X[:,best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs,:], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs,:], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feature_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feature_idx in feature_idxs:
            X_col = X[:,feature_idx]
            thresholds = np.unique(X_col)
            for threshold in thresholds:
                gain = self._information_gain(y, X_col, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_col, split_thresh):
        # parent Entropy
        parent_entropy = entropy(y)

        # generate split
        left_idxs, right_idxs = self._split(X_col, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        # weighted average child Entropy
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l ,e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_col, split_thresh):
        left_idxs = np.argwhere(X_col <= split_thresh).flatten()
        right_idxs = np.argwhere(X_col >= split_thresh).flatten()
        return left_idxs, right_idxs

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common