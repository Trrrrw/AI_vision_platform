import math
from collections import Counter

import numpy as np


ALGORITHM_LABELS = {
    "knn": "KNN（K-近邻）",
    "svm": "SVM（支持向量机）",
    "nb": "朴素贝叶斯",
    "rf": "随机森林",
}


def build_classifier(algorithm_key, params):
    if algorithm_key == "knn":
        return KNNClassifier(
            n_neighbors=params["n_neighbors"],
            weight_mode=params["weight_mode"],
        )
    if algorithm_key == "svm":
        return SimpleOVRSVM(
            C=params["C"],
            kernel=params["kernel"],
            gamma=params["gamma"],
        )
    if algorithm_key == "nb":
        return GaussianNBClassifier(alpha=params["alpha"])
    return RandomForestClassifierCustom(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
    )


class KNNClassifier:
    def __init__(self, n_neighbors=5, weight_mode="uniform"):
        self.n_neighbors = n_neighbors
        self.weight_mode = weight_mode

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(y)
        return self

    def predict(self, X, batch_size=512):
        X = np.asarray(X)
        predictions = []
        for start in range(0, len(X), batch_size):
            batch = X[start : start + batch_size]
            distances = np.sqrt(np.sum((batch[:, None, :] - self.X_train[None, :, :]) ** 2, axis=2))
            nearest = np.argsort(distances, axis=1)[:, : self.n_neighbors]
            batch_predictions = []
            for row_index, neighbor_ids in enumerate(nearest):
                labels = self.y_train[neighbor_ids]
                if self.weight_mode == "distance":
                    weights = 1.0 / (distances[row_index, neighbor_ids] + 1e-9)
                    class_scores = {}
                    for label, weight in zip(labels, weights):
                        class_scores[label] = class_scores.get(label, 0.0) + weight
                    pred = sorted(class_scores.items(), key=lambda item: (-item[1], item[0]))[0][0]
                else:
                    pred = Counter(labels.tolist()).most_common(1)[0][0]
                batch_predictions.append(pred)
            predictions.extend(batch_predictions)
        return np.array(predictions)

    def get_neighbor_report(self, query_point):
        distances = np.sqrt(np.sum((self.X_train - query_point) ** 2, axis=1))
        nearest_ids = np.argsort(distances)[: self.n_neighbors]
        labels = self.y_train[nearest_ids]
        if self.weight_mode == "distance":
            weights = 1.0 / (distances[nearest_ids] + 1e-9)
            class_scores = {}
            for label, weight in zip(labels, weights):
                class_scores[label] = class_scores.get(label, 0.0) + weight
        else:
            weights = np.ones(len(nearest_ids))
            class_scores = {}
            for label, weight in zip(labels, weights):
                class_scores[label] = class_scores.get(label, 0.0) + weight
        predicted_label = sorted(class_scores.items(), key=lambda item: (-item[1], item[0]))[0][0]
        return {
            "indices": nearest_ids,
            "distances": distances[nearest_ids],
            "labels": labels,
            "weights": weights,
            "vote_scores": class_scores,
            "predicted_label": predicted_label,
        }


class GaussianNBClassifier:
    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.class_prior_ = {}
        self.theta_ = {}
        self.var_ = {}
        for cls in self.classes_:
            Xc = X[y == cls]
            self.class_prior_[cls] = float(len(Xc) + self.alpha) / float(len(X) + self.alpha * len(self.classes_))
            self.theta_[cls] = Xc.mean(axis=0)
            self.var_[cls] = Xc.var(axis=0) + self.alpha + 1e-3
        return self

    def _joint_log_likelihood(self, X):
        scores = []
        for cls in self.classes_:
            mean = self.theta_[cls]
            var = self.var_[cls]
            log_prob = -0.5 * np.sum(np.log(2.0 * math.pi * var))
            log_prob -= 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
            log_prob += math.log(self.class_prior_[cls])
            scores.append(log_prob)
        return np.vstack(scores).T

    def predict(self, X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]

    def predict_proba(self, X):
        joint = self._joint_log_likelihood(X)
        joint = joint - joint.max(axis=1, keepdims=True)
        probs = np.exp(joint)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs


class SimpleOVRSVM:
    def __init__(self, C=1.0, kernel="rbf", gamma=1.0, tol=1e-3, max_passes=8):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = []
        for cls in self.classes_:
            binary_y = np.where(y == cls, 1.0, -1.0)
            self.models_.append(self._fit_binary(X, binary_y))
        return self

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def decision_function(self, X):
        all_scores = []
        for model in self.models_:
            if len(model["X"]) == 0:
                all_scores.append(np.zeros(len(X)))
                continue
            kernel_values = self._kernel(X, model["X"])
            all_scores.append(np.dot(kernel_values, model["alphas"] * model["y"]) + model["b"])
        return np.vstack(all_scores).T

    def decision_values_binary(self, X):
        model = self.get_visual_binary_model()
        if model is None or len(model["X"]) == 0:
            return np.zeros(len(X))
        kernel_values = self._kernel(X, model["X"])
        return np.dot(kernel_values, model["alphas"] * model["y"]) + model["b"]

    def get_visual_binary_model(self):
        if len(self.classes_) < 2:
            return None
        if len(self.models_) >= 2:
            return self.models_[-1]
        return self.models_[0]

    def support_vectors(self):
        model = self.get_visual_binary_model()
        return np.empty((0, 2)) if model is None else model["X"]

    def support_vector_count(self):
        model = self.get_visual_binary_model()
        return 0 if model is None else len(model["X"])

    def linear_margin_width(self):
        if self.kernel != "linear":
            return None
        model = self.get_visual_binary_model()
        if model is None or len(model["X"]) == 0:
            return None
        weights = np.dot(model["alphas"] * model["y"], model["X"])
        norm = np.linalg.norm(weights)
        if norm < 1e-9:
            return None
        return 2.0 / norm

    def _fit_binary(self, X, y):
        rng = np.random.RandomState(42)
        sample_count = len(X)
        alphas = np.zeros(sample_count)
        bias = 0.0
        kernel_matrix = self._kernel(X, X)
        passes = 0

        while passes < self.max_passes:
            changed = 0
            for i in range(sample_count):
                Ei = np.sum(alphas * y * kernel_matrix[:, i]) + bias - y[i]
                violates = (y[i] * Ei < -self.tol and alphas[i] < self.C) or (
                    y[i] * Ei > self.tol and alphas[i] > 0
                )
                if not violates:
                    continue

                j = i
                while j == i:
                    j = rng.randint(0, sample_count)

                Ej = np.sum(alphas * y * kernel_matrix[:, j]) + bias - y[j]
                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]

                if y[i] != y[j]:
                    low = max(0.0, alphas[j] - alphas[i])
                    high = min(self.C, self.C + alphas[j] - alphas[i])
                else:
                    low = max(0.0, alphas[i] + alphas[j] - self.C)
                    high = min(self.C, alphas[i] + alphas[j])
                if abs(low - high) < 1e-12:
                    continue

                eta = 2.0 * kernel_matrix[i, j] - kernel_matrix[i, i] - kernel_matrix[j, j]
                if eta >= 0:
                    continue

                alphas[j] -= y[j] * (Ei - Ej) / eta
                alphas[j] = np.clip(alphas[j], low, high)
                if abs(alphas[j] - alpha_j_old) < 1e-5:
                    continue

                alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])

                b1 = (
                    bias
                    - Ei
                    - y[i] * (alphas[i] - alpha_i_old) * kernel_matrix[i, i]
                    - y[j] * (alphas[j] - alpha_j_old) * kernel_matrix[i, j]
                )
                b2 = (
                    bias
                    - Ej
                    - y[i] * (alphas[i] - alpha_i_old) * kernel_matrix[i, j]
                    - y[j] * (alphas[j] - alpha_j_old) * kernel_matrix[j, j]
                )
                if 0.0 < alphas[i] < self.C:
                    bias = b1
                elif 0.0 < alphas[j] < self.C:
                    bias = b2
                else:
                    bias = 0.5 * (b1 + b2)
                changed += 1

            passes = passes + 1 if changed == 0 else 0

        support = alphas > 1e-6
        return {
            "X": X[support],
            "y": y[support],
            "alphas": alphas[support],
            "b": bias,
        }

    def _kernel(self, X1, X2):
        if self.kernel == "linear":
            return np.dot(X1, X2.T)
        if self.kernel == "poly":
            return (1.0 + self.gamma * np.dot(X1, X2.T)) ** 3

        x1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        x2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        dist_sq = x1_sq + x2_sq - 2.0 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * np.maximum(dist_sq, 0.0))


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeClassifierCustom:
    def __init__(self, max_depth=4, min_samples_split=4, max_features=None, seed=0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = np.random.RandomState(seed)

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.root_ = self._grow_tree(X, y, depth=0)
        return self

    def predict(self, X):
        return np.array([self._predict_row(row, self.root_) for row in X])

    def _grow_tree(self, X, y, depth):
        if len(np.unique(y)) == 1:
            return TreeNode(value=int(y[0]))
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return TreeNode(value=majority_class(y))

        feature_indices = np.arange(self.n_features_)
        if self.max_features is not None and self.max_features < self.n_features_:
            feature_indices = self.rng.choice(feature_indices, self.max_features, replace=False)

        best_feature = None
        best_threshold = None
        best_gain = -1.0
        for feature in feature_indices:
            thresholds = np.unique(np.percentile(X[:, feature], [20, 40, 60, 80]))
            for threshold in thresholds:
                gain = gini_gain(y, X[:, feature], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        if best_feature is None or best_gain <= 1e-8:
            return TreeNode(value=majority_class(y))

        mask = X[:, best_feature] <= best_threshold
        left = self._grow_tree(X[mask], y[mask], depth + 1)
        right = self._grow_tree(X[~mask], y[~mask], depth + 1)
        return TreeNode(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def _predict_row(self, row, node):
        if node.value is not None:
            return node.value
        if row[node.feature] <= node.threshold:
            return self._predict_row(row, node.left)
        return self._predict_row(row, node.right)


class RandomForestClassifierCustom:
    def __init__(self, n_estimators=11, max_depth=4, min_samples_split=4):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.trees_ = []
        rng = np.random.RandomState(123)
        max_features = max(1, int(np.sqrt(X.shape[1])))
        for index in range(self.n_estimators):
            sample_idx = rng.choice(len(X), len(X), replace=True)
            tree = DecisionTreeClassifierCustom(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
                seed=100 + index,
            )
            tree.fit(X[sample_idx], y[sample_idx])
            self.trees_.append(tree)
        return self

    def predict(self, X):
        tree_predictions = np.vstack([tree.predict(X) for tree in self.trees_]).T
        result = []
        for row in tree_predictions:
            result.append(Counter(row.tolist()).most_common(1)[0][0])
        return np.array(result)

    def single_tree_predict(self, X):
        if not self.trees_:
            return np.zeros(len(X), dtype=int)
        return self.trees_[0].predict(X)


def majority_class(y):
    return Counter(y.tolist()).most_common(1)[0][0]


def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts.astype(float) / len(y)
    return 1.0 - np.sum(probabilities ** 2)


def gini_gain(y, feature_values, threshold):
    parent = gini_impurity(y)
    left_mask = feature_values <= threshold
    right_mask = ~left_mask
    if left_mask.sum() == 0 or right_mask.sum() == 0:
        return 0.0

    left_score = gini_impurity(y[left_mask])
    right_score = gini_impurity(y[right_mask])
    child_score = left_mask.mean() * left_score + right_mask.mean() * right_score
    return parent - child_score
