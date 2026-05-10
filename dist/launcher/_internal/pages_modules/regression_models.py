import numpy as np


ALGORITHM_LABELS = {
    "linear": "线性回归",
    "poly": "多项式回归",
    "ridge": "岭回归",
    "lasso": "套索回归",
    "svr": "支持向量回归 SVR",
    "tree": "决策树回归",
    "rf": "随机森林回归",
}


def build_regressor(algorithm_key, params):
    if algorithm_key == "linear":
        return LinearRegressionCustom(
            fit_intercept=params["fit_intercept"],
            standardize=params["standardize"],
        )
    if algorithm_key == "poly":
        return PolynomialRegressionCustom(degree=params["degree"])
    if algorithm_key == "ridge":
        return RidgeRegressionCustom(alpha=params["alpha"])
    if algorithm_key == "lasso":
        return LassoRegressionCustom(alpha=params["alpha"])
    if algorithm_key == "svr":
        return KernelSVRCustom(
            C=params["C"],
            epsilon=params["epsilon"],
            kernel=params["kernel"],
            gamma=params["gamma"],
        )
    if algorithm_key == "tree":
        return DecisionTreeRegressorCustom(
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
        )
    return RandomForestRegressorCustom(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
    )


class LinearRegressionCustom:
    def __init__(self, fit_intercept=True, standardize=False):
        self.fit_intercept = fit_intercept
        self.standardize = standardize

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.x_mean_ = X.mean(axis=0) if self.standardize else np.zeros(X.shape[1])
        self.x_scale_ = X.std(axis=0) + 1e-9 if self.standardize else np.ones(X.shape[1])
        X_used = (X - self.x_mean_) / self.x_scale_ if self.standardize else X.copy()

        if self.fit_intercept:
            X_design = np.column_stack([np.ones(len(X_used)), X_used])
            weights = np.linalg.pinv(X_design).dot(y)
            intercept_scaled = weights[0]
            coef_scaled = weights[1:]
        else:
            intercept_scaled = 0.0
            coef_scaled = np.linalg.pinv(X_used).dot(y)

        self.coef_ = coef_scaled / self.x_scale_
        self.intercept_ = intercept_scaled - np.dot(self.coef_, self.x_mean_)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X.dot(self.coef_) + self.intercept_


class PolynomialRegressionCustom:
    def __init__(self, degree=3):
        self.degree = degree

    def fit(self, X, y):
        x = np.asarray(X, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float)
        self.x_mean_ = x.mean()
        self.x_scale_ = x.std() + 1e-9
        z = (x - self.x_mean_) / self.x_scale_
        X_poly = polynomial_features(z, self.degree)
        X_design = np.column_stack([np.ones(len(X_poly)), X_poly])
        weights = np.linalg.pinv(X_design).dot(y)
        self.intercept_ = float(weights[0])
        self.poly_coef_ = weights[1:]
        return self

    def predict(self, X):
        x = np.asarray(X, dtype=float).reshape(-1)
        z = (x - self.x_mean_) / self.x_scale_
        X_poly = polynomial_features(z, self.degree)
        return self.intercept_ + X_poly.dot(self.poly_coef_)


class RidgeRegressionCustom:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.x_mean_ = X.mean(axis=0)
        self.x_scale_ = X.std(axis=0) + 1e-9
        self.y_mean_ = y.mean()

        Xs = (X - self.x_mean_) / self.x_scale_
        yc = y - self.y_mean_
        gram = Xs.T.dot(Xs)
        regularized = gram + self.alpha * np.eye(Xs.shape[1])
        coef_scaled = np.linalg.pinv(regularized).dot(Xs.T).dot(yc)

        self.coef_ = coef_scaled / self.x_scale_
        self.intercept_ = self.y_mean_ - np.dot(self.x_mean_, self.coef_)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X.dot(self.coef_) + self.intercept_


class LassoRegressionCustom:
    def __init__(self, alpha=0.1, max_iter=600, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.x_mean_ = X.mean(axis=0)
        self.x_scale_ = X.std(axis=0) + 1e-9
        self.y_mean_ = y.mean()

        Xs = (X - self.x_mean_) / self.x_scale_
        yc = y - self.y_mean_
        weights = np.zeros(Xs.shape[1], dtype=float)

        for _ in range(self.max_iter):
            max_change = 0.0
            for column in range(Xs.shape[1]):
                residual = yc - Xs.dot(weights) + Xs[:, column] * weights[column]
                rho = np.dot(Xs[:, column], residual) / len(Xs)
                z = np.dot(Xs[:, column], Xs[:, column]) / len(Xs)
                new_weight = soft_threshold(rho, self.alpha) / max(z, 1e-9)
                max_change = max(max_change, abs(new_weight - weights[column]))
                weights[column] = new_weight
            if max_change < self.tol:
                break

        self.coef_ = weights / self.x_scale_
        self.intercept_ = self.y_mean_ - np.dot(self.x_mean_, self.coef_)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X.dot(self.coef_) + self.intercept_


class KernelSVRCustom:
    def __init__(self, C=1.0, epsilon=0.25, kernel="rbf", gamma=0.9, max_iter=700, learning_rate=0.08):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.X_train_ = X
        self.y_mean_ = y.mean()
        yc = y - self.y_mean_

        kernel_matrix = self._kernel(X, X)
        alpha = np.zeros(len(X), dtype=float)
        step = self.learning_rate / max(np.mean(np.diag(kernel_matrix)), 1.0)

        for _ in range(self.max_iter):
            prediction = kernel_matrix.dot(alpha)
            residual = prediction - yc
            outside = np.abs(residual) > self.epsilon
            gradient = kernel_matrix.dot(alpha)
            if np.any(outside):
                penalty = np.sign(residual[outside]) * (np.abs(residual[outside]) - self.epsilon)
                gradient += 2.0 * self.C * kernel_matrix[:, outside].dot(penalty) / len(X)
            alpha -= step * gradient / len(X)

        self.alpha_ = alpha
        self.train_prediction_centered_ = kernel_matrix.dot(alpha)
        self.train_prediction_ = self.train_prediction_centered_ + self.y_mean_
        train_residual = self.train_prediction_ - y
        self.support_mask_ = np.abs(train_residual) >= self.epsilon * 0.98
        if self.support_mask_.sum() == 0:
            support_count = min(6, len(X))
            largest = np.argsort(np.abs(train_residual))[-support_count:]
            self.support_mask_[largest] = True
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        kernel_values = self._kernel(X, self.X_train_)
        return self.y_mean_ + kernel_values.dot(self.alpha_)

    def support_vectors(self):
        return self.X_train_[self.support_mask_]

    def support_indices(self):
        return np.where(self.support_mask_)[0]

    def support_vector_count(self):
        return int(self.support_mask_.sum())

    def _kernel(self, X1, X2):
        if self.kernel == "linear":
            return X1.dot(X2.T)
        if self.kernel == "poly":
            return (1.0 + self.gamma * X1.dot(X2.T)) ** 3

        x1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        x2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        distance_sq = np.maximum(x1_sq + x2_sq - 2.0 * X1.dot(X2.T), 0.0)
        return np.exp(-self.gamma * distance_sq)


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeRegressorCustom:
    def __init__(self, max_depth=4, min_samples_split=6, max_features=None, seed=0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = np.random.RandomState(seed)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.n_features_ = X.shape[1]
        self.root_ = self._grow_tree(X, y, depth=0)
        self.leaf_count_ = count_leaves(self.root_)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_row(row, self.root_) for row in X], dtype=float)

    def _grow_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.var(y) < 1e-8:
            return TreeNode(value=float(y.mean()))

        feature_indices = np.arange(self.n_features_)
        if self.max_features is not None and self.max_features < self.n_features_:
            feature_indices = self.rng.choice(feature_indices, self.max_features, replace=False)

        best_feature = None
        best_threshold = None
        best_gain = 0.0
        for feature in feature_indices:
            thresholds = np.unique(np.percentile(X[:, feature], [10, 20, 30, 40, 50, 60, 70, 80, 90]))
            for threshold in thresholds:
                gain = mse_gain(y, X[:, feature], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        if best_feature is None or best_gain <= 1e-9:
            return TreeNode(value=float(y.mean()))

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        return TreeNode(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def _predict_row(self, row, node):
        if node.value is not None:
            return node.value
        if row[node.feature] <= node.threshold:
            return self._predict_row(row, node.left)
        return self._predict_row(row, node.right)


class RandomForestRegressorCustom:
    def __init__(self, n_estimators=21, max_depth=5, min_samples_split=5):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.trees_ = []
        rng = np.random.RandomState(123)
        max_features = max(1, int(np.sqrt(X.shape[1])))

        for tree_index in range(self.n_estimators):
            sample_index = rng.choice(len(X), len(X), replace=True)
            tree = DecisionTreeRegressorCustom(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_features,
                seed=100 + tree_index,
            )
            tree.fit(X[sample_index], y[sample_index])
            self.trees_.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        tree_predictions = np.vstack([tree.predict(X) for tree in self.trees_])
        return tree_predictions.mean(axis=0)

    def single_tree_predict(self, X):
        X = np.asarray(X, dtype=float)
        if not self.trees_:
            return np.zeros(len(X), dtype=float)
        return self.trees_[0].predict(X)


def polynomial_features(x, degree):
    x = np.asarray(x, dtype=float).reshape(-1)
    return np.column_stack([x ** power for power in range(1, degree + 1)])


def soft_threshold(value, alpha):
    if value > alpha:
        return value - alpha
    if value < -alpha:
        return value + alpha
    return 0.0


def mse_of_targets(y):
    y = np.asarray(y, dtype=float)
    return float(np.mean((y - y.mean()) ** 2))


def mse_gain(y, feature_values, threshold):
    parent_error = mse_of_targets(y)
    left_mask = feature_values <= threshold
    right_mask = ~left_mask
    if left_mask.sum() == 0 or right_mask.sum() == 0:
        return 0.0
    child_error = left_mask.mean() * mse_of_targets(y[left_mask]) + right_mask.mean() * mse_of_targets(y[right_mask])
    return parent_error - child_error


def count_leaves(node):
    if node.value is not None:
        return 1
    return count_leaves(node.left) + count_leaves(node.right)
