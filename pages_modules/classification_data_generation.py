import math

import numpy as np


ALGORITHM_DATASET_OPTIONS = {
    "knn": ["knn_moons", "knn_circles", "knn_local_vote"],
    "svm": ["svm_linear_margin", "svm_soft_margin", "svm_kernel_curve"],
    "nb": ["nb_independent_gaussian", "nb_overlap_gaussian"],
    "rf": ["rf_block_regions", "rf_step_regions", "rf_noisy_blocks"],
}

DATASET_META = {
    "knn_moons": {
        "label": "双月牙近邻投票",
        "summary": "弯曲边界很明显，特别适合观察 K 值变化如何影响局部投票结果。",
    },
    "knn_circles": {
        "label": "同心圆邻域划分",
        "summary": "内外两圈天然依赖局部邻域结构，能直观看到距离加权与普通投票的差别。",
    },
    "knn_local_vote": {
        "label": "局部团块投票",
        "summary": "类簇之间局部重叠，更容易看出 KNN 为什么会受到邻居数量影响。",
    },
    "svm_linear_margin": {
        "label": "线性可分间隔数据",
        "summary": "两类样本近似线性可分，适合观察超平面、间隔带和支持向量。",
    },
    "svm_soft_margin": {
        "label": "软间隔噪声数据",
        "summary": "样本存在一定交叠和噪声，适合观察 C 值变化对容错与间隔的影响。",
    },
    "svm_kernel_curve": {
        "label": "核函数弯曲边界数据",
        "summary": "线性边界不够用，能直观看到不同 kernel 带来的边界差异。",
    },
    "nb_independent_gaussian": {
        "label": "独立高斯分布",
        "summary": "特征近似独立，最适合讲解朴素贝叶斯的概率分类思路。",
    },
    "nb_overlap_gaussian": {
        "label": "概率重叠分布",
        "summary": "类别区域有明显重叠，适合观察概率大小而不是硬边界。",
    },
    "rf_block_regions": {
        "label": "规则分块数据",
        "summary": "类别由多个矩形区域组成，特别适合树模型和随机森林展示分块切分。",
    },
    "rf_step_regions": {
        "label": "阶梯式切分数据",
        "summary": "类别边界像阶梯一样变化，有利于观察树深度和分裂阈值的影响。",
    },
    "rf_noisy_blocks": {
        "label": "带噪声的分块数据",
        "summary": "在复杂局部结构上加入噪声，更容易看出单棵树和森林稳定性的差异。",
    },
}

DEFAULT_DATASET_BY_ALGORITHM = {
    "knn": "knn_moons",
    "svm": "svm_linear_margin",
    "nb": "nb_independent_gaussian",
    "rf": "rf_block_regions",
}


def get_dataset_options(algorithm_key):
    return ALGORITHM_DATASET_OPTIONS[algorithm_key]


def get_default_dataset(algorithm_key):
    return DEFAULT_DATASET_BY_ALGORITHM[algorithm_key]


def get_dataset_label(dataset_key):
    return DATASET_META[dataset_key]["label"]


def get_dataset_summary(dataset_key):
    return DATASET_META[dataset_key]["summary"]


def generate_algorithm_dataset(dataset_key, n_samples, noise, random_state):
    rng = np.random.RandomState(random_state)

    if dataset_key == "knn_moons":
        return make_moons(n_samples, noise, rng)
    if dataset_key == "knn_circles":
        return make_circles(n_samples, noise, rng)
    if dataset_key == "knn_local_vote":
        return make_local_vote_clusters(n_samples, noise, rng)
    if dataset_key == "svm_linear_margin":
        return make_linear_margin_data(n_samples, noise, rng)
    if dataset_key == "svm_soft_margin":
        return make_soft_margin_data(n_samples, noise, rng)
    if dataset_key == "svm_kernel_curve":
        return make_kernel_curve_data(n_samples, noise, rng)
    if dataset_key == "nb_independent_gaussian":
        return make_independent_gaussian_data(n_samples, noise, rng)
    if dataset_key == "nb_overlap_gaussian":
        return make_overlap_gaussian_data(n_samples, noise, rng)
    if dataset_key == "rf_block_regions":
        return make_block_regions_data(n_samples, noise, rng)
    if dataset_key == "rf_step_regions":
        return make_step_regions_data(n_samples, noise, rng)
    return make_noisy_blocks_data(n_samples, noise, rng)


def make_moons(n_samples, noise, rng):
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    theta_outer = rng.rand(n_outer) * math.pi
    theta_inner = rng.rand(n_inner) * math.pi
    outer = np.c_[np.cos(theta_outer), np.sin(theta_outer)]
    inner = np.c_[1.0 - np.cos(theta_inner), 0.46 - np.sin(theta_inner)]

    X = np.vstack([outer, inner])
    y = np.hstack([np.zeros(n_outer, dtype=int), np.ones(n_inner, dtype=int)])
    X += rng.normal(scale=noise, size=X.shape)
    return X, y


def make_circles(n_samples, noise, rng):
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    theta_outer = rng.rand(n_outer) * 2.0 * math.pi
    theta_inner = rng.rand(n_inner) * 2.0 * math.pi
    outer = np.c_[np.cos(theta_outer), np.sin(theta_outer)]
    inner = 0.56 * np.c_[np.cos(theta_inner), np.sin(theta_inner)]

    X = np.vstack([outer, inner])
    y = np.hstack([np.zeros(n_outer, dtype=int), np.ones(n_inner, dtype=int)])
    X += rng.normal(scale=noise, size=X.shape)
    return X, y


def make_local_vote_clusters(n_samples, noise, rng):
    centers = np.array([[-1.6, -0.6], [-0.2, 0.8], [1.3, 0.2], [-1.0, 1.3], [0.4, -0.5], [1.6, 1.0]])
    labels = np.array([0, 0, 0, 1, 1, 1])
    spread = 0.12 + noise * 0.9

    points = []
    targets = []
    for index in range(n_samples):
        center_id = rng.randint(0, len(centers))
        point = centers[center_id] + rng.normal(scale=spread, size=2)
        points.append(point)
        targets.append(labels[center_id])
    return np.array(points), np.array(targets, dtype=int)


def make_linear_margin_data(n_samples, noise, rng):
    n_class0 = n_samples // 2
    n_class1 = n_samples - n_class0
    class0 = rng.multivariate_normal(mean=[-1.4, -0.8], cov=[[0.22, 0.02], [0.02, 0.18]], size=n_class0)
    class1 = rng.multivariate_normal(mean=[1.3, 0.9], cov=[[0.22, 0.02], [0.02, 0.18]], size=n_class1)
    X = np.vstack([class0, class1])
    y = np.hstack([np.zeros(n_class0, dtype=int), np.ones(n_class1, dtype=int)])
    X += rng.normal(scale=noise * 0.45, size=X.shape)
    return X, y


def make_soft_margin_data(n_samples, noise, rng):
    n_class0 = n_samples // 2
    n_class1 = n_samples - n_class0
    class0 = rng.multivariate_normal(mean=[-1.1, -0.6], cov=[[0.36, 0.08], [0.08, 0.30]], size=n_class0)
    class1 = rng.multivariate_normal(mean=[1.0, 0.8], cov=[[0.38, -0.05], [-0.05, 0.28]], size=n_class1)
    X = np.vstack([class0, class1])
    y = np.hstack([np.zeros(n_class0, dtype=int), np.ones(n_class1, dtype=int)])
    X += rng.normal(scale=noise * 0.65, size=X.shape)
    return X, y


def make_kernel_curve_data(n_samples, noise, rng):
    X, y = make_moons(n_samples, max(0.08, noise * 0.8), rng)
    X[:, 0] *= 1.15
    X[:, 1] *= 0.95
    return X, y


def make_independent_gaussian_data(n_samples, noise, rng):
    n_class0 = n_samples // 2
    n_class1 = n_samples - n_class0
    cov0 = [[0.22 + noise * 0.35, 0.0], [0.0, 0.12 + noise * 0.28]]
    cov1 = [[0.18 + noise * 0.30, 0.0], [0.0, 0.20 + noise * 0.32]]
    class0 = rng.multivariate_normal(mean=[-1.4, -0.7], cov=cov0, size=n_class0)
    class1 = rng.multivariate_normal(mean=[1.1, 1.0], cov=cov1, size=n_class1)
    X = np.vstack([class0, class1])
    y = np.hstack([np.zeros(n_class0, dtype=int), np.ones(n_class1, dtype=int)])
    return X, y


def make_overlap_gaussian_data(n_samples, noise, rng):
    n_class0 = n_samples // 2
    n_class1 = n_samples - n_class0
    cov0 = [[0.48 + noise * 0.35, 0.0], [0.0, 0.24 + noise * 0.25]]
    cov1 = [[0.40 + noise * 0.30, 0.0], [0.0, 0.28 + noise * 0.30]]
    class0 = rng.multivariate_normal(mean=[-0.6, -0.5], cov=cov0, size=n_class0)
    class1 = rng.multivariate_normal(mean=[0.9, 0.9], cov=cov1, size=n_class1)
    X = np.vstack([class0, class1])
    y = np.hstack([np.zeros(n_class0, dtype=int), np.ones(n_class1, dtype=int)])
    return X, y


def make_block_regions_data(n_samples, noise, rng):
    X = rng.uniform(-2.4, 2.4, size=(n_samples, 2))
    y = (
        ((X[:, 0] > 0.4) & (X[:, 1] > 0.4))
        | ((X[:, 0] < -0.7) & (X[:, 1] < -0.3))
        | ((X[:, 0] > -0.2) & (X[:, 0] < 0.8) & (X[:, 1] > 1.0))
    ).astype(int)
    y = flip_labels(y, noise * 0.22, rng)
    return X, y


def make_step_regions_data(n_samples, noise, rng):
    X = rng.uniform(-2.4, 2.4, size=(n_samples, 2))
    y = np.zeros(n_samples, dtype=int)
    y[(X[:, 0] > -1.1) & (X[:, 1] > 0.6)] = 1
    y[(X[:, 0] > -0.2) & (X[:, 1] > -0.4)] = 1
    y[(X[:, 0] > 1.0) & (X[:, 1] > -1.0)] = 1
    y = flip_labels(y, noise * 0.20, rng)
    return X, y


def make_noisy_blocks_data(n_samples, noise, rng):
    X, y = make_block_regions_data(n_samples, max(noise, 0.08), rng)
    X += rng.normal(scale=noise * 0.38, size=X.shape)
    return X, y


def flip_labels(y, ratio, rng):
    ratio = max(0.0, min(0.35, ratio))
    flip_mask = rng.rand(len(y)) < ratio
    flipped = y.copy()
    flipped[flip_mask] = 1 - flipped[flip_mask]
    return flipped


def train_test_split(X, y, test_size, random_state):
    rng = np.random.RandomState(random_state)
    indices = np.arange(len(X))
    rng.shuffle(indices)

    test_count = max(1, int(len(X) * test_size))
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


class StandardScaler:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0) + 1e-9
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X):
        return X * self.scale_ + self.mean_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def class_names_from_labels(y):
    return ["类别 {0}".format(label) for label in np.unique(y)]
