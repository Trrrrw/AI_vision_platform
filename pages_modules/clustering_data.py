import math

import numpy as np


ALGORITHM_DATASET_OPTIONS = {
    "kmeans": ["kmeans_blobs", "kmeans_noisy_blobs"],
    "dbscan": ["dbscan_moons", "dbscan_circles", "dbscan_vary_density"],
    "agg": ["agg_nested_blobs", "agg_bridge_groups"],
    "gmm": ["gmm_ellipses", "gmm_overlap_ellipses"],
}


DATASET_META = {
    "kmeans_blobs": {
        "label": "规则球状簇数据",
        "summary": "簇中心明显、形状接近圆形，最适合用来讲解 KMeans 的中心聚类思想。",
    },
    "kmeans_noisy_blobs": {
        "label": "带噪声的中心簇数据",
        "summary": "整体仍然以中心为主，但加入了少量噪声与重叠，更适合观察初始中心和簇数变化带来的差异。",
    },
    "dbscan_moons": {
        "label": "双月牙密度簇数据",
        "summary": "簇形状明显弯曲，非常适合展示 DBSCAN 对任意形状簇的识别能力。",
    },
    "dbscan_circles": {
        "label": "同心圆密度簇数据",
        "summary": "内外两圈结构让 DBSCAN 的密度连通思想非常直观，也能和 KMeans 形成鲜明对比。",
    },
    "dbscan_vary_density": {
        "label": "不同密度混合数据",
        "summary": "同一张图里同时存在稠密簇和稀疏簇，便于理解 eps 和 min_samples 为什么很关键。",
    },
    "agg_nested_blobs": {
        "label": "层次子簇数据",
        "summary": "大簇内部还包含更细的小簇，适合展示层次聚类从细到粗的合并过程。",
    },
    "agg_bridge_groups": {
        "label": "桥接式簇数据",
        "summary": "簇之间有一条稀疏过渡带，不同 linkage 会得到很不一样的合并顺序。",
    },
    "gmm_ellipses": {
        "label": "椭圆高斯分量数据",
        "summary": "簇呈椭圆分布，最适合解释 GMM 的高斯成分和软分配思想。",
    },
    "gmm_overlap_ellipses": {
        "label": "重叠椭圆簇数据",
        "summary": "簇之间存在重叠区域，适合观察“一个点同时属于多个簇”的概率解释。",
    },
}


DEFAULT_DATASET_BY_ALGORITHM = {
    "kmeans": "kmeans_blobs",
    "dbscan": "dbscan_moons",
    "agg": "agg_nested_blobs",
    "gmm": "gmm_ellipses",
}


def get_dataset_options(algorithm_key):
    return ALGORITHM_DATASET_OPTIONS[algorithm_key]


def get_default_dataset(algorithm_key):
    return DEFAULT_DATASET_BY_ALGORITHM[algorithm_key]


def get_dataset_label(dataset_key):
    return DATASET_META[dataset_key]["label"]


def get_dataset_summary(dataset_key):
    return DATASET_META[dataset_key]["summary"]


def generate_clustering_dataset(dataset_key, n_samples, noise, random_state):
    rng = np.random.RandomState(random_state)

    if dataset_key == "kmeans_blobs":
        return make_blobs(rng, n_samples, noise, centers=[(-2.2, -0.8), (0.2, 1.7), (2.2, -0.2)], spread=0.34)
    if dataset_key == "kmeans_noisy_blobs":
        return make_blobs(
            rng,
            n_samples,
            noise,
            centers=[(-2.3, -1.1), (-0.4, 1.6), (1.4, 0.4), (2.6, 1.6)],
            spread=0.38,
            add_noise_points=True,
        )
    if dataset_key == "dbscan_moons":
        return make_moons(rng, n_samples, max(0.05, noise * 0.8 + 0.06))
    if dataset_key == "dbscan_circles":
        return make_circles(rng, n_samples, max(0.04, noise * 0.7 + 0.04))
    if dataset_key == "dbscan_vary_density":
        return make_vary_density_clusters(rng, n_samples, noise)
    if dataset_key == "agg_nested_blobs":
        return make_hierarchical_subclusters(rng, n_samples, noise)
    if dataset_key == "agg_bridge_groups":
        return make_bridge_groups(rng, n_samples, noise)
    if dataset_key == "gmm_ellipses":
        return make_elliptical_gaussians(rng, n_samples, noise, overlap=False)
    return make_elliptical_gaussians(rng, n_samples, noise, overlap=True)


def make_blobs(rng, n_samples, noise, centers, spread=0.35, add_noise_points=False):
    counts = split_counts(n_samples, len(centers))
    points = []
    for count, center in zip(counts, centers):
        cloud = rng.normal(loc=np.array(center), scale=spread + noise * 0.35, size=(count, 2))
        points.append(cloud)
    X = np.vstack(points)
    if add_noise_points:
        noise_count = max(4, n_samples // 16)
        noise_points = rng.uniform(low=[-3.0, -2.4], high=[3.1, 2.8], size=(noise_count, 2))
        X[:noise_count] = noise_points
    rng.shuffle(X)
    return X, {"teacher_dataset": True}


def make_moons(rng, n_samples, noise):
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer
    theta_outer = rng.rand(n_outer) * math.pi
    theta_inner = rng.rand(n_inner) * math.pi
    outer = np.c_[np.cos(theta_outer), np.sin(theta_outer)]
    inner = np.c_[1.0 - np.cos(theta_inner), 0.46 - np.sin(theta_inner)]
    X = np.vstack([outer, inner])
    X += rng.normal(scale=noise, size=X.shape)
    return X, {"teacher_dataset": True}


def make_circles(rng, n_samples, noise):
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer
    theta_outer = rng.rand(n_outer) * 2.0 * math.pi
    theta_inner = rng.rand(n_inner) * 2.0 * math.pi
    outer = np.c_[np.cos(theta_outer), np.sin(theta_outer)]
    inner = 0.52 * np.c_[np.cos(theta_inner), np.sin(theta_inner)]
    X = np.vstack([outer, inner])
    X += rng.normal(scale=noise, size=X.shape)
    return X, {"teacher_dataset": True}


def make_vary_density_clusters(rng, n_samples, noise):
    dense_count = int(n_samples * 0.42)
    sparse_count = int(n_samples * 0.34)
    ring_count = n_samples - dense_count - sparse_count

    dense = rng.normal(loc=[-1.8, 0.3], scale=[0.20 + noise * 0.12, 0.18 + noise * 0.12], size=(dense_count, 2))
    sparse = rng.normal(loc=[1.4, 0.8], scale=[0.42 + noise * 0.18, 0.34 + noise * 0.16], size=(sparse_count, 2))
    theta = rng.rand(ring_count) * 2.0 * math.pi
    ring = np.c_[2.1 + 0.62 * np.cos(theta), -1.0 + 0.36 * np.sin(theta)]
    ring += rng.normal(scale=0.08 + noise * 0.10, size=ring.shape)

    X = np.vstack([dense, sparse, ring])
    rng.shuffle(X)
    return X, {"teacher_dataset": True}


def make_hierarchical_subclusters(rng, n_samples, noise):
    centers = [(-2.2, -0.3), (-1.4, 0.6), (1.1, -0.4), (2.0, 0.7)]
    counts = split_counts(n_samples, len(centers))
    points = []
    for count, center in zip(counts, centers):
        points.append(rng.normal(loc=np.array(center), scale=0.20 + noise * 0.10, size=(count, 2)))
    X = np.vstack(points)
    rng.shuffle(X)
    return X, {"teacher_dataset": True}


def make_bridge_groups(rng, n_samples, noise):
    group_count = int(n_samples * 0.38)
    bridge_count = max(10, int(n_samples * 0.18))
    right_count = n_samples - group_count - bridge_count

    left = rng.normal(loc=[-2.0, -0.2], scale=[0.28 + noise * 0.10, 0.24 + noise * 0.10], size=(group_count, 2))
    right = rng.normal(loc=[2.0, 0.7], scale=[0.32 + noise * 0.12, 0.26 + noise * 0.10], size=(right_count, 2))
    bridge_x = np.linspace(-0.8, 0.9, bridge_count)
    bridge = np.c_[bridge_x, 0.1 + 0.18 * np.sin(bridge_x * 2.4)]
    bridge += rng.normal(scale=0.10 + noise * 0.08, size=bridge.shape)

    X = np.vstack([left, bridge, right])
    rng.shuffle(X)
    return X, {"teacher_dataset": True}


def make_elliptical_gaussians(rng, n_samples, noise, overlap=False):
    counts = split_counts(n_samples, 3 if not overlap else 2)
    if not overlap:
        means = [(-2.2, -0.5), (0.1, 1.7), (2.3, 0.2)]
        covs = [
            [[0.28 + noise * 0.16, 0.10], [0.10, 0.14 + noise * 0.10]],
            [[0.36 + noise * 0.18, -0.16], [-0.16, 0.20 + noise * 0.10]],
            [[0.24 + noise * 0.14, 0.12], [0.12, 0.30 + noise * 0.16]],
        ]
    else:
        means = [(-0.9, -0.1), (1.0, 0.8)]
        covs = [
            [[0.84 + noise * 0.20, 0.42], [0.42, 0.34 + noise * 0.12]],
            [[0.72 + noise * 0.18, -0.35], [-0.35, 0.42 + noise * 0.14]],
        ]
    points = []
    for count, mean, cov in zip(counts, means, covs):
        points.append(rng.multivariate_normal(mean=mean, cov=cov, size=count))
    X = np.vstack(points)
    rng.shuffle(X)
    return X, {"teacher_dataset": True}


def split_counts(total, parts):
    base = total // parts
    counts = [base] * parts
    for index in range(total - base * parts):
        counts[index] += 1
    return counts
