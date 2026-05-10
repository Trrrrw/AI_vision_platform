from collections import deque

import numpy as np


ALGORITHM_LABELS = {
    "kmeans": "KMeans 聚类",
    "dbscan": "DBSCAN 密度聚类",
    "agg": "层次聚类",
    "gmm": "高斯混合模型 GMM",
}


def build_clusterer(algorithm_key, params):
    if algorithm_key == "kmeans":
        return KMeansCustom(
            n_clusters=params["n_clusters"],
            init=params["init"],
            n_init=params["n_init"],
            max_iter=params["max_iter"],
        )
    if algorithm_key == "dbscan":
        return DBSCANCustom(eps=params["eps"], min_samples=params["min_samples"])
    if algorithm_key == "agg":
        return AgglomerativeClusteringCustom(
            n_clusters=params["n_clusters"],
            linkage=params["linkage"],
        )
    return GaussianMixtureCustom(
        n_components=params["n_components"],
        covariance_type=params["covariance_type"],
        max_iter=params["max_iter"],
    )


class KMeansCustom:
    def __init__(self, n_clusters=3, init="kmeans++", n_init=4, max_iter=20):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        best = None
        rng = np.random.RandomState(42)

        for init_round in range(self.n_init):
            seed = 100 + init_round
            centers = self._initialize_centers(X, np.random.RandomState(seed))
            history = [centers.copy()]
            labels = np.zeros(len(X), dtype=int)
            for _ in range(self.max_iter):
                distances = pairwise_distance_sq(X, centers)
                labels = np.argmin(distances, axis=1)
                new_centers = centers.copy()
                for cluster_id in range(self.n_clusters):
                    mask = labels == cluster_id
                    if np.any(mask):
                        new_centers[cluster_id] = X[mask].mean(axis=0)
                    else:
                        new_centers[cluster_id] = X[rng.randint(0, len(X))]
                history.append(new_centers.copy())
                if np.allclose(new_centers, centers, atol=1e-5):
                    centers = new_centers
                    break
                centers = new_centers
            inertia = float(np.sum((X - centers[labels]) ** 2))
            candidate = {
                "labels": labels.copy(),
                "centers": centers.copy(),
                "history": history,
                "inertia": inertia,
            }
            if best is None or candidate["inertia"] < best["inertia"]:
                best = candidate

        self.labels_ = best["labels"]
        self.cluster_centers_ = best["centers"]
        self.center_history_ = best["history"]
        self.initial_centers_ = best["history"][0]
        self.inertia_ = best["inertia"]
        self.iterations_ = len(best["history"]) - 1
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        distances = pairwise_distance_sq(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)

    def _initialize_centers(self, X, rng):
        if self.init == "random":
            indices = rng.choice(len(X), self.n_clusters, replace=False)
            return X[indices].copy()
        centers = [X[rng.randint(0, len(X))]]
        for _ in range(1, self.n_clusters):
            distances = np.min(pairwise_distance_sq(X, np.array(centers)), axis=1)
            probability = distances / np.sum(distances)
            next_index = rng.choice(len(X), p=probability)
            centers.append(X[next_index])
        return np.array(centers, dtype=float)


class DBSCANCustom:
    def __init__(self, eps=0.32, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.X_ = X
        distance_matrix = np.sqrt(pairwise_distance_sq(X, X))
        neighbor_lists = [np.where(distance_matrix[index] <= self.eps)[0] for index in range(len(X))]
        core_mask = np.array([len(neighbors) >= self.min_samples for neighbors in neighbor_lists], dtype=bool)

        labels = np.full(len(X), -1, dtype=int)
        cluster_id = 0
        visited = np.zeros(len(X), dtype=bool)
        for index in range(len(X)):
            if visited[index]:
                continue
            visited[index] = True
            if not core_mask[index]:
                continue
            labels[index] = cluster_id
            queue = deque([index])
            while queue:
                current = queue.popleft()
                for neighbor in neighbor_lists[current]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        if core_mask[neighbor]:
                            queue.append(neighbor)
                    if labels[neighbor] == -1:
                        labels[neighbor] = cluster_id
            cluster_id += 1

        border_mask = (labels != -1) & (~core_mask)
        noise_mask = labels == -1

        self.labels_ = labels
        self.core_sample_mask_ = core_mask
        self.border_mask_ = border_mask
        self.noise_mask_ = noise_mask
        self.neighbor_lists_ = neighbor_lists
        self.cluster_count_ = int(len(np.unique(labels[labels >= 0])))
        return self


class AgglomerativeClusteringCustom:
    def __init__(self, n_clusters=3, linkage="average"):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.X_ = X
        n_samples = len(X)
        point_dist = np.sqrt(pairwise_distance_sq(X, X))
        clusters = {index: [index] for index in range(n_samples)}
        cluster_sizes = {index: 1 for index in range(n_samples)}
        cluster_means = {index: X[index].copy() for index in range(n_samples)}
        next_cluster_id = n_samples
        merge_history = []
        target_snapshot = None

        while len(clusters) > 1:
            cluster_ids = list(clusters.keys())
            best_pair = None
            best_distance = None
            for i in range(len(cluster_ids) - 1):
                left_id = cluster_ids[i]
                for j in range(i + 1, len(cluster_ids)):
                    right_id = cluster_ids[j]
                    distance = cluster_distance(
                        clusters[left_id],
                        clusters[right_id],
                        point_dist,
                        self.linkage,
                        cluster_means[left_id],
                        cluster_means[right_id],
                        cluster_sizes[left_id],
                        cluster_sizes[right_id],
                    )
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_pair = (left_id, right_id)

            left_id, right_id = best_pair
            merged_points = clusters[left_id] + clusters[right_id]
            merge_history.append(
                {
                    "left": left_id,
                    "right": right_id,
                    "distance": float(best_distance),
                    "size": len(merged_points),
                }
            )
            clusters[next_cluster_id] = merged_points
            cluster_sizes[next_cluster_id] = len(merged_points)
            cluster_means[next_cluster_id] = X[merged_points].mean(axis=0)
            del clusters[left_id]
            del clusters[right_id]
            del cluster_sizes[left_id]
            del cluster_sizes[right_id]
            del cluster_means[left_id]
            del cluster_means[right_id]
            next_cluster_id += 1

            if len(clusters) == self.n_clusters:
                target_snapshot = {cluster_id: members[:] for cluster_id, members in clusters.items()}

        if target_snapshot is None:
            target_snapshot = {cluster_id: members[:] for cluster_id, members in clusters.items()}

        labels = np.full(n_samples, -1, dtype=int)
        ordered = sorted(target_snapshot.items(), key=lambda item: min(item[1]))
        for label, (_, members) in enumerate(ordered):
            labels[members] = label

        cut_distance = None
        if self.n_clusters > 1 and len(merge_history) >= n_samples - self.n_clusters + 1:
            index = n_samples - self.n_clusters
            if index < len(merge_history):
                cut_distance = merge_history[index]["distance"]

        self.labels_ = labels
        self.merge_history_ = merge_history
        self.cut_distance_ = cut_distance
        self.cluster_count_ = int(len(np.unique(labels)))
        return self


class GaussianMixtureCustom:
    def __init__(self, n_components=3, covariance_type="full", max_iter=20):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.max_iter = max_iter

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        rng = np.random.RandomState(7)
        means = kmeans_plus_plus_init(X, self.n_components, rng)
        weights = np.ones(self.n_components, dtype=float) / self.n_components
        shared_cov = np.cov(X.T) + np.eye(n_features) * 0.12

        if self.covariance_type == "diag":
            covariances = np.tile(np.diag(shared_cov), (self.n_components, 1))
        else:
            covariances = np.tile(shared_cov[None, :, :], (self.n_components, 1, 1))

        log_likelihood_history = []
        responsibilities = np.zeros((n_samples, self.n_components), dtype=float)

        for _ in range(self.max_iter):
            log_prob = np.zeros((n_samples, self.n_components), dtype=float)
            for component in range(self.n_components):
                log_prob[:, component] = np.log(weights[component] + 1e-12) + gaussian_log_pdf(
                    X,
                    means[component],
                    covariances[component],
                    self.covariance_type,
                )

            row_max = log_prob.max(axis=1, keepdims=True)
            stabilized = np.exp(log_prob - row_max)
            responsibilities = stabilized / stabilized.sum(axis=1, keepdims=True)
            log_likelihood = float(np.sum(row_max + np.log(stabilized.sum(axis=1, keepdims=True) + 1e-12)))
            log_likelihood_history.append(log_likelihood)

            nk = responsibilities.sum(axis=0) + 1e-9
            weights = nk / n_samples
            means = responsibilities.T.dot(X) / nk[:, None]

            if self.covariance_type == "diag":
                new_cov = np.zeros_like(covariances)
                for component in range(self.n_components):
                    diff = X - means[component]
                    new_cov[component] = (responsibilities[:, component][:, None] * (diff ** 2)).sum(axis=0) / nk[component]
                    new_cov[component] += 1e-4
                covariances = new_cov
            else:
                new_cov = np.zeros_like(covariances)
                for component in range(self.n_components):
                    diff = X - means[component]
                    weighted = responsibilities[:, component][:, None] * diff
                    cov = weighted.T.dot(diff) / nk[component]
                    cov += np.eye(n_features) * 1e-4
                    new_cov[component] = cov
                covariances = new_cov

            if len(log_likelihood_history) >= 2 and abs(log_likelihood_history[-1] - log_likelihood_history[-2]) < 1e-4:
                break

        self.weights_ = weights
        self.means_ = means
        self.covariances_ = covariances
        self.responsibilities_ = responsibilities
        self.log_likelihood_history_ = log_likelihood_history
        self.labels_ = np.argmax(responsibilities, axis=1)
        self.cluster_count_ = int(len(np.unique(self.labels_)))
        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        log_prob = np.zeros((len(X), self.n_components), dtype=float)
        for component in range(self.n_components):
            log_prob[:, component] = np.log(self.weights_[component] + 1e-12) + gaussian_log_pdf(
                X,
                self.means_[component],
                self.covariances_[component],
                self.covariance_type,
            )
        row_max = log_prob.max(axis=1, keepdims=True)
        stabilized = np.exp(log_prob - row_max)
        return stabilized / stabilized.sum(axis=1, keepdims=True)


def pairwise_distance_sq(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    a_sq = np.sum(A ** 2, axis=1).reshape(-1, 1)
    b_sq = np.sum(B ** 2, axis=1).reshape(1, -1)
    return np.maximum(a_sq + b_sq - 2.0 * A.dot(B.T), 0.0)


def cluster_distance(left_members, right_members, point_dist, linkage, left_mean, right_mean, left_size, right_size):
    block = point_dist[np.ix_(left_members, right_members)]
    if linkage == "single":
        return float(block.min())
    if linkage == "complete":
        return float(block.max())
    if linkage == "ward":
        diff = left_mean - right_mean
        return float((left_size * right_size) / float(left_size + right_size) * np.dot(diff, diff))
    return float(block.mean())


def kmeans_plus_plus_init(X, n_clusters, rng):
    centers = [X[rng.randint(0, len(X))]]
    for _ in range(1, n_clusters):
        dist_sq = np.min(pairwise_distance_sq(X, np.array(centers)), axis=1)
        probability = dist_sq / np.sum(dist_sq)
        centers.append(X[rng.choice(len(X), p=probability)])
    return np.array(centers, dtype=float)


def gaussian_log_pdf(X, mean, covariance, covariance_type):
    X = np.asarray(X, dtype=float)
    diff = X - mean
    if covariance_type == "diag":
        variance = covariance
        log_det = np.sum(np.log(variance))
        quadratic = np.sum((diff ** 2) / variance, axis=1)
        dimension = X.shape[1]
        return -0.5 * (dimension * np.log(2.0 * np.pi) + log_det + quadratic)

    cov = covariance
    sign, log_det = np.linalg.slogdet(cov)
    if sign <= 0:
        cov = cov + np.eye(cov.shape[0]) * 1e-4
        sign, log_det = np.linalg.slogdet(cov)
    inv_cov = np.linalg.pinv(cov)
    quadratic = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    dimension = X.shape[1]
    return -0.5 * (dimension * np.log(2.0 * np.pi) + log_det + quadratic)
