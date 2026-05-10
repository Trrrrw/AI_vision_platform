import numpy as np


def build_clustering_metrics(X, labels):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    valid_mask = labels >= 0
    clustered_X = X[valid_mask]
    clustered_labels = labels[valid_mask]
    unique_labels = np.unique(clustered_labels)

    result = {
        "cluster_count": int(len(unique_labels)),
        "noise_count": int(np.sum(labels < 0)),
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }

    if len(clustered_X) < 3 or len(unique_labels) < 2:
        return result

    result["silhouette"] = silhouette_score(clustered_X, clustered_labels)
    result["davies_bouldin"] = davies_bouldin_score(clustered_X, clustered_labels)
    result["calinski_harabasz"] = calinski_harabasz_score(clustered_X, clustered_labels)
    return result


def silhouette_score(X, labels):
    distance_matrix = np.sqrt(pairwise_distance_sq(X, X))
    unique_labels = np.unique(labels)
    scores = []
    for index in range(len(X)):
        same_mask = labels == labels[index]
        same_mask[index] = False
        if np.any(same_mask):
            a_value = distance_matrix[index, same_mask].mean()
        else:
            a_value = 0.0

        b_value = None
        for label in unique_labels:
            if label == labels[index]:
                continue
            other_mask = labels == label
            if not np.any(other_mask):
                continue
            candidate = distance_matrix[index, other_mask].mean()
            if b_value is None or candidate < b_value:
                b_value = candidate
        if b_value is None:
            continue
        denominator = max(a_value, b_value, 1e-9)
        scores.append((b_value - a_value) / denominator)

    if not scores:
        return None
    return float(np.mean(scores))


def davies_bouldin_score(X, labels):
    unique_labels = np.unique(labels)
    centroids = []
    scatters = []
    for label in unique_labels:
        cluster = X[labels == label]
        centroid = cluster.mean(axis=0)
        centroids.append(centroid)
        scatters.append(np.mean(np.sqrt(np.sum((cluster - centroid) ** 2, axis=1))))

    centroids = np.array(centroids)
    scatters = np.array(scatters)
    centroid_distance = np.sqrt(pairwise_distance_sq(centroids, centroids))

    scores = []
    for i in range(len(unique_labels)):
        ratios = []
        for j in range(len(unique_labels)):
            if i == j:
                continue
            distance = centroid_distance[i, j]
            if distance < 1e-9:
                continue
            ratios.append((scatters[i] + scatters[j]) / distance)
        if ratios:
            scores.append(max(ratios))
    if not scores:
        return None
    return float(np.mean(scores))


def calinski_harabasz_score(X, labels):
    unique_labels = np.unique(labels)
    n_samples = len(X)
    n_clusters = len(unique_labels)
    overall_mean = X.mean(axis=0)

    between = 0.0
    within = 0.0
    for label in unique_labels:
        cluster = X[labels == label]
        centroid = cluster.mean(axis=0)
        between += len(cluster) * np.sum((centroid - overall_mean) ** 2)
        within += np.sum((cluster - centroid) ** 2)

    if within < 1e-12 or n_clusters <= 1 or n_samples <= n_clusters:
        return None
    numerator = between / (n_clusters - 1)
    denominator = within / (n_samples - n_clusters)
    return float(numerator / max(denominator, 1e-12))


def pairwise_distance_sq(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    a_sq = np.sum(A ** 2, axis=1).reshape(-1, 1)
    b_sq = np.sum(B ** 2, axis=1).reshape(1, -1)
    return np.maximum(a_sq + b_sq - 2.0 * A.dot(B.T), 0.0)
