import numpy as np


def generate_dataset(task: str, input_dim: int = 2, n_samples: int = 200, noise: float = 0.1):
    """生成示例数据集，特征维度自动匹配输入层神经元数。"""
    rng = np.random.default_rng(42)

    if task == "分类":
        # 二分类
        n = n_samples // 2
        if input_dim == 2:
            # 同心圆（二维可视化友好）
            r1 = rng.normal(2.0, 0.5, n)
            theta1 = rng.uniform(0, 2 * np.pi, n)
            x1 = np.column_stack((r1 * np.cos(theta1), r1 * np.sin(theta1)))
            r2 = rng.normal(5.0, 0.8, n)
            theta2 = rng.uniform(0, 2 * np.pi, n)
            x2 = np.column_stack((r2 * np.cos(theta2), r2 * np.sin(theta2)))
        else:
            # 多维高斯
            c1 = rng.normal(0, 1, input_dim)
            c2 = rng.normal(2, 1, input_dim)
            x1 = rng.multivariate_normal(c1, np.eye(input_dim) * 0.5, n)
            x2 = rng.multivariate_normal(c2, np.eye(input_dim) * 0.8, n)

        y1 = np.zeros(n)
        y2 = np.ones(n)
        X = np.vstack((x1, x2))
        y = np.hstack((y1, y2)).reshape(-1, 1)
        return X, y

    elif task == "回归":
        X = rng.uniform(-3, 3, (n_samples, input_dim))
        y = np.sin(X[:, 0]) + rng.normal(0, noise, n_samples)
        return X, y.reshape(-1, 1)

    elif task == "聚类":
        n_per_class = n_samples // 3
        if input_dim == 2:
            centers = np.array([[-2, -2], [2, 2], [2, -2]])
        else:
            centers = rng.uniform(-3, 3, (3, input_dim))
        X = []
        y = []
        for i, c in enumerate(centers):
            pts = rng.multivariate_normal(c, np.eye(input_dim) * 0.5, n_per_class)
            X.append(pts)
            y.append(np.full(len(pts), i))
        X = np.vstack(X)
        y = np.hstack(y).reshape(-1, 1)
        return X, y

    else:
        raise ValueError(f"Unknown task: {task}")
