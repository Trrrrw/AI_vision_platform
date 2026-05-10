import numpy as np


class SimpleNN:
    """简单的全连接神经网络，用于教学演示。支持切换激活函数与损失函数。"""

    def __init__(self, layer_sizes: list, activation: str = "relu", loss_type: str = "cross_entropy"):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.activation_name = activation
        self.loss_type = loss_type

        self.W = []
        self.b = []
        for i in range(self.n_layers - 1):
            # He init for ReLU, Xavier for sigmoid/tanh/linear
            if activation == "relu":
                scale = np.sqrt(2.0 / layer_sizes[i])
            else:
                scale = np.sqrt(1.0 / layer_sizes[i])
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros((1, layer_sizes[i + 1]))
            self.W.append(w)
            self.b.append(b)

        self.z = []
        self.a = []
        self.grads_W = [np.zeros_like(w) for w in self.W]
        self.grads_b = [np.zeros_like(b) for b in self.b]

    def _activation(self, x):
        if self.activation_name == "relu":
            return np.maximum(0, x)
        elif self.activation_name == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation_name == "tanh":
            return np.tanh(x)
        elif self.activation_name == "linear":
            return x
        else:
            return x

    def _activation_derivative(self, x):
        if self.activation_name == "relu":
            return (x > 0).astype(float)
        elif self.activation_name == "sigmoid":
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        elif self.activation_name == "tanh":
            return 1 - np.tanh(x) ** 2
        elif self.activation_name == "linear":
            return np.ones_like(x)
        else:
            return np.ones_like(x)

    def _output_activation(self, x):
        """输出层激活：交叉熵配合 sigmoid，回归配合线性（恒等）。"""
        if self.loss_type == "cross_entropy":
            return 1 / (1 + np.exp(-x))
        else:
            return x

    def _output_activation_derivative(self, x):
        """输出层激活导数。"""
        if self.loss_type == "cross_entropy":
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        else:
            return np.ones_like(x)

    def forward(self, X):
        """前向传播，缓存中间结果供反向传播与可视化使用。"""
        self.a = [X]
        self.z = []
        out = X
        for i in range(self.n_layers - 1):
            z = out @ self.W[i] + self.b[i]
            self.z.append(z)
            if i < self.n_layers - 2:
                # 隐藏层使用用户选择的激活函数
                out = self._activation(z)
            else:
                # 输出层根据损失函数决定激活
                out = self._output_activation(z)
            self.a.append(out)
        return out

    def backward(self, X, y):
        """反向传播，支持 cross_entropy / mse / mae 三种损失。"""
        m = X.shape[0]
        a_out = self.a[-1]
        z_out = self.z[-1]

        # 计算输出层 delta = dL/dz_out
        # 注意：delta 不含 1/m 的缩放，因为后续 dW 计算会统一除以 m
        if self.loss_type == "cross_entropy":
            # 输出层为 sigmoid，binary cross entropy 的梯度简化为 a_out - y
            delta = a_out - y
        elif self.loss_type == "mse":
            # 输出层为线性，MSE 的梯度链式法则
            delta = (a_out - y) * 2 * self._output_activation_derivative(z_out)
        elif self.loss_type == "mae":
            # 输出层为线性，MAE 的梯度链式法则
            delta = np.sign(a_out - y) * self._output_activation_derivative(z_out)
        else:
            delta = a_out - y

        grads_W = []
        grads_b = []
        for i in range(self.n_layers - 2, -1, -1):
            dW = (self.a[i].T @ delta) / m
            db = np.mean(delta, axis=0, keepdims=True)
            grads_W.insert(0, dW)
            grads_b.insert(0, db)

            if i > 0:
                delta = (delta @ self.W[i].T) * self._activation_derivative(self.z[i - 1])

        self.grads_W = grads_W
        self.grads_b = grads_b
        return grads_W, grads_b

    def update(self, lr: float):
        """梯度下降更新权重与偏置。"""
        for i in range(self.n_layers - 1):
            self.W[i] -= lr * self.grads_W[i]
            self.b[i] -= lr * self.grads_b[i]

    def get_weights(self):
        return self.W

    def get_biases(self):
        return self.b

    def get_gradients(self):
        return self.grads_W, self.grads_b
