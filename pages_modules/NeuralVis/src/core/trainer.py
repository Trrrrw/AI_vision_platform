import numpy as np
from .neural_network import SimpleNN
from .dataset import generate_dataset


class Trainer:
    """训练控制器，支持激活函数与损失函数切换。"""

    def __init__(
        self,
        task: str = "分类",
        layer_sizes: list = None,
        activation: str = "relu",
        loss_type: str = "cross_entropy",
        lr: float = 0.01,
        epochs: int = 100,
        batch_size: int = 32,
    ):
        self.task = task
        self.layer_sizes = layer_sizes or [2, 4, 1]
        self.activation = activation
        self.loss_type = loss_type
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size

        self.nn = SimpleNN(self.layer_sizes, activation=activation, loss_type=loss_type)
        self.X, self.y = generate_dataset(task, input_dim=self.layer_sizes[0])
        self.current_epoch = 0
        self.loss_history = []
        self.acc_history = []

    def reset(self):
        self.nn = SimpleNN(self.layer_sizes, activation=self.activation, loss_type=self.loss_type)
        self.X, self.y = generate_dataset(self.task, input_dim=self.layer_sizes[0])
        self.current_epoch = 0
        self.loss_history.clear()
        self.acc_history.clear()

    def step(self):
        """单步训练，返回包含可视化所需全部信息的字典。"""
        output = self.nn.forward(self.X)
        loss = self._compute_loss(output, self.y)
        self.nn.backward(self.X, self.y)
        self.nn.update(self.lr)

        self.loss_history.append(float(loss))
        self.current_epoch += 1

        acc = None
        if self.task == "分类":
            acc = float(np.mean((output > 0.5).astype(int) == self.y))
            self.acc_history.append(acc)

        return {
            "epoch": self.current_epoch,
            "loss": float(loss),
            "accuracy": acc,
            "weights": [w.copy() for w in self.nn.W],
            "biases": [b.copy() for b in self.nn.b],
            "grads_W": [g.copy() for g in self.nn.grads_W],
            "grads_b": [g.copy() for g in self.nn.grads_b],
            "activations": [a.copy() for a in self.nn.a],
            "zs": [z.copy() for z in self.nn.z],
        }

    def _compute_loss(self, output, y):
        m = y.shape[0]
        eps = 1e-8
        if self.loss_type == "cross_entropy":
            return -np.mean(y * np.log(output + eps) + (1 - y) * np.log(1 - output + eps))
        elif self.loss_type == "mse":
            return np.mean((output - y) ** 2)
        elif self.loss_type == "mae":
            return np.mean(np.abs(output - y))
        else:
            return -np.mean(y * np.log(output + eps) + (1 - y) * np.log(1 - output + eps))
