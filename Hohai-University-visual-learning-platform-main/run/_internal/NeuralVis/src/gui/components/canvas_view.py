import numpy as np
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem
from PySide6.QtCore import Qt, QPointF, QTimer
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont


class CanvasView(QGraphicsView):
    """中心可视化画布：自动构建网络、展示前向/反向传播动画、权重可视化。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(-2000, -2000, 4000, 4000)
        self.setScene(self.scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setBackgroundBrush(QBrush(QColor("#0f172a")))

        # 存储可视化元素引用
        self.layer_texts = []          # 每层标题 QGraphicsTextItem
        self.neuron_items = []         # [层][神经元] QGraphicsEllipseItem
        self.neuron_labels = []        # [层][神经元] QGraphicsTextItem
        self.connection_items = []     # [层][前层神经元][当前层神经元] QGraphicsLineItem

        self._draw_grid()

        # 动画相关
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._anim_step)
        self._anim_queue = []

    def _draw_grid(self):
        pen = QPen(QColor("#1e293b"))
        pen.setWidth(1)
        step = 40
        for x in range(-2000, 2001, step):
            self.scene.addLine(x, -2000, x, 2000, pen)
        for y in range(-2000, 2001, step):
            self.scene.addLine(-2000, y, 2000, y, pen)

    def clear_canvas(self):
        self.scene.clear()
        self.layer_texts.clear()
        self.neuron_items.clear()
        self.neuron_labels.clear()
        self.connection_items.clear()
        self._draw_grid()

    def build_network_from_params(self, input_size: int, hidden_layers: list, output_size: int, activation: str = "relu"):
        """根据参数自动构建可视化网络层级，并保存元素引用。"""
        self.clear_canvas()
        layer_sizes = [input_size] + hidden_layers + [output_size]
        n_layers = len(layer_sizes)
        x_spacing = 220
        y_spacing = 70
        x_start = -((n_layers - 1) * x_spacing) / 2

        self.neuron_items = []
        self.neuron_labels = []
        self.connection_items = []
        self.layer_texts = []

        neuron_radius = 14
        layer_positions = []

        for i, size in enumerate(layer_sizes):
            x = x_start + i * x_spacing
            total_h = max(160, size * y_spacing)
            y_start_layer = -total_h / 2
            layer_positions.append((x, y_start_layer, total_h))

            # 层标题
            layer_name = "输入层" if i == 0 else ("输出层" if i == n_layers - 1 else f"隐藏层 {i}")
            title = self.scene.addText(f"{layer_name}\n({size} neurons)", QFont("Microsoft YaHei", 10, QFont.Bold))
            title.setDefaultTextColor(QColor("#94a3b8"))
            title.setPos(x - title.boundingRect().width() / 2, y_start_layer - 50)
            self.layer_texts.append(title)


            n_items = []
            l_items = []
            for n in range(size):
                y = y_start_layer + n * y_spacing + y_spacing / 2
                # 神经元圆点
                ellipse = self.scene.addEllipse(
                    x - neuron_radius, y - neuron_radius,
                    neuron_radius * 2, neuron_radius * 2,
                    QPen(QColor("#475569")), QBrush(QColor("#334155"))
                )
                ellipse.setZValue(2)
                n_items.append(ellipse)

                # 数值标签（位于神经元右侧）
                label = self.scene.addText("", QFont("Microsoft YaHei", 8))
                label.setDefaultTextColor(QColor("#e2e8f0"))
                label.setPos(x + neuron_radius + 6, y - label.boundingRect().height() / 2)
                l_items.append(label)

            self.neuron_items.append(n_items)
            self.neuron_labels.append(l_items)

        # 层间连线
        pen_default = QPen(QColor("#475569"))
        pen_default.setWidth(1)
        for i in range(1, n_layers):
            prev_size = layer_sizes[i - 1]
            curr_size = layer_sizes[i]
            x_prev = layer_positions[i - 1][0]
            x_curr = layer_positions[i][0]
            y_start_prev = layer_positions[i - 1][1]
            y_start_curr = layer_positions[i][1]

            conn_layer = []
            for pn in range(prev_size):
                py = y_start_prev + pn * y_spacing + y_spacing / 2
                row = []
                for cn in range(curr_size):
                    cy = y_start_curr + cn * y_spacing + y_spacing / 2
                    line = self.scene.addLine(x_prev + neuron_radius, py, x_curr - neuron_radius, cy, pen_default)
                    line.setZValue(-1)
                    row.append(line)
                conn_layer.append(row)
            self.connection_items.append(conn_layer)

    def update_forward_visual(self, activations):
        """前向传播可视化：更新神经元颜色与数值（取第一个样本）。"""
        for layer_idx, neurons in enumerate(self.neuron_items):
            if layer_idx >= len(activations):
                continue
            vals = activations[layer_idx]
            display = vals[0] if vals.ndim > 1 else vals  # 取第一个样本
            for n_idx, neuron in enumerate(neurons):
                val = float(display[n_idx])
                color = self._value_to_color(val, mode="forward")
                neuron.setBrush(QBrush(color))
                neuron.setPen(QPen(QColor("#94a3b8"), 1))
                self.neuron_labels[layer_idx][n_idx].setPlainText(f"{val:.2f}")

        # 高亮数据流连线（所有正向连线短暂变亮）
        for layer_conn in self.connection_items:
            for row in layer_conn:
                for line in row:
                    line.setPen(QPen(QColor("#38bdf8"), 1.5))
        # 300ms 后恢复权重颜色（由主窗口在收到 weights 后调用 update_weights_visual 恢复）

    def update_backward_visual(self, grads_W, zs):
        """反向传播可视化：展示误差/梯度回传。用红色系表示梯度大小。"""
        n_layers = len(self.neuron_items)
        # 神经元颜色：基于该层预激活 z 的梯度绝对值平均值
        for layer_idx in range(n_layers):
            if layer_idx == 0:
                continue  # 输入层通常不显示梯度
            grad_idx = layer_idx - 1  # grads_W 的索引
            if grad_idx >= len(grads_W):
                continue
            # 估算每个神经元的梯度：取 grads_W[grad_idx] 对应输出的平均绝对值
            g = grads_W[grad_idx]  # shape: (prev_size, curr_size)
            per_neuron_grad = np.mean(np.abs(g), axis=0)  # (curr_size,)
            max_g = float(np.max(per_neuron_grad)) + 1e-8
            for n_idx, neuron in enumerate(self.neuron_items[layer_idx]):
                intensity = float(per_neuron_grad[n_idx]) / max_g
                color = self._value_to_color(intensity, mode="backward")
                neuron.setBrush(QBrush(color))
                neuron.setPen(QPen(QColor("#fca5a5"), 1))
                self.neuron_labels[layer_idx][n_idx].setPlainText(f"∇{intensity:.3f}")

        # 连线颜色：基于对应权重的梯度绝对值
        for i, layer_conn in enumerate(self.connection_items):
            g = grads_W[i]
            max_g = float(np.max(np.abs(g))) + 1e-8
            for pn, row in enumerate(layer_conn):
                for cn, line in enumerate(row):
                    intensity = float(np.abs(g[pn, cn])) / max_g
                    width = 1 + intensity * 3
                    color = self._value_to_color(intensity, mode="backward")
                    line.setPen(QPen(color, width))

    def update_weights_visual(self, weights):
        """权重可视化：用连线粗细和蓝/红颜色表示权重大小与正负。"""
        for i, layer_conn in enumerate(self.connection_items):
            w = weights[i]
            max_w = float(np.max(np.abs(w))) + 1e-8
            for pn, row in enumerate(layer_conn):
                for cn, line in enumerate(row):
                    val = float(w[pn, cn])
                    intensity = abs(val) / max_w
                    width = 1 + intensity * 3
                    # 正权重偏蓝，负权重偏红
                    if val >= 0:
                        r = int(30 + intensity * 80)
                        g = int(41 + intensity * 150)
                        b = int(59 + intensity * 190)
                    else:
                        r = int(30 + intensity * 220)
                        g = int(41 + intensity * 70)
                        b = int(59 + intensity * 80)
                    color = QColor(r, g, b)
                    line.setPen(QPen(color, width))

    def _value_to_color(self, val: float, mode: str = "forward") -> QColor:
        """将数值映射为颜色。forward 用蓝色系，backward 用红色系。"""
        val = max(0.0, min(1.0, abs(val)))
        if mode == "forward":
            # 从 #334155 (51,65,85) 到 #38bdf8 (56,189,248)
            r = int(51 + val * (56 - 51))
            g = int(65 + val * (189 - 65))
            b = int(85 + val * (248 - 85))
        else:
            # 从 #334155 到 #f87171 (248,113,113)
            r = int(51 + val * (248 - 51))
            g = int(65 + val * (113 - 65))
            b = int(85 + val * (113 - 85))
        return QColor(r, g, b)

    # ========== 简单逐层动画接口 ==========
    def start_forward_animation(self, activations, on_finished=None):
        """逐层播放前向传播高亮动画。"""
        self._anim_type = "forward"
        self._anim_data = activations
        self._anim_stage = 0
        self._anim_on_finished = on_finished
        self._anim_timer.start(250)

    def start_backward_animation(self, grads_W, on_finished=None):
        """逐层反向播放反向传播高亮动画。"""
        self._anim_type = "backward"
        self._anim_data = grads_W
        self._anim_stage = len(self.neuron_items) - 1
        self._anim_on_finished = on_finished
        self._anim_timer.start(250)

    def _anim_step(self):
        if self._anim_type == "forward":
            stage = self._anim_stage
            if stage >= len(self.neuron_items):
                self._anim_timer.stop()
                if self._anim_on_finished:
                    self._anim_on_finished()
                return
            # 只高亮当前层
            self._reset_neuron_colors()
            vals = self._anim_data[stage]
            display = vals[0] if vals.ndim > 1 else vals
            for n_idx, neuron in enumerate(self.neuron_items[stage]):
                val = float(display[n_idx])
                neuron.setBrush(QBrush(self._value_to_color(val, "forward")))
                neuron.setPen(QPen(QColor("#f8fafc"), 2))
                self.neuron_labels[stage][n_idx].setPlainText(f"{val:.2f}")
            # 高亮进入当前层的连线
            if stage > 0 and stage - 1 < len(self.connection_items):
                for row in self.connection_items[stage - 1]:
                    for line in row:
                        line.setPen(QPen(QColor("#38bdf8"), 2))
            self._anim_stage += 1
        elif self._anim_type == "backward":
            stage = self._anim_stage
            if stage < 1:
                self._anim_timer.stop()
                if self._anim_on_finished:
                    self._anim_on_finished()
                return
            self._reset_neuron_colors()
            g = self._anim_data[stage - 1]
            per_neuron_grad = np.mean(np.abs(g), axis=0)
            max_g = float(np.max(per_neuron_grad)) + 1e-8
            for n_idx, neuron in enumerate(self.neuron_items[stage]):
                intensity = float(per_neuron_grad[n_idx]) / max_g
                neuron.setBrush(QBrush(self._value_to_color(intensity, "backward")))
                neuron.setPen(QPen(QColor("#fca5a5"), 2))
                self.neuron_labels[stage][n_idx].setPlainText(f"∇{intensity:.3f}")
            # 高亮离开当前层的连线（往上一层）
            if stage - 1 < len(self.connection_items):
                for row in self.connection_items[stage - 1]:
                    for line in row:
                        line.setPen(QPen(QColor("#f87171"), 2))
            self._anim_stage -= 1

    def _reset_neuron_colors(self):
        for neurons in self.neuron_items:
            for neuron in neurons:
                neuron.setBrush(QBrush(QColor("#334155")))
                neuron.setPen(QPen(QColor("#475569"), 1))
