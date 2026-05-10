from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSpinBox, QDoubleSpinBox,
    QPushButton, QFormLayout, QGroupBox, QHBoxLayout, QComboBox, QLineEdit
)
from PySide6.QtCore import Signal
from PySide6.QtGui import QPainter, QColor, QPolygonF
from PySide6.QtWidgets import QProxyStyle, QStyle, QStyleOption


class SpinBoxArrowStyle(QProxyStyle):
    def __init__(self):
        super().__init__()

    def drawPrimitive(self, element, option, painter, widget=None):
        if element == QStyle.PE_IndicatorSpinUp:
            self._draw_up_arrow(painter, option.rect)
        elif element == QStyle.PE_IndicatorSpinDown:
            self._draw_down_arrow(painter, option.rect)
        else:
            super().drawPrimitive(element, option, painter, widget)

    def _draw_up_arrow(self, painter: QPainter, rect):
        painter.save()
        cx = rect.center().x()
        y_top = rect.top() + 3
        y_bottom = rect.bottom() - 1
        half_w = 4
        arrow = QPolygonF([
            QPointF(cx - half_w, y_bottom),
            QPointF(cx + half_w, y_bottom),
            QPointF(cx, y_top),
        ])
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#f8fafc"))
        painter.drawPolygon(arrow)
        painter.restore()

    def _draw_down_arrow(self, painter: QPainter, rect):
        painter.save()
        cx = rect.center().x()
        y_top = rect.top() + 1
        y_bottom = rect.bottom() - 3
        half_w = 4
        arrow = QPolygonF([
            QPointF(cx - half_w, y_top),
            QPointF(cx + half_w, y_top),
            QPointF(cx, y_bottom),
        ])
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#f8fafc"))
        painter.drawPolygon(arrow)
        painter.restore()


class ParameterPanel(QWidget):
    """右侧参数控制面板（精简版）：仅保留网络结构、激活函数、损失函数、学习率。"""

    params_changed = Signal(dict)
    build_network = Signal()
    reset_network = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(240)
        self.setMaximumWidth(320)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(14)

        title = QLabel("参数控制")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #f8fafc;")
        layout.addWidget(title)

        # 网络结构
        net_box = QGroupBox("网络结构")
        net_form = QFormLayout(net_box)
        net_form.setSpacing(10)

        self.input_spin = QSpinBox()
        self.input_spin.setRange(1, 64)
        self.input_spin.setValue(2)
        net_form.addRow("输入维度:", self.input_spin)

        self.hidden_layers_spin = QSpinBox()
        self.hidden_layers_spin.setRange(1, 5)
        self.hidden_layers_spin.setValue(2)
        net_form.addRow("隐藏层层数:", self.hidden_layers_spin)

        self.hidden_neurons_container = QWidget()
        self.hidden_neurons_layout = QVBoxLayout(self.hidden_neurons_container)
        self.hidden_neurons_layout.setContentsMargins(0, 0, 0, 0)
        self.hidden_neurons_layout.setSpacing(6)
        net_form.addRow(self.hidden_neurons_container)

        self._spin_style = SpinBoxArrowStyle()

        self._rebuild_hidden_neuron_spins(2)
        self.hidden_layers_spin.valueChanged.connect(self._on_hidden_layers_changed)

        self.output_spin = QSpinBox()
        self.output_spin.setRange(1, 64)
        self.output_spin.setValue(1)
        net_form.addRow("输出层神经元:", self.output_spin)

        layout.addWidget(net_box)

        # 函数与优化
        func_box = QGroupBox("函数与训练")
        func_form = QFormLayout(func_box)
        func_form.setSpacing(10)

        self.activation_combo = QComboBox()
        self.activation_combo.addItems(["ReLU", "Sigmoid", "Tanh", "Linear"])
        func_form.addRow("激活函数:", self.activation_combo)

        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["交叉熵", "MSE", "MAE"])
        func_form.addRow("损失函数:", self.loss_combo)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setValue(0.1)
        func_form.addRow("学习率 η:", self.lr_spin)

        layout.addWidget(func_box)

        # 按钮
        btn_layout = QHBoxLayout()
        self.build_btn = QPushButton("构建网络")
        self.reset_btn = QPushButton("重置")
        btn_layout.addWidget(self.build_btn)
        btn_layout.addWidget(self.reset_btn)
        layout.addLayout(btn_layout)

        layout.addStretch()

        # 信号
        self.build_btn.clicked.connect(self._on_build)
        self.reset_btn.clicked.connect(self.reset_network.emit)

        # 为所有 SpinBox 应用自定义箭头样式
        self.input_spin.setStyle(self._spin_style)
        self.hidden_layers_spin.setStyle(self._spin_style)
        self.output_spin.setStyle(self._spin_style)
        self.lr_spin.setStyle(self._spin_style)

        self.setStyleSheet("""
            QWidget {
                color: #e2e8f0;
            }
            QGroupBox {
                border: 1px solid #334155;
                border-radius: 6px;
                margin-top: 8px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QComboBox {
                background: #1e293b;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 4px;
            }
            QSpinBox, QDoubleSpinBox {
                background: #1e293b;
                color: #f8fafc;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 4px;
                padding-right: 22px;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 20px;
                border: none;
                border-left: 1px solid #475569;
                background: #334155;
                border-top-right-radius: 4px;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
                background: #475569;
            }
            QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {
                background: #64748b;
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                width: 0px;
                height: 0px;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 20px;
                border: none;
                border-left: 1px solid #475569;
                border-top: 1px solid #475569;
                background: #334155;
                border-bottom-right-radius: 4px;
            }
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background: #475569;
            }
            QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
                background: #64748b;
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                width: 0px;
                height: 0px;
            }
            QLineEdit {
                background: #1e293b;
                border: 1px solid #475569;
                border-radius: 4px;
                padding: 4px;
            }
            QPushButton {
                background: #3b82f6;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                color: #fff;
            }
            QPushButton:hover {
                background: #2563eb;
            }
            QPushButton:pressed {
                background: #1d4ed8;
            }
        """)

    def _on_hidden_layers_changed(self, value: int):
        self._rebuild_hidden_neuron_spins(value)

    def _rebuild_hidden_neuron_spins(self, count: int):
        # 清空现有控件
        while self.hidden_neurons_layout.count():
            item = self.hidden_neurons_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.hidden_neuron_spins = []
        defaults = [8, 8, 8, 8, 8]
        for i in range(count):
            spin = QSpinBox()
            spin.setRange(1, 64)
            spin.setValue(defaults[i])
            spin.setStyle(self._spin_style)
            self.hidden_neuron_spins.append(spin)
            self.hidden_neurons_layout.addWidget(QLabel(f"第 {i + 1} 层神经元:"))
            self.hidden_neurons_layout.addWidget(spin)

    def _on_build(self):
        hidden_layers = [spin.value() for spin in self.hidden_neuron_spins]
        activation_map = {"ReLU": "relu", "Sigmoid": "sigmoid", "Tanh": "tanh", "Linear": "linear"}
        loss_map = {"交叉熵": "cross_entropy", "MSE": "mse", "MAE": "mae"}
        params = {
            "input_size": self.input_spin.value(),
            "hidden_layers": hidden_layers,
            "output_size": self.output_spin.value(),
            "activation": activation_map[self.activation_combo.currentText()],
            "loss_type": loss_map[self.loss_combo.currentText()],
            "lr": self.lr_spin.value(),
        }
        self.params_changed.emit(params)
        self.build_network.emit()

    def set_params(self, params: dict):
        """从外部配置加载参数到面板控件。"""
        # 输入/输出层
        self.input_spin.setValue(params.get("input_size", 2))
        self.output_spin.setValue(params.get("output_size", 1))

        # 隐藏层
        hidden_layers = params.get("hidden_layers", [8, 8])
        self.hidden_layers_spin.setValue(len(hidden_layers))
        self._rebuild_hidden_neuron_spins(len(hidden_layers))
        for spin, val in zip(self.hidden_neuron_spins, hidden_layers):
            spin.setValue(val)

        # 激活函数
        activation_reverse = {"relu": "ReLU", "sigmoid": "Sigmoid", "tanh": "Tanh", "linear": "Linear"}
        act_text = activation_reverse.get(params.get("activation", "relu"), "ReLU")
        idx = self.activation_combo.findText(act_text)
        if idx >= 0:
            self.activation_combo.setCurrentIndex(idx)

        # 损失函数
        loss_reverse = {"cross_entropy": "交叉熵", "mse": "MSE", "mae": "MAE"}
        loss_text = loss_reverse.get(params.get("loss_type", "cross_entropy"), "交叉熵")
        idx = self.loss_combo.findText(loss_text)
        if idx >= 0:
            self.loss_combo.setCurrentIndex(idx)

        # 学习率
        self.lr_spin.setValue(params.get("lr", 0.1))

    def get_params(self):
        hidden_layers = [spin.value() for spin in self.hidden_neuron_spins]
        activation_map = {"ReLU": "relu", "Sigmoid": "sigmoid", "Tanh": "tanh", "Linear": "linear"}
        loss_map = {"交叉熵": "cross_entropy", "MSE": "mse", "MAE": "mae"}
        return {
            "input_size": self.input_spin.value(),
            "hidden_layers": hidden_layers,
            "output_size": self.output_spin.value(),
            "activation": activation_map[self.activation_combo.currentText()],
            "loss_type": loss_map[self.loss_combo.currentText()],
            "lr": self.lr_spin.value(),
        }
