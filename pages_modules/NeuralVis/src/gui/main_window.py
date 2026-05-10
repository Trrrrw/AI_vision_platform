from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSplitter, QToolBar, QMessageBox, QFileDialog
)
from PySide6.QtCore import Qt, QTimer
from .components.canvas_view import CanvasView
from .components.parameter_panel import ParameterPanel
from .components.result_panel import ResultPanel
from .components.theory_panel import TheoryPanel
from ..core.trainer import Trainer


class MainWindow(QMainWindow):
    def __init__(self, initial_params: dict = None):
        super().__init__()
        self.setWindowTitle("NeuralVis - AI算法可视化学习模块")
        self.resize(1440, 960)
        self.setStyleSheet("background: #0f172a;")

        self.trainer = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._training_step)
        self.current_params = None

        self._init_ui()
        self._init_toolbar()
        self._connect_signals()

        # 如果提供了外部初始参数，自动构建网络
        if initial_params:
            self._apply_external_params(initial_params)

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # 顶部工具栏
        self.toolbar = QToolBar()
        self.toolbar.setStyleSheet("""
            QToolBar {
                background: #1e293b;
                border-bottom: 1px solid #334155;
                padding: 4px;
                spacing: 8px;
            }
            QPushButton {
                background: #334155;
                color: #f8fafc;
                border: none;
                border-radius: 4px;
                padding: 6px 14px;
            }
            QPushButton:hover {
                background: #475569;
            }
            QPushButton:pressed {
                background: #64748b;
            }
        """)
        self.addToolBar(self.toolbar)

        # 主体水平分割：画布区域 + 右侧参数面板
        h_splitter = QSplitter(Qt.Horizontal)

        # 中间区域：画布 + 底部分割
        mid_widget = QWidget()
        mid_layout = QVBoxLayout(mid_widget)
        mid_layout.setContentsMargins(0, 0, 0, 0)
        mid_layout.setSpacing(0)

        # 原理讲解面板（放在画布上方，重置/保存按钮下方）
        self.theory_panel = TheoryPanel()
        mid_layout.addWidget(self.theory_panel)

        # 垂直分割：画布 + 结果面板
        v_splitter = QSplitter(Qt.Vertical)
        self.canvas = CanvasView()
        v_splitter.addWidget(self.canvas)
        self.result_panel = ResultPanel()
        v_splitter.addWidget(self.result_panel)
        v_splitter.setSizes([700, 220])
        mid_layout.addWidget(v_splitter, stretch=1)

        h_splitter.addWidget(mid_widget)

        # 右侧参数面板
        self.parameter_panel = ParameterPanel()
        h_splitter.addWidget(self.parameter_panel)
        h_splitter.setSizes([1100, 260])

        vbox.addWidget(h_splitter, stretch=1)

    def _init_toolbar(self):
        btn_reset = QPushButton("重置")
        btn_save = QPushButton("保存")
        self.btn_run_pause = QPushButton("运行")
        self.btn_step = QPushButton("单步执行")
        self.toolbar.addWidget(btn_reset)
        self.toolbar.addWidget(btn_save)
        self.toolbar.addWidget(self.btn_run_pause)
        self.toolbar.addWidget(self.btn_step)

        btn_reset.clicked.connect(self._on_reset_network)
        btn_save.clicked.connect(self._on_save)
        self.btn_run_pause.clicked.connect(self._on_run_pause)

    def _connect_signals(self):
        self.parameter_panel.params_changed.connect(self._on_params_changed)
        self.parameter_panel.build_network.connect(self._on_build_network)
        self.parameter_panel.reset_network.connect(self._on_reset_network)
        self.btn_step.clicked.connect(self._on_step)

    def _on_params_changed(self, params: dict):
        self.current_params = params
        hidden_str = " → ".join(str(n) for n in params["hidden_layers"])
        self.theory_panel.set_text(
            f"网络结构：输入层 {params['input_size']} 神经元 → 隐藏层 [{hidden_str}] → 输出层 {params['output_size']} 神经元\n"
            f"激活函数：{params['activation']} | 损失函数：{params['loss_type']} | 学习率：{params['lr']}"
        )

    def _on_build_network(self):
        params = self.parameter_panel.get_params()
        self.current_params = params
        layer_sizes = [params["input_size"]] + params["hidden_layers"] + [params["output_size"]]
        self.trainer = Trainer(
            task="分类",
            layer_sizes=layer_sizes,
            activation=params["activation"],
            loss_type=params["loss_type"],
            lr=params["lr"],
            epochs=1000,
            batch_size=32,
        )
        self.canvas.build_network_from_params(
            params["input_size"], params["hidden_layers"], params["output_size"],
            activation=params["activation"]
        )
        self.result_panel.clear_all()
        self.result_panel.append_log(
            f"[Build] 结构={layer_sizes}, 激活={params['activation']}, 损失={params['loss_type']}, lr={params['lr']}"
        )
        self.theory_panel.set_text(
            "网络已构建。神经网络通过层与层之间的全连接将输入数据映射到输出。\n"
            "每个连接都有一个权重，训练的目标就是不断调整这些权重，使预测结果更接近真实值。"
        )

    def _on_reset_network(self):
        self._stop_training()
        self.canvas.clear_canvas()
        if self.trainer:
            self.trainer.reset()
        self.trainer = None
        self.result_panel.clear_all()
        self.result_panel.append_log("[Reset] 网络与训练状态已重置。")
        self.theory_panel.set_text(
            "欢迎使用神经网络可视化学习工具！\n"
            "在右侧面板设置网络结构、激活函数、损失函数和学习率后，点击“构建网络”开始。"
        )

    def _on_step(self):
        if not self.trainer:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("提示")
            msg_box.setText("请先点击“构建网络”。")
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setStyleSheet("QMessageBox { background-color: white; } QLabel { background-color: white; color: black; } QPushButton { background-color: #e0e0e0; color: black; }")
            msg_box.exec()
            return
        self._stop_training()
        info = self.trainer.step()
        self._update_ui_from_step(info)
        # 单步执行时播放动画
        def show_weights():
            self.canvas._reset_neuron_colors()
            self.canvas.update_weights_visual(info["weights"])

        self.canvas.start_forward_animation(
            info["activations"],
            on_finished=lambda: self.canvas.start_backward_animation(
                info["grads_W"],
                on_finished=show_weights
            )
        )

    def _on_run_pause(self):
        if not self.trainer:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("提示")
            msg_box.setText("请先点击“构建网络”。")
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.setStyleSheet("QMessageBox { background-color: white; } QLabel { background-color: white; color: black; } QPushButton { background-color: #e0e0e0; color: black; }")
            msg_box.exec()
            return
        if self.timer.isActive():
            self._stop_training()
            self.result_panel.append_log("[Pause] 训练已暂停。")
        else:
            self.btn_run_pause.setText("暂停")
            self.timer.start(300)  # 每 300ms 一步
            self.result_panel.append_log("[Run] 开始自动训练。")

    def _stop_training(self):
        self.timer.stop()
        self.btn_run_pause.setText("运行")

    def _training_step(self):
        if not self.trainer:
            return
        if self.trainer.current_epoch >= self.trainer.epochs:
            self._stop_training()
            self.result_panel.append_log("[Done] 训练已完成。")
            return

        info = self.trainer.step()
        self._update_ui_from_step(info)
        # 自动训练时直接刷新可视化（不播放逐层动画，保证流畅）
        self.canvas.update_forward_visual(info["activations"])
        self.canvas.update_backward_visual(info["grads_W"], info["zs"])
        self.canvas.update_weights_visual(info["weights"])

    def _update_ui_from_step(self, info):
        self.result_panel.update_metrics(
            info["epoch"], info["loss"], info.get("accuracy")
        )
        self.result_panel.update_loss_curve(self.trainer.loss_history)
        self.result_panel.update_weights(info["weights"])
        self.result_panel.append_log(
            f"[Step] Epoch {info['epoch']} | Loss {info['loss']:.6f}"
            + (f" | Acc {info['accuracy']:.4f}" if info.get("accuracy") is not None else "")
        )
        self._update_theory_by_epoch(info["epoch"])

    def _update_theory_by_epoch(self, epoch: int):
        if epoch == 1:
            self.theory_panel.set_text(
                "前向传播：数据从输入层经过隐藏层流向输出层，每层通过加权求和与激活函数进行非线性变换，最终得到预测值。"
            )
        elif epoch == 2:
            self.theory_panel.set_text(
                "反向传播：计算预测值与真实值之间的误差（损失），然后从输出层向输入层逐层传递梯度，更新每个权重。"
            )
        elif epoch == 3:
            self.theory_panel.set_text(
                "权重更新：根据梯度和学习率调整权重。学习率决定了每次更新的步长，太大可能震荡，太小收敛慢。"
            )
        elif epoch == 4:
            self.theory_panel.set_text(
                "整体循环：神经网络不断重复“前向传播 → 计算损失 → 反向传播 → 权重更新”的过程，直到损失收敛。"
            )

    def _apply_external_params(self, params: dict):
        """从外部（如 Streamlit 导出的 JSON）加载参数并自动构建网络。"""
        # 更新参数面板
        self.parameter_panel.set_params(params)
        self.current_params = params
        # 自动构建网络
        self._on_build_network()
        self.result_panel.append_log("[Load] 已从外部配置加载参数并构建网络。")

    def _on_save(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存训练状态", "", "Text Files (*.txt)")
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    if self.trainer:
                        f.write(f"Epoch: {self.trainer.current_epoch}\n")
                        f.write(f"Loss History: {self.trainer.loss_history}\n")
                        f.write(f"Accuracy History: {self.trainer.acc_history}\n")
                    else:
                        f.write("No training data.\n")
                self.result_panel.append_log(f"[Save] 已保存到 {path}")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", str(e))
