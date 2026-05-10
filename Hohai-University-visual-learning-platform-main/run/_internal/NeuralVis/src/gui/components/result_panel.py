from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QTabWidget
from PySide6.QtCore import Qt
import numpy as np

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MPL = True
except Exception:
    HAS_MPL = False


class ResultPanel(QWidget):
    """底部结果面板：左侧损失曲线，右侧指标/权重/日志。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.setMaximumHeight(320)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # 左侧：损失曲线
        if HAS_MPL:
            self.figure = Figure(figsize=(4, 2.5), facecolor="#0f172a")
            self.ax = self.figure.add_subplot(111)
            self.ax.set_facecolor("#0f172a")
            self.ax.tick_params(colors="#94a3b8")
            self.ax.spines["bottom"].set_color("#334155")
            self.ax.spines["top"].set_color("#334155")
            self.ax.spines["left"].set_color("#334155")
            self.ax.spines["right"].set_color("#334155")
            self.ax.set_title("Loss Curve", color="#e2e8f0", fontsize=10)
            self.ax.set_xlabel("Epoch", color="#94a3b8", fontsize=9)
            self.ax.set_ylabel("Loss", color="#94a3b8", fontsize=9)
            self.line_loss, = self.ax.plot([], [], color="#38bdf8", linewidth=1.5, label="Loss")
            self.ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#e2e8f0")
            self.canvas = FigureCanvas(self.figure)
            layout.addWidget(self.canvas, stretch=2)
        else:
            self.canvas = QLabel("matplotlib 未安装，无法显示损失曲线")
            self.canvas.setStyleSheet("color: #94a3b8;")
            layout.addWidget(self.canvas, stretch=2)
            self.figure = None

        # 右侧：Tab
        self.tabs = QTabWidget()

        self.metric_label = QLabel("Epoch: 0\nLoss: --\nAccuracy: --")
        self.metric_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.metric_label.setStyleSheet("padding: 10px; font-size: 13px; color: #e2e8f0;")
        self.tabs.addTab(self.metric_label, "训练指标")

        self.weight_text = QTextEdit()
        self.weight_text.setReadOnly(True)
        self.weight_text.setStyleSheet("background: #1e293b; color: #e2e8f0; border: none;")
        self.tabs.addTab(self.weight_text, "权重数值")

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background: #1e293b; color: #e2e8f0; border: none;")
        self.tabs.addTab(self.log_text, "运行日志")

        layout.addWidget(self.tabs, stretch=1)

        self.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #334155;
                background: #0f172a;
            }
            QTabBar::tab {
                background: #1e293b;
                color: #94a3b8;
                padding: 6px 14px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #334155;
                color: #f8fafc;
            }
        """)

    def update_metrics(self, epoch: int, loss: float, accuracy: float = None):
        text = f"Epoch: {epoch}\nLoss: {loss:.6f}"
        if accuracy is not None:
            text += f"\nAccuracy: {accuracy:.4f}"
        self.metric_label.setText(text)

    def update_loss_curve(self, loss_history: list):
        if self.figure is None or not loss_history:
            return
        epochs = list(range(1, len(loss_history) + 1))
        self.line_loss.set_data(epochs, loss_history)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

    def update_weights(self, weights: list):
        lines = []
        for i, w in enumerate(weights):
            lines.append(f"Layer {i+1} shape: {w.shape}")
            lines.append(np.array2string(w, precision=4, suppress_small=True))
            lines.append("")
        self.weight_text.setPlainText("\n".join(lines))

    def append_log(self, message: str):
        self.log_text.append(message)

    def clear_all(self):
        self.metric_label.setText("Epoch: 0\nLoss: --\nAccuracy: --")
        self.weight_text.clear()
        self.log_text.clear()
        if self.figure:
            self.line_loss.set_data([], [])
            self.ax.relim()
            self.ax.autoscale_view()
            self.canvas.draw()
