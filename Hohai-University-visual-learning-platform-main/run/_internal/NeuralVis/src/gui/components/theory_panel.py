from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit


class TheoryPanel(QWidget):
    """底部原理讲解区"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.setMaximumHeight(140)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("原理讲解")
        title.setStyleSheet("font-size: 13px; font-weight: bold; color: #f8fafc;")
        layout.addWidget(title)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 8px;
                font-size: 13px;
                line-height: 1.5;
            }
        """)
        self.text_edit.setPlainText(
            "欢迎使用神经网络可视化学习工具！\n"
            "在右侧面板设置网络结构、激活函数、损失函数和学习率后，点击“构建网络”，"
            "然后使用顶部工具栏的“运行/暂停”或底部“单步执行”开始观察训练过程。"
        )
        layout.addWidget(self.text_edit)

    def set_text(self, text: str):
        self.text_edit.setPlainText(text)
