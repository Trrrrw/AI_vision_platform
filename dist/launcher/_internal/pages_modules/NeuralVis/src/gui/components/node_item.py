from PySide6.QtWidgets import (
    QGraphicsItem, QGraphicsRectItem, QGraphicsTextItem, QGraphicsEllipseItem,
    QGraphicsProxyWidget, QSpinBox, QVBoxLayout, QWidget, QLabel
)
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPen, QBrush, QColor, QFont
from .port_item import PortItem


class NodeItem(QGraphicsItem):
    """左侧组件库中的可拖拽节点 / 画布上的网络组件"""

    def __init__(self, title: str, node_type: str, width: int = 140, height: int = 80):
        super().__init__()
        self.title = title
        self.node_type = node_type  # input, hidden, output, activation, loss, optimizer, control
        self.width = width
        self.height = height
        self.input_ports = []
        self.output_ports = []

        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

        # 背景
        self.bg = QGraphicsRectItem(0, 0, width, height, self)
        self.bg.setBrush(QBrush(QColor("#1e293b")))
        self.bg.setPen(QPen(QColor("#475569"), 2))

        # 标题
        self.text = QGraphicsTextItem(title, self)
        self.text.setDefaultTextColor(QColor("#f1f5f9"))
        font = QFont("Microsoft YaHei", 10, QFont.Bold)
        self.text.setFont(font)
        self.text.setPos((width - self.text.boundingRect().width()) / 2, 8)

        self._setup_ports()

    def _setup_ports(self):
        if self.node_type in ("input", "hidden", "output"):
            # 神经网络层：左侧输入端口，右侧输出端口
            inp = PortItem("in", True, self)
            inp.setPos(0, self.height / 2)
            self.input_ports.append(inp)

            out = PortItem("out", False, self)
            out.setPos(self.width, self.height / 2)
            self.output_ports.append(out)
        else:
            # 函数/控制组件
            inp = PortItem("in", True, self)
            inp.setPos(0, self.height / 2)
            self.input_ports.append(inp)

            out = PortItem("out", False, self)
            out.setPos(self.width, self.height / 2)
            self.output_ports.append(out)

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter, option, widget=None):
        pass

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            # 移动时通知所有连线更新
            for port in self.input_ports + self.output_ports:
                if hasattr(port, "connections"):
                    for conn in port.connections:
                        conn.update_path()
        return super().itemChange(change, value)
