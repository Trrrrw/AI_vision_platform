from PySide6.QtWidgets import QGraphicsPathItem
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPen, QColor, QPainterPath


class ConnectionItem(QGraphicsPathItem):
    """组件之间的连线"""

    def __init__(self, start_port, end_port=None):
        super().__init__()
        self.start_port = start_port
        self.end_port = end_port
        self.setPen(QPen(QColor("#94a3b8"), 2, Qt.SolidLine))
        self.setZValue(-1)
        self.update_path()

    def set_end_pos(self, pos: QPointF):
        """拖拽过程中临时更新终点"""
        self.temp_end = pos
        self.update_path()

    def finalize(self, end_port):
        self.end_port = end_port
        self.setPen(QPen(QColor("#38bdf8"), 2, Qt.SolidLine))
        self.update_path()

    def update_path(self):
        if self.start_port:
            p1 = self.start_port.scene_center()
        else:
            return

        if self.end_port:
            p2 = self.end_port.scene_center()
        else:
            p2 = getattr(self, "temp_end", p1)

        path = QPainterPath()
        path.moveTo(p1)
        dx = abs(p2.x() - p1.x()) * 0.5
        cp1 = QPointF(p1.x() + dx, p1.y())
        cp2 = QPointF(p2.x() - dx, p2.y())
        path.cubicTo(cp1, cp2, p2)
        self.setPath(path)
