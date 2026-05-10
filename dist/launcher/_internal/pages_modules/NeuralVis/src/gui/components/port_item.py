from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem
from PySide6.QtCore import Qt, QRectF, QPointF
from PySide6.QtGui import QPen, QBrush, QColor


class PortItem(QGraphicsEllipseItem):
    """端口项：可拖拽连线"""

    RADIUS = 8
    DIAMETER = RADIUS * 2

    def __init__(self, name: str, is_input: bool, parent_node):
        super().__init__(-self.RADIUS, -self.RADIUS, self.DIAMETER, self.DIAMETER, parent_node)
        self.name = name
        self.is_input = is_input
        self.parent_node = parent_node
        self.setBrush(QBrush(QColor("#38bdf8")))
        self.setPen(QPen(QColor("#0f172a"), 2))
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

    def hoverEnterEvent(self, event):
        self.setBrush(QBrush(QColor("#f472b6")))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setBrush(QBrush(QColor("#38bdf8")))
        super().hoverLeaveEvent(event)

    def scene_center(self) -> QPointF:
        return self.mapToScene(0, 0)
