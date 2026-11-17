"""Ivory glass desktop widgets, painted locally without bitmap assets."""
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QRadialGradient
from PySide6.QtWidgets import QFrame, QGraphicsDropShadowEffect, QLabel, QWidget


STYLESHEET = """
QWidget { color:#2c4039; font-size:13px; }
QMainWindow, QWidget#background { background:#f8f6f0; }
QFrame#glass { background:rgba(255,255,255,140); border:1px solid rgba(255,255,255,230); border-radius:24px; }
QLabel { background:transparent; border:0; }
QLabel[role="eyebrow"] { color:#688379; font-size:11px; font-weight:600; letter-spacing:2px; }
QLabel[role="title"] { font-size:27px; font-weight:600; color:#263c33; }
QLabel[role="heading"] { font-size:17px; font-weight:600; }
QLabel[role="muted"] { color:#7d8b83; font-size:12px; }
QLabel[role="stat"] { color:#3e7765; font-size:22px; font-weight:600; }
QPushButton { background:rgba(255,255,255,160); border:1px solid rgba(255,255,255,240); border-radius:16px; padding:10px 16px; color:#405a50; }
QPushButton:hover { background:#eef6f0; border-color:#c3dcd1; }
QPushButton:pressed { background:#dcece3; }
QPushButton:disabled { color:#a8b3ac; background:rgba(245,246,241,120); border-color:#f7f8f4; }
QPushButton[role="primary"] { color:#fff; background:#79a99a; border:1px solid #91beae; font-weight:600; }
QPushButton[role="primary"]:hover { background:#669d8b; }
QPushButton[role="primary"]:disabled { background:#c3d5cd; border-color:#c3d5cd; color:#f8fbf8; }
QPushButton[role="danger"] { color:#a47768; }
QPushButton[role="danger"]:hover { background:#f7e9e1; border-color:#e7cfc2; }
QPushButton[role="nav"] { text-align:left; padding:14px 18px; border-radius:18px; background:transparent; border-color:transparent; }
QPushButton[role="nav"]:checked { background:rgba(215,237,225,190); border-color:#fafffb; color:#365e4f; font-weight:600; }
QPushButton[role="nav"]:hover { background:rgba(233,243,235,170); }
QLineEdit, QSpinBox, QComboBox, QTextEdit { background:rgba(255,255,255,150); border:1px solid #e2e8df; border-radius:13px; padding:9px 12px; selection-background-color:#c9e2d6; selection-color:#274238; }
QLineEdit:focus, QSpinBox:focus, QComboBox:focus, QTextEdit:focus { border-color:#91b9a8; background:#fffefa; }
QComboBox::drop-down { border:0; width:25px; }
QSpinBox::up-button, QSpinBox::down-button { width:18px; border:0; }
QCheckBox { spacing:8px; color:#526a5f; padding:5px 0; }
QCheckBox::indicator { width:17px; height:17px; border:1px solid #bfd3c6; border-radius:6px; background:#fdfefa; }
QCheckBox::indicator:checked { background:#85b5a2; border:4px solid #d9eade; }
QListWidget { background:transparent; border:0; outline:0; }
QListWidget::item { padding:12px 8px; border-radius:12px; margin:3px 0; color:#718378; }
QListWidget::item:hover { background:rgba(231,241,231,140); }
QListWidget::item:selected { background:rgba(220,237,226,190); color:#365e4f; }
QTableWidget { background:transparent; border:0; gridline-color:#e9ede4; selection-background-color:#dfede3; selection-color:#2f5142; outline:0; }
QTableWidget::item { padding:7px 8px; border-bottom:1px solid #edf0e9; }
QHeaderView::section { background:rgba(238,243,234,140); color:#7c8d80; border:0; border-bottom:1px solid #e5ebe1; padding:10px 6px; font-size:11px; }
QTableCornerButton::section { background:transparent; border:0; }
QScrollBar:vertical { background:transparent; width:7px; margin:4px 0; }
QScrollBar::handle:vertical { background:#ccd9ce; border-radius:3px; min-height:28px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:transparent; }
QToolTip { color:#365547; background:#fffdf7; border:1px solid #d7e4d9; padding:5px; }
"""


class GlassBackground(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('background')

    def paintEvent(self, event):
        painter=QPainter(self)
        painter.fillRect(self.rect(),QColor('#f8f6f0'))
        for x,y,radius,color in [(.35,.25,.55,QColor(174,220,202,95)),(.94,.81,.48,QColor(236,193,184,72)),(.12,.95,.34,QColor(202,221,208,55))]:
            gradient=QRadialGradient(QPointF(self.width()*x,self.height()*y),self.width()*radius)
            gradient.setColorAt(0,color)
            gradient.setColorAt(1,QColor(color.red(),color.green(),color.blue(),0))
            painter.fillRect(self.rect(),gradient)


class GlassCard(QFrame):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setObjectName('glass')
        shadow=QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)
        shadow.setOffset(0,10)
        shadow.setColor(QColor(58,91,68,17))
        self.setGraphicsEffect(shadow)


def label(text,role=None):
    widget=QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    if role:widget.setProperty('role',role)
    return widget


def primary(button):
    button.setProperty('role','primary')
    return button


def line_icon(kind):
    pixels=QPixmap(40,40)
    pixels.fill(Qt.GlobalColor.transparent)
    pixels.setDevicePixelRatio(2)
    painter=QPainter(pixels)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor('#597e6d'),1.3))
    if kind=='capture':
        painter.drawRoundedRect(2,4,16,12,3,3)
        painter.drawEllipse(7,7,6,6)
    else:
        painter.drawRoundedRect(4,2,12,16,2,2)
        for y in [7,10,13]:painter.drawLine(7,y,13,y)
    painter.end()
    return QIcon(pixels)
