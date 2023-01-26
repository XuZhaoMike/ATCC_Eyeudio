from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import pyqtSignal


class LabelMouse(QLabel):
    double_clicked = pyqtSignal()

    # Mouse double click event
    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit()

    def mouseMoveEvent(self):
        """
        Event triggered when the mouse crosses label2
        :return:
        """
        print('Event triggered when the mouse crosses label2')


class Label_click_Mouse(QLabel):
    clicked = pyqtSignal()

    # Mouse click event
    def mousePressEvent(self, event):
        self.clicked.emit()