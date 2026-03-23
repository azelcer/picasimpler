"""

"""
from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QFrame,
)

import logging as _lgn
from helpers import pyqt_helpers as _pqth

_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


class EventsGroupingWidget(QFrame):

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent=parent, *args, **kwargs)
        self._parent = parent
        self._init_GUI()

    def freeze(self):
        self.setEnabled(False)

    def thaw(self):
        self.setEnabled(True)

    def _init_GUI(self):
        layout = QVBoxLayout()
        self._dist_sb = _pqth.create_labeled_float("Max dist / nm", layout, 10, 1, 1)
        self._group_button = QPushButton("Group into events", self)
        layout.addWidget(self._group_button)
        layout.addStrut(1)
        self._min_length_sb = _pqth.create_labeled_int("Min frames / nm", layout, 2, 2, None)
        self._filter_button = QPushButton("Filter", self)
        layout.addWidget(self._filter_button)
        self.setLayout(layout)
        layout.addStretch(0)
        self._group_button.pressed.connect(self._parent.group_events)
        self._filter_button.pressed.connect(self._parent.filter_events)

    def get_distance(self) -> float:
        return self._dist_sb.value()

    def get_length_limits(self) -> tuple[int, int]:
        return (self._min_length_sb.value(), None)
