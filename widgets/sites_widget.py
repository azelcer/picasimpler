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


class SitesGroupingWidget(QFrame):

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent=parent, *args, **kwargs)
        self._parent = parent
        self._init_GUI()

    def freeze(self):
        self.setEnabled(False)

    def thaw(self):
        self.setEnabled(True)

    def _init_GUI(self):
        # TODO: add mehtod selection dropbox
        layout = QVBoxLayout()
        self._dist_sb = _pqth.create_labeled_float("Max dist / nm", layout, 80, 1, 1, maximum=100)
        self._group_button = QPushButton("Group events into sites", self)
        layout.addWidget(self._group_button)
        layout.addStretch(1)
        self.setLayout(layout)
        self._group_button.pressed.connect(self._parent.group_sites)

    def get_distance(self) -> float:
        return self._dist_sb.value()
