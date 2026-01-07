"""

"""
from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QFrame,
    QWidget,
)

import logging as _lgn
from helpers import pyqt_helpers as _pqth
from simpler_tools import SimplerAnalysisParameters

_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


class SimplerWidget(QFrame):

    def __init__(self, parent: QWidget, *args, **kwargs):
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
        self._alpha_sb = _pqth.create_labeled_float("\u03B1<sub>F</sub>", layout, 10, 1, 1)
        self._dF_sb = _pqth.create_labeled_float("d<sub>F</sub> / nm", layout, 10, 1, 1)
        self._N0_sb = _pqth.create_labeled_int("N<sub>0</sub>", layout, 10000,)
        self._filter_button = QPushButton("Filter", self)
        self._filter_button.pressed.connect(self._parent.filter_data)
        layout.addWidget(self._filter_button)
        layout.addStretch(1)
        self.setLayout(layout)

    def get_analysis_parameters(self) -> SimplerAnalysisParameters:
        return SimplerAnalysisParameters(
            self._dist_sb.value(),
            self._alpha_sb.value(),
            self._dF_sb.value(),
            self._N0_sb.value(),
        )
