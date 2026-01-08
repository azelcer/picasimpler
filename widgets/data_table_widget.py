"""

"""
import numpy as _np

from PyQt5.QtCore import (
    Qt as _Qt,
    QAbstractTableModel as _QAbstractTableModel,
)
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QFrame,
    QTableView,
)
from PyQt5 import QtGui as _QtGui

import logging as _lgn

from simpler_tools import SIMPLERData

_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


class _SIMPLERTableModel(_QAbstractTableModel):
    def __init__(self, data: SIMPLERData):
        super().__init__()
        self._data = data
        self._columns = data.get_column_names()

    def data(self, index, role):
        if role == _Qt.DisplayRole:
            # suponemos siempre numpy
            val = self._data.data[index.row()][index.column()]
            return "-" if _np.isnan(val) else str(val)
        # elif role == _Qt.BackgroundRole:
        #     if not self._data.data[index.row()]["valid"]:
        #         return _QtGui.QBrush(_QtGui.QColor(0xc0c0c0))

    def rowCount(self, index):
        return self._data.data.shape[0]

    def columnCount(self, index):
        return len(self._columns)

    def headerData(self, section: int, orientation: _Qt.Orientation, role: _Qt.DisplayRole):
        if role == _Qt.DisplayRole:
            if orientation == _Qt.Orientation.Vertical:
                return str(section)
            return str(self._columns[section])


class DataTableWidget(QFrame):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_GUI()

    def _init_GUI(self):
        layout = QVBoxLayout(self)
        self.setLayout(layout)
        self._table = QTableView(self)
        layout.addWidget(self._table)

    def set_data(self, data):
        self._table.setModel(_SIMPLERTableModel(data))
