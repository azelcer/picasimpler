"""

"""
from PyQt5.QtCore import (
    Qt,
    QAbstractTableModel,
)
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QFrame,
    QTableView,
)

import logging as _lgn

from picasimpler.dev.simpler_tools import FluoEvent


_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


class _EventTableModel(QAbstractTableModel):
    def __init__(self, data: list[FluoEvent]):
        super().__init__()
        self._data = data
        self._columns = ["Length", "Init frame", "End frame", "Localizations list"]

    def data(self, index, role):
        if role == Qt.DisplayRole:
            row = self._data[index.row()]
            match index.column():
                case 0:
                    return str(row.length)
                case 1:
                    return str(row._initial_frame)
                case 2:
                    return str(row._final_frame)
                case 3:
                    return ", ".join([str(_) for _ in row._localization_list])

    def rowCount(self, index):
        return len(self._data)

    def columnCount(self, index):
        return len(self._columns)

    def headerData(self, section: int, orientation: Qt.Orientation, role: Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Orientation.Vertical:
                return str(section)
            return str(self._columns[section])


class EventsTableWidget(QFrame):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_GUI()

    def _init_GUI(self):
        layout = QVBoxLayout(self)
        self.setLayout(layout)
        self._table = QTableView(self)
        layout.addWidget(self._table)

    def set_data(self, data: list[FluoEvent]):
        self._table.setModel(_EventTableModel(data))

    def reset(self):
        self._table.setModel(None)
