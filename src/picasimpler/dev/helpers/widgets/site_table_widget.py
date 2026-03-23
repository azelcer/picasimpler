"""

"""

from PyQt5.QtCore import (
    Qt as _Qt,
    QAbstractTableModel as _QAbstractTableModel,
    QItemSelection as _QItemSelection
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


class _SiteTableModel(_QAbstractTableModel):
    def __init__(self, data: list[list[FluoEvent]]):
        super().__init__()
        self._data = data
        self._columns = ["Events list", "Localizations list"]

    def data(self, index, role):
        if role == _Qt.DisplayRole:
            row = self._data[index.row()]
            match index.column():
                case 0:
                    return ", ".join([str(evt.idx) for evt in row])
                case 1:
                    return ", ".join([str(_) for evt in row for _ in evt._localization_list])

    def rowCount(self, index):
        return len(self._data)

    def columnCount(self, index):
        return len(self._columns)

    def headerData(self, section: int, orientation: _Qt.Orientation, role: _Qt.DisplayRole):
        if role == _Qt.DisplayRole:
            if orientation == _Qt.Orientation.Vertical:
                return str(section)
            return str(self._columns[section])


class SitesTableWidget(QFrame):

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self._parent = parent
        self._init_GUI()

    def _init_GUI(self):
        layout = QVBoxLayout(self)
        self.setLayout(layout)
        self._table = QTableView(self)
        self._table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        layout.addWidget(self._table)

    # pyqtslot(_QItemSelection, _QItemSelection)
    def _sel_changed(self, selected: _QItemSelection, deselected: _QItemSelection):
        # Sólo funciona porque es SelectRows
        # es esto o un set
        sel = [index.row() for index in selected.indexes() if index.column() == 0]
        desel = [index.row() for index in deselected.indexes() if index.column() == 0]
        self._parent._sites_selection_changed(sel, desel)

    def get_selected_rows(self):
        # es esto o un set
        selection = [index.row() for index in self._table.selectedIndexes() if index.column() == 0]
        return selection

    def toggle_selection(self, idx: int):
        selection_model = self._table.selectionModel()
        selection_model.blockSignals(True)
        index = self._table.model().index(idx, 0)

        selection_model.select(index, selection_model.Toggle | selection_model.Rows)
        selection_model.blockSignals(False)
        self._table.scrollTo(index)
        self._table.model().layoutChanged.emit()
        if selection_model.isRowSelected(idx):
            return True
        return False

        # was_selected = selection_model.isRowSelected(idx)
        # mode = selection_model.Rows | (
        #     selection_model.Deselect if was_selected else selection_model.Select)
        # selection_model.select(index, mode)
        # selection_model.blockSignals(False)
        # self._table.scrollTo(index)
        # self._table.model().layoutChanged.emit()
        # return not was_selected

    def set_data(self, data):
        self._table.setModel(_SiteTableModel(data))
        self._table.selectionModel().selectionChanged.connect(self._sel_changed)

    def reset(self):
        self._table.setModel(None)
