"""
Necesita:
    numpy
    pyqt5
    hdf5
    matplotlib
    pyyaml
    scipy
    scikit
"""
import numpy as _np
import pathlib as _pathlib
from PyQt5.QtCore import (
    pyqtSignal,
    pyqtSlot,
    Qt,
)
from PyQt5.QtWidgets import (
    QMainWindow,
    QAction,
    QHBoxLayout,
    QVBoxLayout,
    QMessageBox,
    QFileDialog,
    QStatusBar,
    QWidget,
    QDockWidget,
)
from PyQt5 import QtGui as _QtGui

import logging as _lgn

from simpler_tools import SIMPLERData


from widgets.plot_widget import DataPlotWidget
from widgets.simpler_widget import SimplerWidget
from widgets.events_widget import EventsGroupingWidget
from widgets.sites_widget import SitesGroupingWidget
from widgets.data_table_widget import DataTableWidget
from widgets.events_table_widget import EventsTableWidget
from widgets.site_table_widget import SitesTableWidget


from helpers.running_helpers import background_runner


_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


_APP_NAME = "PicaSIMPLER"


def make_window_title(filename: str | _pathlib.Path | None) -> str:
    if not filename:
        return _APP_NAME
    filename = _pathlib.Path(filename)
    return f"{_APP_NAME} - {filename.stem}"


def wrap_in_dock(parent: QWidget, title: str, content: QWidget) -> QDockWidget:
    dock_widget = QDockWidget(title, parent)
    dock_widget.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
    dock_widget.setWidget(content)
    return dock_widget


class Frontend(QMainWindow):
    """Coso."""

    _modified = False
    _data = None
    _freezable_widgets: list[QWidget] = []
    _data_grouped_signal = pyqtSignal()
    _data_load_signal = pyqtSignal()
    _sites_grouped_signal = pyqtSignal()

    def __init__(self, *args, **kwargs):
        """Init Frontend."""
        super().__init__(*args, **kwargs)
        self._setup_gui()
        self._setup_menus()
        self.setWindowTitle(_APP_NAME)
        # self.setWindowIcon()
        self._status_bar: QStatusBar = self.statusBar()
        self.notify('Ready')  # Not for frames!
        # print(self._SIMPLER_widget.get_analysis_parameters())
        self._runner = background_runner()
        self._data_load_signal.connect(self._data_loaded_handler)
        self._data_grouped_signal.connect(self._data_grouped_handler)
        self._sites_grouped_signal.connect(self._sites_grouped_handler)

    def _setup_menus(self):
        """Setup menues."""
        open_act = QAction('&Open', self)
        open_act.setShortcut('Ctrl+O')
        open_act.setStatusTip('Open file')
        open_act.triggered.connect(self.file_open)
        save_act = QAction('&Save', self)
        save_act.setShortcut('Ctrl+S')
        save_act.setStatusTip('Save file')
        save_act.triggered.connect(self.file_save)
        self._menu_bar = mb = self.menuBar()
        fileMenu = mb.addMenu('&File')
        fileMenu.addAction(open_act)
        fileMenu.addAction(save_act)

    def _setup_gui(self):
        """Create and lay out all GUI objects."""
        # GUI layout
        cw = QWidget()
        central_layout = QVBoxLayout()
        cw.setLayout(central_layout)
        self._event_grouping_widget = EventsGroupingWidget(self)
        self._freezable_widgets.append(self._event_grouping_widget)
        self._sites_grouping_widget = SitesGroupingWidget(self)
        self._freezable_widgets.append(self._sites_grouping_widget)
        self._SIMPLER_widget = SimplerWidget(self)
        self._freezable_widgets.append(self._SIMPLER_widget)
        self._plot_widget = DataPlotWidget(self)
        self._localizations_table_widget = DataTableWidget(self)
        self._events_table_widget = EventsTableWidget(self)
        self._sites_table_widget = SitesTableWidget(self)
        upper_layout = QHBoxLayout()
        upper_layout.addWidget(self._event_grouping_widget)
        upper_layout.addWidget(self._sites_grouping_widget)
        upper_layout.addWidget(self._SIMPLER_widget)
        lower_layout = QHBoxLayout()
        lower_layout.addWidget(self._plot_widget)
        self._sites_dock = wrap_in_dock(self, "Sites table", self._sites_table_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._sites_dock)
        self._evt_dock = wrap_in_dock(self, "Events table", self._events_table_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._evt_dock)
        self._loc_dock = wrap_in_dock(self, "Localizations table", self._localizations_table_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._loc_dock)
        self.tabifyDockWidget(self._sites_dock, self._evt_dock)
        self.tabifyDockWidget(self._evt_dock, self._loc_dock)

        central_layout.addLayout(upper_layout, stretch=0)
        central_layout.addLayout(lower_layout, stretch=1)
        self.setCentralWidget(cw)
        return

    def _freeze_all(self):
        for w in self._freezable_widgets:
            w.freeze()
        self._menu_bar.setEnabled(False)

    def _thaw_all(self):
        self._menu_bar.setEnabled(True)
        for w in self._freezable_widgets:
            w.thaw()

    def notify(self, msg: str):
        """Convenience function."""
        self._status_bar.showMessage(msg)

    def file_save(self):
        """Checks and opens a file."""
        if not self._modified:
            QMessageBox.information(
                self, 'Message', "Nothing to save",
                QMessageBox.Ok, QMessageBox.Ok,
            )
            return
        print("not implemented")

    def group_events(self):
        if self._data is None:
            _lgr.info("No data to group")
            return
        self._runner.submit(self._data_grouped_cb, self._data.group_events, args=(self._event_grouping_widget.get_distance(),))
        self.notify("Grouping data...")
        self._freeze_all()

    def filter_events(self):
        if self._data is None:
            _lgr.info("No data to group")
            return
        self._data.filter_events(*self._event_grouping_widget.get_length_limits())
        self._plot_widget.data_updated()
        self._evt_dock.raise_()
        self.notify("Events filtered by length")

    def group_sites(self):
        if self._data is None:
            _lgr.info("No data to group")
            return
        if not self._data._runs:  # TODO: do not deep link
            _lgr.info("Data not grouped into events")
            return
        self._runner.submit(self._sites_grouped_cb, self._data.group_sites, args=(self._sites_grouping_widget.get_distance(), "DBSCAN"))
        self.notify("Grouping sites...")
        self._freeze_all()

    def site_toggle_selection(self, idx: int):
        """Updates table."""
        return self._sites_table_widget.toggle_selection(idx)

    def _data_grouped_cb(self, rv):
        self._data_grouped_signal.emit()

    @pyqtSlot()
    def _data_grouped_handler(self):
        self._runner.cleanup()
        self._thaw_all()
        self._events_table_widget.set_data(self._data.get_events())
        self._plot_widget.data_updated()
        self._evt_dock.raise_()
        self.notify("Data grouped")

    def filter_data(self):
        if self._data is None:
            _lgr.info("No data to filter")
            return
        self._data.filter_data(self._SIMPLER_widget.get_analysis_parameters())
        self._plot_widget.update_data(self._data)

    def file_open(self):
        """Checks and opens a file."""
        if self._modified:
            reply = QMessageBox.question(
                self, 'Message', "Are you sure to quit?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        fname = self._ask_file_open()
        if fname:
            self._fname = fname
            self._runner.submit(self._data_loaded_cb, self._do_file_open, args=(fname,))
            self._freeze_all()
            # self._do_file_open(fname)

    def _ask_file_open(self):
        """Ask a filename to open."""
        fname = QFileDialog.getOpenFileName(
            self, 'Open file', filter="Picasso HDF5 (*.hdf5)",  # TODO: remember dir
        )
        return fname[0]

    def _do_file_open(self, fname: str | _pathlib.Path):
        """Load a file."""
        self._data = SIMPLERData(fname)

    def _data_loaded_cb(self, rv):
        self._data_load_signal.emit()

    @pyqtSlot()
    def _data_loaded_handler(self):
        self._thaw_all()
        self._runner.cleanup()
        self._plot_widget.set_data(self._data)
        self._localizations_table_widget.set_data(self._data)
        self._events_table_widget.reset()
        self._sites_table_widget.reset()
        self.setWindowTitle(make_window_title(self._fname))
        self.notify(f"Opened {_pathlib.Path(self._fname).stem}")

    def _sites_grouped_cb(self, rv):
        self._sites_grouped_signal.emit()

    @pyqtSlot()
    def _sites_grouped_handler(self):
        self._thaw_all()
        self._runner.cleanup()
        self._plot_widget.data_updated()
        self._sites_table_widget.set_data(self._data.get_sites())
        self._sites_dock.raise_()
        self.notify("Events grouped into sites!")

    def _sites_selection_changed(self, new_sel_idx: list[int], new_desel_idx: list[int]):
        """Called from table widget.

        Updates zoom.
        """
        # Asegurarse de que todos los seleccionados tengan los colores y líneas
        self._plot_widget.change_selected_patches(new_sel_idx, True)
        self._plot_widget.change_selected_patches(new_desel_idx, False)
        indexes = self._sites_table_widget.get_selected_rows()
        if not indexes:  # empty list
            return
        shift_arr = _np.array((-1, 1, ))
        sites = self._data.get_sites()
        positions = _np.array([evt.center for i in indexes for evt in sites[i]])
        lim_x, lim_y = zip(positions.min(axis=0), positions.max(axis=0), )
        plus_x = abs(lim_x[0] - lim_x[1]) * .05
        plus_y = abs(lim_y[0] - lim_y[1]) * .05
        lim_x += shift_arr * plus_x
        lim_y += shift_arr * plus_y
        self._plot_widget.zoom_to(lim_x, lim_y)

    @pyqtSlot(_QtGui.QCloseEvent)
    def closeEvent(self, event):
        """Shut down."""
        if not self._runner.cleanup():
            QMessageBox.information(
                self,
                "Can't exit",
                "Background task still running",
                QMessageBox.Ok,
                QMessageBox.Ok,
            )
            event.ignore()
            return
        if not self._modified:
            event.accept()
            return
        reply = QMessageBox.question(
            self,
            "Changes have been made",
            "Are you sure to quit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
        # super().closeEvent(*args, **kwargs)


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    if not QApplication.instance():
        app = QApplication([])
    else:
        app = QApplication.instance()
    app.setStyle('Windows')
    gui = Frontend()
    gui.show()
    gui.activateWindow()
    gui.raise_()

    app.exec_()
    app.quit()
