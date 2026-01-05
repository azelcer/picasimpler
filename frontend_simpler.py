"""

"""
import numpy as _np
from scipy.spatial import ConvexHull
import pathlib as _pathlib
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QAbstractTableModel
from PyQt5.QtWidgets import (
    # QGroupBox,
    QMainWindow,
    QAction,
    QLabel,
    QPushButton,
    QCheckBox,
    QHBoxLayout,
    QVBoxLayout,
    # QFormLayout,
    # QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QMessageBox,
    QFileDialog,
    QStatusBar,
    QFrame,
    QBoxLayout,
    QWidget,
    QTableView,
)
from PyQt5 import QtGui as _QtGui
# import pyqtgraph as _pg
from matplotlib.figure import Figure
from matplotlib.collections import EllipseCollection, PatchCollection
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas # or backend_qt6agg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar # or backend_qt6agg
import logging as _lgn
from simpler_tools import SimplerAnalysisParameters, SIMPLERData, FluoEvent

from threading import Thread, Event


_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


_APP_NAME = "PicaSIMPLER"


def make_window_title(filename: str | _pathlib.Path | None) -> str:
    if not filename:
        return _APP_NAME
    filename = _pathlib.Path(filename)
    return f"{_APP_NAME} - {filename.stem}"


# pyqt helpers
def create_labeled_float(name: str, external_layout: QBoxLayout,
                         value: float, decimals: int, step: float,
                         minimum: float = 0., maximum: float = None
                         ) -> QDoubleSpinBox:
    """Creates a labeled float spinbox."""
    hlayout = QHBoxLayout()
    sb = QDoubleSpinBox()
    sb.setDecimals(decimals)
    sb.setMinimum(minimum)
    if maximum is not None:
        sb.setMaximum(maximum)
    sb.setSingleStep(step)
    sb.setValue(value)
    hlayout.addWidget(QLabel(name))
    hlayout.addWidget(sb)
    external_layout.addLayout(hlayout)
    return sb


def create_labeled_int(name: str, external_layout: QBoxLayout,
                       value: int, step: int = 1,
                       minimum: int | None = None, maximum: int | None = None,
                       ) -> QDoubleSpinBox:
    """Creates a labeled float spinbox."""
    hlayout = QHBoxLayout()
    sb = QSpinBox()
    if minimum is not None:
        sb.setMinimum(minimum)
    if maximum is None:
        maximum = (1 << 31) - 1  # signed 32 bit
    sb.setMaximum(maximum)

    sb.setValue(value)
    hlayout.addWidget(QLabel(name))
    hlayout.addWidget(sb)
    external_layout.addLayout(hlayout)
    return sb


class background_runner:

    # _task_finished_evt = Event()

    def __init__(self):
        self._running = False
        self._thread = None
        self._callback: callable = None

    def submit(self, callback: callable, function: callable, args: list = [], kwargs: dict = {}):
        if self._running or self._thread:
            _lgr.error("Background task already running")
            return False
        self._target = function
        self._callback = callback
        self._thread = Thread(target=self._do_run, args=args, kwargs=kwargs)
        self._running = True
        self._thread.start()

    def _do_run(self, *args, **kwargs):
        try:
            self._rv = self._target(*args, **kwargs)
        except Exception as e:
            print("exception", e, type(e))
            self._rv = None
        self._running = False
        self._callback(self._rv)

    def cleanup(self):
        if self._running:
            _lgr.error("Background task still running")
            return False
        if not self._thread:
            _lgr.error("No background task running")
            return True
        self._thread.join()
        self._thread = None
        return True


class SIMPLERTableModel(QAbstractTableModel):
    def __init__(self, data: SIMPLERData):
        super().__init__()
        self._data = data
        self._columns = data.get_column_names()

    def data(self, index, role):
        if role == Qt.DisplayRole:
            # suponemos siempre numpy
            val = self._data.data[index.row()][index.column()]
            return "-" if _np.isnan(val) else str(val)
        elif role == Qt.BackgroundRole:
            if not self._data.data[index.row()]["valid"]:
                return _QtGui.QBrush(_QtGui.QColor(0xc0c0c0))

    def rowCount(self, index):
        return self._data.data.shape[0]

    def columnCount(self, index):
        return len(self._columns)

    def headerData(self, section: int, orientation: Qt.Orientation, role: Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Orientation.Vertical:
                return str(section)
            return str(self._columns[section])


class FluoEventTableModel(QAbstractTableModel):
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
        # elif role == Qt.BackgroundRole:
        #     if not self._data.data[index.row()]["valid"]:
        #         return _QtGui.QBrush(_QtGui.QColor(0xc0c0c0))

    def rowCount(self, index):
        return len(self._data)

    def columnCount(self, index):
        return len(self._columns)

    def headerData(self, section: int, orientation: Qt.Orientation, role: Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Orientation.Vertical:
                return str(section)
            return str(self._columns[section])

# table.cellClicked.connect(self.handle_cell_click)
# def handle_cell_click(self, row, column):
#     print(f'Cell clicked: Row {row}, Column {column}') #
# table.selectionModel().selectionChanged.connect(self.handle_selection_changed)
# def handle_selection_changed(self, selected, deselected):
#     # Code to handle the change in selection
#     pass #
# self.tv.setSelectionBehavior(QtWidgets.QTableView.SelectRows)
# QtGui.QAbstractItemView.SelectRows if you haven't loaded QtGui


class SimplerWidget(QFrame):

    apply_signal = pyqtSignal(SimplerAnalysisParameters)

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
        self._dist_sb = create_labeled_float("Max dist / nm", layout, 10, 1, 1)
        self._alpha_sb = create_labeled_float("\u03B1<sub>F</sub>", layout, 10, 1, 1)
        self._dF_sb = create_labeled_float("d<sub>F</sub> / nm", layout, 10, 1, 1)
        self._N0_sb = create_labeled_int("N<sub>0</sub>", layout, 10000,)
        self._filter_button = QPushButton("Filter", self)
        self._filter_button.pressed.connect(self._parent.filter_data)
        self._apply_button = QPushButton("Apply", self)
        layout.addWidget(self._filter_button)
        layout.addWidget(self._apply_button)
        self.setLayout(layout)

    def get_analysis_parameters(self) -> SimplerAnalysisParameters:
        return SimplerAnalysisParameters(
            self._dist_sb.value(),
            self._alpha_sb.value(),
            self._dF_sb.value(),
            self._N0_sb.value(),
        )


class SimplerCalibrationWidget(QFrame):

    apply_signal = pyqtSignal(SimplerAnalysisParameters)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_GUI()

    def freeze(self):
        self.setEnabled(False)

    def thaw(self):
        self.setEnabled(True)

    def _init_GUI(self):
        layout = QVBoxLayout()
        self._dist_sb = create_labeled_float("Max dist / nm", layout, 10, 1, 1)
        # self._alpha_sb = create_labeled_float("\u03B1<sub>F</sub>", layout, 10, 1, 1)
        # self._dF_sb = create_labeled_float("d<sub>F</sub> / nm", layout, 10, 1, 1)
        # self._N0_sb = create_labeled_int("N<sub>0</sub>", layout, 10000,)
        self._clusterize_button = QPushButton("Clusterize", self)
        layout.addWidget(self._clusterize_button)
        self.setLayout(layout)

    def get_analysis_parameters(self) -> SimplerAnalysisParameters:
        return SimplerAnalysisParameters(
            self._dist_sb.value(),
            self._alpha_sb.value(),
            self._dF_sb.value(),
            self._N0_sb.value(),
            )


class EventsGroupingWidget(QFrame):

    apply_signal = pyqtSignal()

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
        self._dist_sb = create_labeled_float("Max dist / nm", layout, 10, 1, 1)
        self._group_button = QPushButton("Group into events", self)
        layout.addWidget(self._group_button)
        self.setLayout(layout)
        self._group_button.pressed.connect(self._parent.group_events)

    def get_distance(self) -> float:
        return self._dist_sb.value()


class SitesGroupingWidget(QFrame):

    apply_signal = pyqtSignal()

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
        self._dist_sb = create_labeled_float("Max dist / nm", layout, 80, 1, 1, maximum=100)
        self._group_button = QPushButton("Group events into sites", self)
        layout.addWidget(self._group_button)
        self.setLayout(layout)
        self._group_button.pressed.connect(self._parent.group_sites)

    def get_distance(self) -> float:
        return self._dist_sb.value()


class DataTableWidget(QFrame):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_GUI()

    def _init_GUI(self):
        layout = QVBoxLayout(self)
        self.setLayout(layout)
        self._table = QTableView(self)
        layout.addWidget(QLabel("Localizations data"))
        layout.addWidget(self._table)

    def set_data(self, data):
        self._table.setModel(SIMPLERTableModel(data))


class EventsTableWidget(QFrame):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_GUI()

    def _init_GUI(self):
        layout = QVBoxLayout(self)
        self.setLayout(layout)
        self._table = QTableView(self)
        layout.addWidget(QLabel("Events data"))
        layout.addWidget(self._table)

    def set_data(self, data):
        self._table.setModel(FluoEventTableModel(data))


class DataPlotWidget(QFrame):

    # apply_signal = pyqtSignal(SimplerAnalysisParameters)
    _marker_size = 1.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_GUI()
        self._init_graphs()
        self._data = None
        self._events = None

    def _init_GUI(self):
        layout = QHBoxLayout()
        plt_lyt = QVBoxLayout()
        # self._plot = _pg.PlotWidget()
        # self._plot.setAspectLocked(1)
        self.fig = Figure()  # figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)  # Add a subplot to the figure
        self.ax.axis("equal")
        self._plot = FigureCanvas(self.fig)
        toolbar = NavigationToolbar(self._plot, self)
        plt_lyt.addWidget(toolbar)
        plt_lyt.addWidget(self._plot)
        layout.addLayout(plt_lyt, stretch=3)

        chk_layout = QVBoxLayout()
        self._ungrouped_chk = QCheckBox("Ungrouped")
        self._grouped_chk = QCheckBox("Grouped")
        self._events_chk = QCheckBox("Events")
        self._ungrouped_chk.setCheckState(1)
        self._grouped_chk.setCheckState(1)
        self._events_chk.setCheckState(1)
        self._ungrouped_chk.stateChanged.connect(self._graph_selection_changed)
        self._grouped_chk.stateChanged.connect(self._graph_selection_changed)
        self._events_chk.stateChanged.connect(self._graph_selection_changed)
        chk_layout.addWidget(self._ungrouped_chk)
        chk_layout.addWidget(self._grouped_chk)
        chk_layout.addWidget(self._events_chk)
        layout.addLayout(chk_layout)
        self.setLayout(layout)

    def _init_graphs(self):
        self.ax.clear()
        self._ungrouped_scatter: Line2D = self.ax.plot([], [], marker="o", ls="", ms=self._marker_size, c="blue")[0]
        self._grouped_scatter: Line2D = self.ax.plot([], [], marker="o", ls="", ms=self._marker_size, c="red")[0]
        self._events_scatter = self.ax.add_collection(EllipseCollection([], [], []))
        self._sites_scatter = self.ax.add_collection(PatchCollection([]))

    def set_data(self, new_data: SIMPLERData):
        """Cleans everything."""

        self._data = new_data
        # Meter transformacion a µm
        # self._scatter_plot = self._plot.plot(x, y, pen=None, symbolpen=None, symbol="s", symbolSize=.2, pxMode=False) # default True == son pixeles
        # p.setDownsampling(ds=None, auto=True, method="subsample")
        # p.setClipToView(True)
        # p.disableAutoRange()
        self._init_graphs()
        self._update_graphs()
        self._graph_selection_changed(1)
        self.ax.relim()
        self.ax.autoscale_view()
        self.ax.autoscale(enable=True, axis='both')
        self._plot.draw()

    def _update_graphs(self):
        data = self._data.get_ungrouped_locations()
        self._ungrouped_scatter.set_data(data["x"], data["y"])

        data = self._data.get_grouped_locations()
        self._grouped_scatter.set_data(data["x"], data["y"])

        self._events_scatter.remove()
        sigmas = self._data.get_events_sizes()
        self._events_scatter = self.ax.add_collection(EllipseCollection(
                widths=sigmas, heights=sigmas, angles=0, units='xy',
                # facecolors=plt.cm.hsv(duraciones / duraciones.max()),
                offsets=self._data.get_events_locations(), transOffset=self.ax.transData,
                alpha=0.4,
                )
            )
        self._sites_scatter.remove()
        sites = self._data.get_sites()
        patches = []
        for site in sites:
            or_points = _np.array([_.center for _ in site])
            if len(site) < 3:
                vertex = or_points
            else:
                ch = ConvexHull(or_points)
                vertex = ch.points[ch.vertices]
            patches.append(Polygon(vertex, closed=True, color="r"))
        p = PatchCollection(patches, alpha=0.3)
        # p.set_color("r")
        self._sites_scatter = self.ax.add_collection(p)

    def data_updated(self):
        self._update_graphs()
        # Collections are not updated but replaced: visibility is forgotten
        self._graph_selection_changed(1)

    # SLOTS
    @pyqtSlot(int)
    def _graph_selection_changed(self, checked: int):
        if self._ungrouped_chk.checkState():
            self._ungrouped_scatter.set_visible(True)
        else:
            self._ungrouped_scatter.set_visible(False)

        if self._grouped_chk.checkState():
            self._grouped_scatter.set_visible(True)
        else:
            self._grouped_scatter.set_visible(False)

        if self._events_scatter:
            if self._events_chk.checkState():
                self._events_scatter.set_visible(True)
            else:
                self._events_scatter.set_visible(False)
        self._plot.draw()
        # self._plot.draw_idle()


class Frontend(QMainWindow):
    """Coso.

    Implemented as a QFrame so it can be easily integrated within a larger app.
    """

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
        self._plot_widget = DataPlotWidget()
        self._localizations_table_widget = DataTableWidget(self)
        self._events_table_widget = EventsTableWidget(self)
        upper_layout = QHBoxLayout()
        upper_layout.addWidget(self._event_grouping_widget)
        upper_layout.addWidget(self._sites_grouping_widget)
        upper_layout.addWidget(self._SIMPLER_widget)
        lower_layout = QHBoxLayout()
        lower_layout.addWidget(self._plot_widget)
        lower_layout.addWidget(self._events_table_widget)
        lower_layout.addWidget(self._localizations_table_widget)

        central_layout.addLayout(upper_layout)
        central_layout.addLayout(lower_layout)
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

    def _data_grouped_cb(self, rv):
        self._data_grouped_signal.emit()

    @pyqtSlot()
    def _data_grouped_handler(self):
        self._runner.cleanup()
        self._thaw_all()
        self._events_table_widget.set_data(self._data.get_events())
        self._plot_widget.data_updated()
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
        self.setWindowTitle(make_window_title(self._fname))
        self.notify(f"Opened {_pathlib.Path(self._fname).stem}")

    def _sites_grouped_cb(self, rv):
        self._sites_grouped_signal.emit()

    @pyqtSlot()
    def _sites_grouped_handler(self):
        self._thaw_all()
        self._runner.cleanup()
        self._plot_widget.data_updated()
        self.notify("Events grouped into sites!")

    @pyqtSlot(_QtGui.QCloseEvent)
    def closeEvent(self, event):
        """Shut down."""
        if not self._runner.cleanup():
            QMessageBox.information(
                self, "Can't exit", "Background task still running",
                QMessageBox.Ok, QMessageBox.Ok,
                )
            event.ignore()
            return
        if not self._modified:
            event.accept()
            return
        reply = QMessageBox.question(self, 'Message',
                                     "Are you sure to quit?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No
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
