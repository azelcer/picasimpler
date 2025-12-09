"""

"""
import numpy as _np
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
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas # or backend_qt6agg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar # or backend_qt6agg
import logging as _lgn
from simpler_tools import SimplerAnalysisParameters, SIMPLERData


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
        self._table.setModel(SIMPLERTableModel(data))


class DataPlotWidget(QFrame):

    # apply_signal = pyqtSignal(SimplerAnalysisParameters)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_GUI()

    def _init_GUI(self):
        layout = QHBoxLayout()
        plt_lyt = QVBoxLayout()
        # self._plot = _pg.PlotWidget()
        # self._plot.setAspectLocked(1)
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)  # Add a subplot to the figure
        self.ax.axis("scaled")
        # self.sp = self.ax.scatter([], [])
        self._plot = FigureCanvas(self.fig)
        toolbar = NavigationToolbar(self._plot, self)
        plt_lyt.addWidget(toolbar)
        plt_lyt.addWidget(self._plot)
        layout.addLayout(plt_lyt, stretch=3)
        # layout.addWidget(self._plot)

        chk_layout = QVBoxLayout()
        self._unfiltered_chk = QCheckBox("Not included")
        self._filtered_chk = QCheckBox("Included")
        self._SIMPLER_equiv_chk = QCheckBox("SIMPLER equivalent")
        chk_layout.addWidget(self._unfiltered_chk)
        chk_layout.addWidget(self._filtered_chk)
        chk_layout.addWidget(self._SIMPLER_equiv_chk)
        layout.addLayout(chk_layout)
        self.setLayout(layout)

    def set_data(self, new_data: SIMPLERData):
        # borrar todo
        # actualizar
        # FIXME: cambiar este acceso feo, es sólo para arrancar a dibujar
        data = new_data.data
        x = data["x"]
        y = data["y"]
        # Meter transformacion a µm
        # self._scatter_plot = self._plot.plot(x, y, pen=None, symbolpen=None, symbol="s", symbolSize=.2, pxMode=False) # default True == son pixeles
        # p.setDownsampling(ds=None, auto=True, method="subsample")
        # p.setClipToView(True)
        # p.disableAutoRange()
        self.ax.clear()
        self.ax.scatter(x, y)

    def update_data(self, new_data: SIMPLERData):
        # borrar todo
        # actualizar
        # FIXME: cambiar este acceso feo, es sólo para arrancar a dibujar
        data = new_data.data
        x = data["x"][data["valid"]]
        y = data["y"][data["valid"]]
        # Meter transformacion a µm
        # self._scatter_plot.setData(x, y)
        self.ax.clear()
        self.ax.scatter(x, y)
        # self._scatter_plot.draw()


class Frontend(QMainWindow):
    """Coso.

    Implemented as a QFrame so it can be easily integrated within a larger app.
    """

    _modified = False
    _data = None

    def __init__(self, *args, **kwargs):
        """Init Frontend."""
        super().__init__(*args, **kwargs)
        self._setup_gui()
        self._setup_menus()
        self.setWindowTitle(_APP_NAME)
        # self.setWindowIcon()
        self._status_bar: QStatusBar = self.statusBar()
        self._status_bar.showMessage('Ready')  # Not for frames!
        # print(self._SIMPLER_widget.get_analysis_parameters())

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
        self._SIMPLER_widget = SimplerWidget(self)
        self._plot_widget = DataPlotWidget()
        self._table_widget = DataTableWidget(self)
        lower_layout = QHBoxLayout()
        lower_layout.addWidget(self._plot_widget)
        lower_layout.addWidget(self._table_widget)
        central_layout.addWidget(self._SIMPLER_widget)
        central_layout.addLayout(lower_layout)
        self.setCentralWidget(cw)
        return

    def notify(self, msg: str):
        """Convenience function."""
        self._status_bar.showMessage(str)

    def file_save(self):
        """Checks and opens a file."""
        if not self._modified:
            QMessageBox.information(
                self, 'Message', "Noting to save",
                QMessageBox.Ok, QMessageBox.Ok,
                )
            return
        print("not implemented")

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
            self._do_file_open(fname)
            self.setWindowTitle(make_window_title(fname))
            self._status_bar.showMessage(f"Opened {_pathlib.Path(fname).stem}")

    def _ask_file_open(self):
        """Ask a filename to open."""
        fname = QFileDialog.getOpenFileName(
            self, 'Open file', filter="Picasso HDF5 (*.hdf5)",  # TODO: remember dir
            )
        return fname[0]

    def _do_file_open(self, fname: str | _pathlib.Path):
        """Load a file."""
        self._data = SIMPLERData(fname)
        self._plot_widget.set_data(self._data)
        self._table_widget.set_data(self._data)

    @pyqtSlot(_QtGui.QCloseEvent)
    def closeEvent(self, event):
        """Shut down."""
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
