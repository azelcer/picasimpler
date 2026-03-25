"""

"""
from collections.abc import Iterable
import numpy as _np
from scipy.spatial import ConvexHull
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QVBoxLayout,
    QFrame,
    QWidget,
)
from matplotlib.figure import Figure
# from matplotlib.collections import EllipseCollection, PatchCollection
# from matplotlib.patches import Polygon
import mpl_toolkits.mplot3d  # noqa
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
# from matplotlib.backend_bases import PickEvent
import logging as _lgn
# import itertools as _it

from simpler_tools import SIMPLERData


_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


class SIMPLERPlotWidget(QFrame):

    _marker_size = 1.

    def __init__(self, parent: QWidget, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if parent:
        #     self.setWindowFlag(Qt.WindowCloseButtonHint, False)
        self._init_GUI()
        self._init_graphs()
        self._data = None
        self._site_idx = None
        self._parent = parent

    def _init_GUI(self):
        layout = QHBoxLayout()
        plt_lyt = QVBoxLayout()
        self.fig = Figure()  # figsize=(5, 4), dpi=100)
        self.ax_i, self.ax_z = self.fig.subplots(1, 2, squeeze=True, subplot_kw={"projection": "3d"})  # Add a subplot to the figure
        self._plot = FigureCanvas(self.fig)
        toolbar = NavigationToolbar(self._plot, self)
        plt_lyt.addWidget(toolbar)
        plt_lyt.addWidget(self._plot)
        layout.addLayout(plt_lyt, stretch=3)
        self.setLayout(layout)
        self.resize(300, 300)
        self.show()

    def _init_graphs(self):
        self.ax_i.clear()
        self.ax_z.clear()
        self._z_scatter, = self.ax_z.plot([], [], [], marker="o", ls="", ms=self._marker_size)
        self._I_scatter, = self.ax_i.plot([], [], [], marker="o", ls="", ms=self._marker_size)

    def set_data(self, new_data: SIMPLERData, site_idx: int | None):
        """Cleans everything."""
        self._data = new_data
        self._site_idx = site_idx
        self._init_graphs()
        self._update_graphs()

        # self.ax.margins(.05)
        # self.ax_i.relim()
        self.ax_i.autoscale(enable=True, axis='both')
        self.ax_i.autoscale_view()
        print("autoescalado")
        self._plot.draw()

    def _update_graphs(self):
        if self._site_idx is None:
            return

        locations = self._data.get_site_locations(self._site_idx)
        data = self._data.data
        data_x, data_y, photons, z = zip(*[(data[_]["x"], data[_]["y"], data[_]["photons"], data["z"]) for _ in locations])
        data_x = _np.array(data_x)
        data_y = _np.array(data_y)
        photons = _np.array(photons)
        # mean_x = _np.average(data_x)
        # mean_y = _np.average(data_y)
        # self._I_scatter.set_data(data_x - mean_x, data_y - mean_y)
        self._I_scatter.set_data(data_x, data_y)
        self._I_scatter.set_3d_properties(photons)
        self.ax_i.set_xlim(min(data_x), max(data_x))
        self.ax_i.set_ylim(min(data_y), max(data_y))
        self.ax_i.set_zlim(min(photons), max(photons))

    def data_updated(self):
        self._update_graphs()
