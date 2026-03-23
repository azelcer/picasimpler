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
from matplotlib.collections import EllipseCollection, PatchCollection
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.backend_bases import PickEvent
import logging as _lgn

from picasimpler.dev.simpler_tools import SIMPLERData


_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.DEBUG)


_APP_NAME = "PicaSIMPLER"

# Placeholers for customization
_UNSELECTED_LINEWITDH = 1
_SELECTED_LINEWITDH = 4
_UNSELECTED_LINECOLOR = (0, 128 / 255, 0, .3)
_SELECTED_LINECOLOR = (1., 0, 0, 1.)


class DataPlotWidget(QFrame):

    _marker_size = 1.

    def __init__(self, parent: QWidget, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self._init_GUI()
        self._init_graphs()
        self._data = None
        self._events = None
        self._parent = parent

    def _init_GUI(self):
        layout = QHBoxLayout()
        plt_lyt = QVBoxLayout()
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
        self._sites_chk = QCheckBox("Sites")
        self._ungrouped_chk.setCheckState(1)
        self._grouped_chk.setCheckState(1)
        self._events_chk.setCheckState(1)
        self._sites_chk.setCheckState(1)
        self._ungrouped_chk.stateChanged.connect(self._graph_selection_changed)
        self._grouped_chk.stateChanged.connect(self._graph_selection_changed)
        self._events_chk.stateChanged.connect(self._graph_selection_changed)
        self._sites_chk.stateChanged.connect(self._graph_selection_changed)
        chk_layout.addWidget(self._ungrouped_chk)
        chk_layout.addWidget(self._grouped_chk)
        chk_layout.addWidget(self._events_chk)
        chk_layout.addWidget(self._sites_chk)
        layout.addLayout(chk_layout)
        self.setLayout(layout)

        self._plot.mpl_connect('scroll_event', self._on_scroll)
        self._plot.mpl_connect('pick_event', self._on_pick)

    def _on_scroll(self, event):
        _ZOOM_FACTOR = 0.75
        if event.inaxes is not self.ax:
            return
        pos = (event.xdata, event.ydata, )
        x_lims = self.ax.get_xlim()
        y_lims = self.ax.get_ylim()
        mins, maxs = list(zip(x_lims, y_lims))
        if event.button == 'up':
            factor = _ZOOM_FACTOR
        elif event.button == 'down':
            factor = 1. / _ZOOM_FACTOR
        new_mins = [p - (p - mn) * factor for p, mn in zip(pos, mins)]
        new_maxs = [(mx - p) * factor + p for p, mx in zip(pos, maxs)]
        self.ax.set_xlim(new_mins[0], new_maxs[0])
        self.ax.set_ylim(new_mins[1], new_maxs[1])
        self._plot.draw()

    def change_selected_patches(self, indexes: int | list[int], selected: bool):
        """Update patches according to new state."""
        lw = self._sites_patches.get_linewidth()
        lc = self._sites_patches.get_edgecolor()
        # print(lc)
        if not isinstance(indexes, Iterable):
            indexes = [indexes]
        new_lw = _SELECTED_LINEWITDH if selected else _UNSELECTED_LINEWITDH
        new_color = _SELECTED_LINECOLOR if selected else _UNSELECTED_LINECOLOR
        for idx in indexes:
            lw[idx] = new_lw
            lc[idx] = new_color
        self._sites_patches.set_linewidth(lw)
        self._sites_patches.set_edgecolor(lc)
        self._plot.draw_idle()

    def _on_pick(self, event: PickEvent):
        if (idx := getattr(event, "ind", None)) is None:
            _lgr.error("No index in pick event")
            return
        idx = idx[0]

        if event.mouseevent.button == 1 and event.mouseevent.dblclick:
            self.change_selected_patches(idx, self._parent.site_toggle_selection(idx))

    def zoom_to(self, lim_x, lim_y):
        self.ax.set_xlim(*lim_x)
        self.ax.set_ylim(*lim_y)
        self._plot.draw()

    def _init_graphs(self):
        self.ax.clear()
        self._ungrouped_scatter: Line2D = self.ax.plot([], [], marker="o", ls="", ms=self._marker_size, c="blue")[0]
        self._grouped_scatter: Line2D = self.ax.plot([], [], marker="o", ls="", ms=self._marker_size, c="red")[0]
        self._events_scatter = self.ax.add_collection(EllipseCollection([], [], []))
        self._sites_patches = PatchCollection([])
        self._sites_scatter = self.ax.add_collection(self._sites_patches)

    def set_data(self, new_data: SIMPLERData):
        """Cleans everything."""
        self._data = new_data
        self._init_graphs()
        self._update_graphs()
        self._graph_selection_changed(1)
        self.ax.margins(.05)
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
        self._events_scatter = self.ax.add_collection(
            EllipseCollection(
                widths=sigmas, heights=sigmas, angles=0, units='xy',
                # facecolors=plt.cm.hsv(duraciones / duraciones.max()),
                offsets=self._data.get_events_locations(), transOffset=self.ax.transData,
                alpha=0.4,
            )
        )
        self._sites_scatter.remove()
        sites = self._data.get_sites()
        self._patches = []
        for site in sites:
            or_points = _np.array([_.center for _ in site])
            if len(site) < 3:
                vertex = or_points
            else:
                ch = ConvexHull(or_points)
                vertex = ch.points[ch.vertices]
            self._patches.append(Polygon(vertex, closed=True,))
        # alpha is set on face and edgecolors
        p = PatchCollection(self._patches, match_original=False, picker=True)
        p.set_color(_UNSELECTED_LINECOLOR)
        # This is needed to be able to access individual Patch properties:
        #    using 'match_original=True' seems to freeze the properties
        p.set_edgecolor([_UNSELECTED_LINECOLOR] * len(self._patches))
        p.set_linewidth([_UNSELECTED_LINEWITDH] * len(self._patches))
        self._sites_scatter = self.ax.add_collection(p)
        self._sites_patches = p
        self._sites_scatter.set_picker(True)

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
        if self._sites_scatter:
            if self._sites_chk.checkState():
                self._sites_scatter.set_visible(True)
            else:
                self._sites_scatter.set_visible(False)
        self._plot.draw()
        # self._plot.draw_idle()