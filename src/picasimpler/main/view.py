from pathlib import Path
from PyQt6 import QtCore
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtGui import QShortcut, QKeySequence
import pyqtgraph as pg

from picasimpler.UI.calibration_ui import Ui_MainWindow
from picasimpler.main.analysis import SIMPLER, Clusterization
from picasimpler.helpers.status import UIColor

_translate = QtCore.QCoreApplication.translate

class View(QMainWindow):
    """
    this class produces and manages the GUI
    """
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.create_shortcuts()
        self._setup_plot_widgets()
        self.set_color_frame(UIColor.GRAY)
        self.setWindowTitle(_translate("MainWindow", "SIMPLER calibration GUI"))
        
    def create_shortcuts(self):
        """
        This fucntion connects some keyboard inputs to buttons in the GUI
        """
        prev_orig_shcut = QShortcut(QKeySequence(QtCore.Qt.Key.Key_Left), self)
        prev_orig_shcut.activated.connect(self.ui.prev_orig_button.click)
        next_orig_shcut = QShortcut(QKeySequence(QtCore.Qt.Key.Key_Right), self)
        next_orig_shcut.activated.connect(self.ui.next_orig_button.click)
        disc_selec_orig_shcut = QShortcut(QKeySequence(QtCore.Qt.Key.Key_Space), self)
        disc_selec_orig_shcut.activated.connect(self.ui.disc_selec_orig_button.click)
        
    def reset_ui(self):
        """
        This function calls all the basic functions which reset UI elements
        """
        self.set_color_frame(UIColor.GRAY)
        self.reset_counters_onui()
        self.reset_plots()
        
    def _setup_plot_widgets(self):
        """
        this function, called at View instatiation, sets up the plots to be filled later on
        """
        self.ui.xn_widget.invertY(True)
        self.ui.yn_widget.invertY(True)
        self._xn_plot = self.ui.xn_widget.getPlotItem()
        self._yn_plot = self.ui.yn_widget.getPlotItem()
        
    def set_color_frame(self, frame_color: UIColor):
        """
        This function sets the color of the frame around the scatter plots to indicate whether
        the current origami is selcted for calibration or not
        """
        self.ui.color_frame.setStyleSheet(
            "QFrame { background-color: "+ frame_color.rgba_str +"; }"
        )
        
    def update_data_file_onui(self, data_path: Path):
        """
        This function updates the file name and parent directory reported on UI
        """
        self.ui.dir_label.setText(str(data_path.parent))            
        self.ui.filename_label.setText(data_path.stem)
        
    def update_analysis_status_onui(self, analysis_status_msg: str):
        """
        This function updates the analysis status reported on UI
        """
        self.ui.analysis_status_label.setText(analysis_status_msg)
        
    def update_analysis_counter(self, elem_num, tot_elem):
        """
        This function changes the color of the frame containing the scatter plots
        thta indicates whether the currenbt origami is selected for calibration
        """
        self.ui.analysis_counter_label.setText(str(elem_num)+" / "+str(tot_elem))
        
    def update_curr_orig_count(self, curr_orig_num, tot_orig):
        self.ui.current_orig_label.setText(
            str(curr_orig_num) + " / " + str(tot_orig)
        )
        
    def reset_counters_onui(self):
        """
        This function is called whenever a new file browsed, and it empties counters displayed on the UI
        """
        self.ui.analysis_counter_label.setText("")
        self.ui.current_orig_label.setText("")
        
    def reset_plots(self):
        """
        This function empties the xN and yN plots on the UI
        """
        self._xn_plot.clear()
        self._yn_plot.clear()
        self._xn_plot.enableAutoRange()
        self._yn_plot.enableAutoRange()
        
    def plot_orig(self, simpler: SIMPLER, orig_num: int):
        """
        This function plots the xN and yN projections of all the localization of the chosen origami
        """
        self.reset_plots()
        xn_scatter = pg.ScatterPlotItem(
            simpler.get_loc_x(orig_num),
            simpler.get_loc_n(orig_num),
            brush=UIColor.V.brush,
            pen=UIColor.V.pen
        )
        yn_scatter = pg.ScatterPlotItem(
            simpler.get_loc_y(orig_num),
            simpler.get_loc_n(orig_num),
            brush=UIColor.V.brush,
            pen=UIColor.V.pen
        )
        self._xn_plot.addItem(xn_scatter)
        self._yn_plot.addItem(yn_scatter)
        self.set_color_frame(UIColor.GRAY)
        
    def plot_orig_wclust(self, simpler: SIMPLER, clust: Clusterization, orig_num: int, is_selected: bool):
        """
        This function plots the xN and yN projections of all the localization of the chosen origami,
        superimposed with the corresponding clusterization results
        """
        self.reset_plots()
        xn_scatter_clust = pg.ScatterPlotItem(
            clust.get_clust_x(orig_num),
            clust.get_clust_n(orig_num),
            brush=UIColor.B.brush,
            pen=UIColor.B.pen
        )
        yn_scatter_clust = pg.ScatterPlotItem(
            clust.get_clust_y(orig_num),
            clust.get_clust_n(orig_num),
            brush=UIColor.B.brush,
            pen=UIColor.B.pen
        )
        xn_scatter_noise = pg.ScatterPlotItem(
            clust.get_noise_x(orig_num),
            clust.get_noise_n(orig_num),
            brush=UIColor.LB.brush,
            pen=UIColor.LB.pen
        )
        yn_scatter_noise = pg.ScatterPlotItem(
            clust.get_noise_y(orig_num),
            clust.get_noise_n(orig_num),
            brush=UIColor.LB.brush,
            pen=UIColor.LB.pen
        )
        xn_clust_centers = pg.ScatterPlotItem(
            clust.clust_means[orig_num, :, 0],
            clust.clust_means[orig_num, :, 2],
            brush=UIColor.Y.brush,
            pen=UIColor.Y.pen
        )
        yn_clust_centers = pg.ScatterPlotItem(
            clust.clust_means[orig_num, :, 1],
            clust.clust_means[orig_num, :, 2],
            brush=UIColor.Y.brush,
            pen=UIColor.Y.pen
        )
        self._xn_plot.addItem(xn_scatter_clust)
        self._yn_plot.addItem(yn_scatter_clust)
        self._xn_plot.addItem(xn_scatter_noise)
        self._yn_plot.addItem(yn_scatter_noise)
        self._xn_plot.addItem(xn_clust_centers)
        self._yn_plot.addItem(yn_clust_centers)
        if is_selected:
            self.set_color_frame(UIColor.G)
            self.update_selec_button_todiscard()
        else:
            self.set_color_frame(UIColor.R)
            self.update_discard_button_toselec()
        
    def update_discard_button_toselec(self):
        """
        This function set the text on the discard/select origami button to "select"
        """
        self.ui.disc_selec_orig_button.setText("Select\norigami")
        
    def update_selec_button_todiscard(self):
        """
        This function set the text on the discard/select origami button to "discard"
        """
        self.ui.disc_selec_orig_button.setText("Discard\norigami")
        
    def update_selec_orig_counter(self, selec_orig_list):
        """
        This function updates the counter of selected origamis, reporting the current number of
        selected origamis on the total number of origamis
        """
        self.ui.selec_orig_label.setText(
            str(sum(selec_orig_list)) + " / " + str(len(selec_orig_list))
        )
