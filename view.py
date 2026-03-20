from pathlib import Path
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QMainWindow
import pyqtgraph as pg
from UI.calibration_ui import Ui_MainWindow

from analysis import ClusterData

class View(QMainWindow):
    """
    this class produces and manages the GUI
    """
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self._setup_plot_widgets()
        
    def _setup_plot_widgets(self):
        """
        this function, called at View instatiation, sets up the plots to be filled later on
        """
        self.ui.xn_widget.invertY(True)
        self.ui.yn_widget.invertY(True)
        self._xn_plot = self.ui.xn_widget.getPlotItem()
        self._yn_plot = self.ui.yn_widget.getPlotItem()
        
    def update_data_file_onui(self, data_path: Path):
        """
        This function updates the file name and parent directory reported on UI
        """
        self.ui.dir_label.setText(str(data_path.parent))            
        self.ui.filename_label.setText(data_path.stem)
        
    def update_analysis_status(self, analysis_status: str):
        """
        This function updates the analysis status reported on UI
        """
        self.ui.analysis_status_label.setText(analysis_status)
        
    def update_analysis_counter(self, elem_num, tot_elem):
        self.ui.analysis_counter_label.setText(str(elem_num)+" / "+str(tot_elem))
        
    def update_curr_orig_count(self, curr_orig_num, tot_orig):
        self.ui.current_orig_label.setText(
            str(curr_orig_num) + " / " + str(tot_orig)
        )
        
    def plot_orig_wclust(self, clust_data: ClusterData, orig_num: int):
        """
        This function plots the xN and yN projections of all the localization of the chosen
        origami, superimposed with the corresponding clusterization results
        """
        self._xn_plot.clear()
        self._yn_plot.clear()
        xn_scatter = pg.ScatterPlotItem(clust_data.get_loc_x(orig_num), clust_data.get_loc_n(orig_num))
        yn_scatter = pg.ScatterPlotItem(clust_data.get_loc_y(orig_num), clust_data.get_loc_n(orig_num))
        xn_clust_centers = pg.ScatterPlotItem(clust_data.clust_means[orig_num, :, 0], clust_data.clust_means[orig_num, :, 2], pen='y')
        yn_clust_centers = pg.ScatterPlotItem(clust_data.clust_means[orig_num, :, 1], clust_data.clust_means[orig_num, :, 2], pen='y')
        self._xn_plot.addItem(xn_scatter)
        self._yn_plot.addItem(yn_scatter)
        self._xn_plot.addItem(xn_clust_centers)
        self._yn_plot.addItem(yn_clust_centers)
        
