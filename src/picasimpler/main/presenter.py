import sys
from pathlib import Path
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QFileDialog

from picasimpler.main.view import View
from picasimpler.main.analysis import AnalysisWorker
from picasimpler.helpers.status import AnalysisStatus, FrameColor

class PresenterSignals(QObject):
    request_start_filtering = pyqtSignal()
    request_start_clustering = pyqtSignal()

class Presenter(QObject):
    """
    this class manages connections and signals between frontend and backend
    """
    def __init__(self, view: View):
        super().__init__()
        self._view = view
        self.signals = PresenterSignals()
        self._make_ui_connect()
        # initialize some variables
        self._analysis_worker = None
        self._analysis_thread = None
        self.tot_elem_curr_analysis_step = 0
        self.curr_displ_orig_num = None
        self.analysis_status = AnalysisStatus.PRE_ANALYSIS
        
    def show_ui(self):
        self._view.show()
        
    def _make_ui_connect(self):
        """
        this functions makes all the connection with the signals coming from the UI
        """
        # connect signals from UI to presenter
        self._view.ui.browse_file_button.clicked.connect(self._browse_file)
        self._view.ui.filter_button.clicked.connect(self._start_filtering)
        self._view.ui.cluster_button.clicked.connect(self._start_clustering)
        self._view.ui.next_orig_button.clicked.connect(self._order_plot_next_orig)
        self._view.ui.prev_orig_button.clicked.connect(self._order_plot_prev_orig)
        
    def _make_analysis_connect(self):
        """
        this functions makes all the connection with the signals coming from the analysis thread
        """
        # connect signals from presenter to analysis worker
        self.signals.request_start_filtering.connect(self._analysis_worker.do_filt)
        self.signals.request_start_clustering.connect(self._analysis_worker.do_clust)
        # connect signals from analysis worker to presenter
        self._analysis_worker.signals.tell_analysis_step_start.connect(self._on_new_analysis_step)
        self._analysis_worker.signals.tell_analysis_elem_done.connect(self._on_new_analysis_elem)
        self._analysis_worker.signals.tell_filt_done.connect(self._on_filt_done)
        self._analysis_worker.signals.tell_clust_done.connect(self._on_clust_done)
        
    def _browse_file(self, *args):
        """
        this function opens a window to choose the hdf5 file used for calibration
        """
        if not self.analysis_status.is_analysing:
            filepath_str, _ = QFileDialog.getOpenFileName(
                self._view,
                directory=str(Path.home()), # home directory, OS independent
                caption="Select calibration file",
                filter="(*.hdf5)"
            )
            if filepath_str:
                self._reset_analysis()
                self._view.update_analysis_status_onui(self.analysis_status.msg)
                self._view.reset_ui()
                self.data_path = Path(filepath_str)
                self.metadata_path = self.data_path.parent / Path(self.data_path.stem + ".yaml")
                self._view.update_data_file_onui(self.data_path)
            
    def _reset_analysis(self):
        """
        This function kills previous analysis thread if still ongoing
        """
        if self._analysis_thread is not None:
            self._analysis_thread.quit()
            self._analysis_thread.wait()
        self._analysis_worker = None
        self.analysis_status = AnalysisStatus.PRE_ANALYSIS
            
    def _prep_analysis(self):
        """
        This function starts a new analysis thread, creates a new analysis worker and connects its signals
        to the Presenter
        """
        # analysis thread preparation, to allow dynamic updating of the GUI
        self._analysis_worker = AnalysisWorker(self.data_path, self.metadata_path)
        self._analysis_thread = QThread()
        # start loading data for analysis
        self._analysis_worker.load_data()
        # make signal connections
        self._make_analysis_connect()
        self._analysis_worker.moveToThread(self._analysis_thread)
        self._analysis_thread.start()
            
    def _start_filtering(self):
        """
        this function tells the analysis worker to perform all the filtering steps needed before site clustering
        """
        if not self.analysis_status.is_analysing:
            self._reset_analysis()
            self._prep_analysis()
            # actual analysis
            if self._analysis_worker.data.is_data_file_open and self._analysis_worker.data.is_metadata_file_open:
                self.signals.request_start_filtering.emit()
            else:
                self._analysis_thread.quit()
                self._analysis_thread.wait()
                self._analysis_worker = None
            
    def _start_clustering(self):
        """
        This function tells the analysis worker to start site clusterization procedure
        """
        if not self.analysis_status.is_analysing:
            if self.analysis_status.passed_analysis_step(AnalysisStatus.FILT_DONE):
                self.signals.request_start_clustering.emit()
                
            
    @pyqtSlot(AnalysisStatus, int)
    def _on_new_analysis_step(self, analysis_status: AnalysisStatus, tot_elem_curr_analysis_step: int):
        """
        this function is called whenever a new analysis step is started by the analysis worker.
        It tells the View to update the analysis status on UI, and it updates the total number of
        elements in the current analysis step
        """
        self.analysis_status = analysis_status
        self.tot_elem_curr_analysis_step = tot_elem_curr_analysis_step
        self._view.update_analysis_counter(0, self.tot_elem_curr_analysis_step)
        self._view.update_analysis_status_onui(self.analysis_status.msg)
        
    @pyqtSlot(int)
    def _on_new_analysis_elem(self, elem_num: int):
        """
        This function is called everytime a new individual element is analyzed within an
        analysis step. It tells the View to update the counter on the UI
        """
        self._view.update_analysis_counter(elem_num, self.tot_elem_curr_analysis_step)
        
    @pyqtSlot()
    def _on_filt_done(self):
        """
        This function is called once all the filtering steps are done.
        It displays the first origami scatter plot without clustering
        """
        self.analysis_status = AnalysisStatus.FILT_DONE
        self._view.update_analysis_status_onui(self.analysis_status.msg)
        self.curr_displ_orig_num = 0
        self._view.plot_orig(self._analysis_worker.data.simpler_locs, self.curr_displ_orig_num)
        self._view.update_curr_orig_count(self.curr_displ_orig_num + 1, self._analysis_worker.data.tot_orig)
        
    @pyqtSlot()
    def _on_clust_done(self):
        """
        This function is called once the site clusterization is completed.
        It displayed the first origami scatter plot with clustering
        """
        self.analysis_status = AnalysisStatus.CLUST_DONE
        self._view.update_analysis_status_onui(self.analysis_status.msg)
        self.curr_displ_orig_num = 0
        self._view.plot_orig_wclust(self._analysis_worker.data.simpler_locs, self._analysis_worker.data.cluster_res, self.curr_displ_orig_num)
        self._view.update_curr_orig_count(self.curr_displ_orig_num + 1, self._analysis_worker.data.tot_orig_after_clust)
        
    @pyqtSlot()
    def _order_plot_next_orig(self):
        if self.analysis_status.passed_analysis_step(AnalysisStatus.CLUST_DONE):
            self.curr_displ_orig_num += 1
            self.curr_displ_orig_num = self.curr_displ_orig_num % self._analysis_worker.data.tot_orig_after_clust
            self._view.update_curr_orig_count(self.curr_displ_orig_num + 1, self._analysis_worker.data.tot_orig_after_clust)
            self._view.plot_orig_wclust(self._analysis_worker.data.simpler_locs, self._analysis_worker.data.cluster_res, self.curr_displ_orig_num)
        elif self.analysis_status.passed_analysis_step(AnalysisStatus.FILT_DONE):
            self.curr_displ_orig_num += 1
            self.curr_displ_orig_num = self.curr_displ_orig_num % self._analysis_worker.data.tot_orig
            self._view.update_curr_orig_count(self.curr_displ_orig_num + 1, self._analysis_worker.data.tot_orig)
            self._view.plot_orig(self._analysis_worker.data.simpler_locs, self.curr_displ_orig_num)

        
    @pyqtSlot()
    def _order_plot_prev_orig(self):
        if self.analysis_status.passed_analysis_step(AnalysisStatus.CLUST_DONE):
            self.curr_displ_orig_num -= 1
            self.curr_displ_orig_num = self.curr_displ_orig_num % self._analysis_worker.data.tot_orig_after_clust
            self._view.update_curr_orig_count(self.curr_displ_orig_num + 1, self._analysis_worker.data.tot_orig_after_clust)
            self._view.plot_orig_wclust(self._analysis_worker.data.simpler_locs, self._analysis_worker.data.cluster_res, self.curr_displ_orig_num)
        elif self.analysis_status.passed_analysis_step(AnalysisStatus.FILT_DONE):
            self.curr_displ_orig_num -= 1
            self.curr_displ_orig_num = self.curr_displ_orig_num % self._analysis_worker.data.tot_orig
            self._view.update_curr_orig_count(self.curr_displ_orig_num + 1, self._analysis_worker.data.tot_orig)
            self._view.plot_orig(self._analysis_worker.data.simpler_locs, self.curr_displ_orig_num)
        
