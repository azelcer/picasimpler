from __future__ import annotations

import logging as _lgn
from pathlib import Path
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QFileDialog
from functools import wraps

from picasimpler.main.view import View
from picasimpler.main.analysis import AnalysisWorker
from picasimpler.helpers.status import AnalysisStatus, FrameColor

_lgn.basicConfig()
_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.INFO)


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
        
    def check_analysis_status(ref_analysis_status: AnalysisStatus):
        """
        This decorator wraps a function in a if statement that gets executed only if the
        analysis status has passed the reference step given as an argument.
        Optionally, if must_not_analysing is True, it wraps with another if statement that checks
        whether the program is not analyzing right now
        """
        def check_analysis_status_innderdecor(decorated_func):
            @wraps(decorated_func)
            def wrapper_func(self: Presenter, *args, **kwargs):
                if self.analysis_status.passed_analysis_step(ref_analysis_status) and not self.analysis_status.is_analysing:
                    return decorated_func(self, *args, **kwargs)
                else:
                    return
            return wrapper_func
        return check_analysis_status_innderdecor
        
    @property
    def analysis_status(self: Presenter):
        return self._analysis_status
    
    @analysis_status.setter
    def analysis_status(self, status: AnalysisStatus):
        self._analysis_status = status
        self._view.update_analysis_status_onui(status.msg)
        
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
        self._view.ui.disc_selec_orig_button.clicked.connect(self._disc_selec_orig)
        
    def _make_analysis_connect(self):
        """
        this functions makes all the connection with the signals coming from the analysis thread
        """
        # connect signals from presenter to analysis worker
        self.signals.request_start_filtering.connect(self._analysis_worker.do_filt)
        self.signals.request_start_clustering.connect(self._analysis_worker.do_clust)
        # connect signals from analysis worker to presenter
        self._analysis_worker.signals.tell_analysis_step_start.connect(self._on_new_analysis_step)
        self._analysis_worker.signals.tell_filt_done.connect(self._on_filt_done)
        self._analysis_worker.signals.tell_clust_done.connect(self._on_clust_done)
        # connect signals from other analysis helper classes to presenter
        self._analysis_worker.simpler_signals.tell_analysis_elem_done.connect(self._on_new_analysis_elem)
        self._analysis_worker.clust_signals.tell_analysis_elem_done.connect(self._on_new_analysis_elem)
        
    @pyqtSlot()
    def _browse_file(self):
        """
        this function opens a window to choose the hdf5 file used for calibration
        """
        filepath_str, _ = QFileDialog.getOpenFileName(
            self._view,
            directory=str(Path.home()), # home directory, OS independent
            caption="Select calibration file",
            filter="(*.hdf5)"
        )
        if filepath_str:
            self.data_path = Path(filepath_str)
            self.metadata_path = self.data_path.parent / Path(self.data_path.stem + ".yaml")
            self._reset_analysis()
            self._prep_analysis()
            if self.analysis_status==AnalysisStatus.DATA_LOADED:
                self._view.reset_ui()
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
        if self._analysis_worker.data.is_data_file_open and self._analysis_worker.data.is_metadata_file_open:
            self.analysis_status = AnalysisStatus.DATA_LOADED
        else:
            self._reset_analysis()
            return
        # make signal connections
        self._make_analysis_connect()
        self._analysis_worker.moveToThread(self._analysis_thread)
        self._analysis_thread.start()
            
    @pyqtSlot()
    @check_analysis_status(AnalysisStatus.DATA_LOADED)
    def _start_filtering(self):
        """
        this function tells the analysis worker to perform all the filtering steps needed before site clustering
        """
        self.signals.request_start_filtering.emit()
    
    @pyqtSlot()
    @check_analysis_status(AnalysisStatus.FILT_DONE)
    def _start_clustering(self):
        """
        This function tells the analysis worker to start site clusterization procedure
        """
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
        self.curr_displ_orig_num = 0
        self._view.plot_orig(self._analysis_worker.simpler, self.curr_displ_orig_num)
        self._view.update_curr_orig_count(self.curr_displ_orig_num + 1, self._analysis_worker.data.tot_orig)
        self._view.set_color_frame(FrameColor.GRAY)

    @pyqtSlot(bool)
    def _on_clust_done(self, are_there_clust):
        """
        This function is called once the site clusterization is completed.
        It displayed the first origami scatter plot with clustering
        """
        if not are_there_clust:
            self.analysis_status = AnalysisStatus.FILT_DONE
            _lgr.warning("No valid clusters found")
            return
        else:
            self.analysis_status = AnalysisStatus.CLUST_DONE
            self.curr_displ_orig_num = 0
            self._view.plot_orig_wclust(
                self._analysis_worker.simpler,
                self._analysis_worker.clust,
                self.curr_displ_orig_num,
                self._analysis_worker.clust.selec_orig_list[self.curr_displ_orig_num]
            )
            self._view.update_curr_orig_count(self.curr_displ_orig_num + 1, self._analysis_worker.clust.tot_orig_kept)
            self._view.update_selec_orig_counter(self._analysis_worker.clust.selec_orig_list)

    def _do_plot_shift(self, shift: int):
        if self.analysis_status.passed_analysis_step(AnalysisStatus.CLUST_DONE):
            self.curr_displ_orig_num += shift
            self.curr_displ_orig_num = self.curr_displ_orig_num % self._analysis_worker.clust.tot_orig_kept
            self._view.update_curr_orig_count(self.curr_displ_orig_num + 1, self._analysis_worker.clust.tot_orig_kept)
            self._view.plot_orig_wclust(
                self._analysis_worker.simpler,
                self._analysis_worker.clust,
                self.curr_displ_orig_num,
                self._analysis_worker.clust.selec_orig_list[self.curr_displ_orig_num]
            )
        elif self.analysis_status.passed_analysis_step(AnalysisStatus.FILT_DONE):
            self.curr_displ_orig_num += shift
            self.curr_displ_orig_num = self.curr_displ_orig_num % self._analysis_worker.data.tot_orig
            self._view.update_curr_orig_count(self.curr_displ_orig_num + 1, self._analysis_worker.data.tot_orig)
            self._view.plot_orig(self._analysis_worker.simpler, self.curr_displ_orig_num)

    @pyqtSlot()
    def _order_plot_next_orig(self):
        self._do_plot_shift(1)

    @pyqtSlot()
    def _order_plot_prev_orig(self):
        self._do_plot_shift(-1)
        
    @pyqtSlot()
    @check_analysis_status(AnalysisStatus.CLUST_DONE)
    def _disc_selec_orig(self):
        """
        This function discards/select an origami based on its current state,
        and update the corresponsing button and frame color
        """
        if self._analysis_worker.clust.selec_orig_list[self.curr_displ_orig_num]:
            self._analysis_worker.clust.selec_orig_list[self.curr_displ_orig_num] = False
            self._view.set_color_frame(FrameColor.RED)
            self._view.update_discard_button_toselec()
            self._view.update_selec_orig_counter(self._analysis_worker.clust.selec_orig_list)
        else:
            self._analysis_worker.clust.selec_orig_list[self.curr_displ_orig_num] = True
            self._view.set_color_frame(FrameColor.GREEN)
            self._view.update_selec_button_todiscard()
            self._view.update_selec_orig_counter(self._analysis_worker.clust.selec_orig_list)
