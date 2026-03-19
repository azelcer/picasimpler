import sys
from pathlib import Path
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QFileDialog

from view import View
from analysis import AnalysisWorker
from analysis_status import AnalysisStatus

class PresenterSignals(QObject):
    request_start_analysis = pyqtSignal()

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
        
    def show_ui(self):
        self._view.show()
        
    def _make_ui_connect(self):
        """
        this functions makes all the connection with the signals coming from the UI
        """
        # connect signals from UI to presenter
        self._view.ui.browse_file_button.clicked.connect(self._browse_file)
        self._view.ui.analysis_button.clicked.connect(self._start_analysis)
        
    def _make_analysis_connect(self):
        """
        this functions makes all the connection with the signals coming from the analysis thread
        """
        # connect signals from presenter to analysis worker
        self.signals.request_start_analysis.connect(self._analysis_worker.do_analysis)
        # connect signals from analysis worker to presenter
        self._analysis_worker.signals.tell_analysis_step_start.connect(self._on_new_analysis_step)
        self._analysis_worker.signals.tell_analysis_elem_done.connect(self._on_new_analysis_elem)
        
    def _browse_file(self, *args):
        """
        this function opens a window to choose the hdf5 file used for calibration
        """
        filepath_str, _ = QFileDialog.getOpenFileName(
            self._view,
            directory=str(Path.home()), # home directory, OS independent
            caption="Select calibration file",
            filter="(*.hdf5)"
        )
        if filepath_str is not None:
            self._reset_analysis()
            self._view.update_analysis_status(AnalysisStatus.PRE_ANALYSIS.value)
            self._view.update_analysis_counter(0,0)
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
            
    def _start_analysis(self):
        """
        this function starts a new analysis thread, creates a new analysis worker and then tells it to
        performs all the analysis steps needed before fitting SIMPLER parameters
        """
        self._reset_analysis()
        # analysis thread preparation, to allow dynamic updating of the GUI
        self._analysis_worker = AnalysisWorker(self.data_path, self.metadata_path)
        self._analysis_thread = QThread()
        self._make_analysis_connect()
        self._analysis_worker.moveToThread(self._analysis_thread)
        self._analysis_thread.start()
        # actual analysis
        if self._analysis_worker.data.is_data_file_open and self._analysis_worker.data.is_metadata_file_open:
            self.signals.request_start_analysis.emit()
        else:
            self._analysis_thread.quit()
            self._analysis_thread.wait()
            self._analysis_worker = None
            
    @pyqtSlot(AnalysisStatus, int)
    def _on_new_analysis_step(self, analysis_status: AnalysisStatus, tot_elem_curr_analysis_step):
        """
        this function is called whenever a new analysis step is started by the analysis worker.
        It tells the View to update the analysis status on UI, and it updates the total number of
        elements in the current analysis step
        """
        self.tot_elem_curr_analysis_step = tot_elem_curr_analysis_step
        self._view.update_analysis_counter(0, self.tot_elem_curr_analysis_step)
        self._view.update_analysis_status(analysis_status.value)
        
    @pyqtSlot(int)
    def _on_new_analysis_elem(self, elem_num):
        """
        This function is called everytime a new individual element is analyzed within an
        analysis step. It tells the View to update the counter on the UI
        """
        self._view.update_analysis_counter(elem_num, self.tot_elem_curr_analysis_step)
        
    @pyqtSlot()
    def _on_analysis_done(self):
        """
        This function is called once all the analysis steps are done.
        It displays the first origami scatter plot 
        """
        pass