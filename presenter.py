import sys
from pathlib import Path
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QFileDialog

from view import View
from analysis import AnalysisWorker

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
        
    def show_ui(self):
        self._view.show()
        
    def _make_ui_connect(self):
        """
        this functions makes all the connection with the signals coming from the UI
        """
        self._view.ui.browse_file_button.clicked.connect(self._browse_file)
        self._view.ui.analysis_button.clicked.connect(self._start_analysis)
        
    def _make_analysis_connect(self):
        """
        this functions makes all the connection with the signals coming from the analysis thread
        """
        self.signals.request_start_analysis.connect(self._analysis_worker.do_analysis)
        
    def _browse_file(self, *args):
        """
        this function opens a window to choose the hdf5 file used for calibration
        """
        filepath_str, _ = QFileDialog.getOpenFileName(
            self._view,
            caption="Select calibration file",
            filter="(*.hdf5)"
        )
        if filepath_str is not None:
            self.data_path = Path(filepath_str)
            self.metadata_path = self.data_path.parent / Path(self.data_path.stem + ".yaml")
            self._view.ui.analysis_status_label.setText("")
            self._view.ui.dir_label.setText(str(self.data_path.parent))            
            self._view.ui.filename_label.setText(self.data_path.stem)
            
    def _start_analysis(self):
        """
        this function performs the analysis steps needed before fitting SIMPLER parameters
        """
        # analysis thread preparation, to allow dynamic updating of the GUI
        self._analysis_worker = AnalysisWorker(self.data_path, self.metadata_path)
        self._analysis_thread = QThread()
        self._make_analysis_connect()
        self._analysis_worker.moveToThread(self._analysis_thread)
        self._analysis_thread.start()
        # actual analysis
        if self._analysis_worker.data.is_file_open:
            self.signals.request_start_analysis.emit()
        
        
        