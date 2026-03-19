from pathlib import Path
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QMainWindow

from UI.calibration_ui import Ui_MainWindow

class View(QMainWindow):
    """
    this class produces and manages the GUI
    """
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
    def update_data_file_onui(self, data_path: Path):
        self.ui.dir_label.setText(str(data_path.parent))            
        self.ui.filename_label.setText(data_path.stem)
        
    def update_analysis_status(self, analysis_status: str):
        """
        This function updates the analysis status reported on UI
        """
        self.ui.analysis_status_label.setText(analysis_status)
        
    def update_analysis_counter(self, elem_num, tot_elem):
        self.ui.analysis_counter_label.setText(str(elem_num)+" / "+str(tot_elem))
