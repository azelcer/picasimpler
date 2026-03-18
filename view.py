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
        