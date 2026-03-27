from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QLineEdit
    
class CustomQLineEdit(QLineEdit):
    '''
    This custom QLineEdit class inherits the usual QLineEdit class but adds a
    signal which is emitted in 2 cases:
    1) return key is pressed
    2) editingFinished is emitted and something in the text was changed manually (so, not
    from the code) since last time it was emitted.
    This is basically a custom version of editingFinished which is not emitted if the
    field was changed from the code
    '''
    manual_editing_finished = pyqtSignal()
    def __init__(self, parent):
        super().__init__(parent)
        self._is_manually_edited: bool = False        
        # when text is edited manually, change status
        self.textEdited.connect(self._set_edited_totrue)
        # pressing return key should alwas emit the signal
        self.returnPressed.connect(self._emit_manual_editing_finished)
        # this will emit the signal only if something was changed manually
        self.editingFinished.connect(self._manual_editing_check)
        
    def _set_edited_totrue(self):
        self._is_manually_edited = True
        
    def _emit_manual_editing_finished(self):
        self.manual_editing_finished.emit()
        self._is_manually_edited = False
    
    def _manual_editing_check(self):
        if self._is_manually_edited:
            self._emit_manual_editing_finished()