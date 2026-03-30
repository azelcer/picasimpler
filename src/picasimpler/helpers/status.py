from __future__ import annotations

from enum import Enum, auto

import pyqtgraph as pg
from PyQt6.QtGui import QPen, QBrush, QColor

from picasimpler.helpers.utils import hex_to_rgba, cust_auto


class AnalysisStatus(Enum):
    """
    This Enum class contains all the possible statuses of the analysis.
    It's important that they are in order (same order as they will be done in the analysis)
    and each one has a message and a bool that states whether it is an active analysis step,
    meaning the analysis worker is busy
    """
    msg: str
    is_analysing: bool

    def __new__(cls, prog_order, msg, is_analysing):

        obj = object.__new__(cls)
        obj._value_ = cust_auto(cls, prog_order)
        obj.msg = msg
        obj.is_analysing = is_analysing
        return obj

    PRE_ANALYSIS = (auto(), "Please browse file", False)
    DATA_LOADED = (auto(), "Ready to analyze", False)
    KIN_FILT = (auto(), "Kinetics filtering...", True)
    SIMPLER_FILT = (auto(), "SIMPLER localization filtering...", True)
    FILT_DONE = (auto(), "All filtering steps completed", False)
    PRE_CLUST = (auto(), "Pre-clustering de-noising...", True)
    SITE_CLUST = (auto(), "Site clustering...", True)
    CLUST_DONE = (auto(), "Clusterization completed", False)

    def passed_analysis_step(self, reference_step: AnalysisStatus):
        return self.value >= reference_step.value


class UIColor(Enum):
    """
    This Enum class contains all the possible colors for plots and UI elements,
    in hexadecimal format, but it has methods to access the rgb format too.
    """
    
    rgba: tuple
    rgba_str: str
    col: QColor
    pen: QPen
    brush: QBrush
    
    def __new__(cls, value):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.rgba = hex_to_rgba(value)
        obj.rgba_str = "rgba" + str(obj.rgba)
        obj.col = QColor(*obj.rgba)
        obj.pen = pg.mkPen(color=value)
        obj.brush = pg.mkBrush(color=value)
        return obj
    
    W = "#FFFFFF"
    GRAY = "#323232"
    DR ="#B10000"
    MAG = "#D71B60"
    G = "#05FE04"
    DG = "#348B01FF"
    B = "#1582e9ff"
    IND = "#4a2dbe66"
    LB = "#2dbeb766"
    V = "#821ec066"
    Y = "#ffff00"
    OG = "#ff8800"
    
class MessageType(Enum):
    '''
    Enum class to list all type of message that can be printed on UI,
    with respective default colors
    '''
    
    title: str
    col: QColor
    
    def __new__(
        cls, description: str|None, title, color: QColor|None
    ):
        msg_obj = object.__new__(cls)
        msg_obj._value_ = description
        msg_obj.title = title
        msg_obj.col = color
        return msg_obj
    
    SIMPLE = 'simple message', None, None
    INFO = 'info message', 'Info', UIColor.B.col
    STATUS = 'status message', 'Status', UIColor.DG.col
    WARNING = 'warning message', 'Warning', UIColor.OG.col
    ERROR = 'error message', 'ERROR', UIColor.DR.col
    
if __name__=="__main__":
    for memb in AnalysisStatus.__members__.values():
        print(memb.value)
        print(memb.msg)
        print(memb.is_analysing)
        
    for memb in UIColor.__members__.values():
        print(memb.value)
        print(memb.rgba_str)
        

