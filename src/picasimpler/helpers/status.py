from __future__ import annotations

from enum import Enum, auto

import pyqtgraph as pg
from PyQt6.QtGui import QPen, QBrush

from picasimpler.helpers.conversions import hex_to_rgba

def _cust_auto(enum_cls, prog_order):
    if type(prog_order) is auto:
        return len(enum_cls.__members__) + 1
    else:
        raise TypeError(f"First element of Enum member must be auto(), got {type(prog_order).__name__!r}")

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
        obj._value_ = _cust_auto(cls, prog_order)
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
    pen: QPen
    brush: QBrush
    
    def __new__(cls, value):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.rgba = hex_to_rgba(value)
        obj.rgba_str = "rgba" + str(obj.rgba)
        obj.pen = pg.mkPen(color=value)
        obj.brush = pg.mkBrush(color=value)
        return obj
    
    GRAY = "#323232"
    R = "#D71B60"
    G = "#05FE04"
    B = "#4a2dbe66"
    LB = "#2dbeb766"
    V = "#821ec066"
    Y = "#ffff00"
    OG = "#ff8800"
    
if __name__=="__main__":
    for memb in AnalysisStatus.__members__.values():
        print(memb.value)
        print(memb.msg)
        print(memb.is_analysing)
        
    for memb in UIColor.__members__.values():
        print(memb.value)
        print(memb.rgba_str)
        

