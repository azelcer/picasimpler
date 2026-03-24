from __future__ import annotations

from enum import Enum, auto

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
    KIN_FILT = (auto(), "Kinetics filtering", True)
    SIMPLER_FILT = (auto(), "SIMPLER localization filtering", True)
    FILT_DONE = (auto(), "All filtering steps completed", False)
    SITE_CLUST = (auto(), "Site clustering", True)
    CLUST_DONE = (auto(), "Clusterization completed", False)
    
    def passed_analysis_step(self, reference_step: AnalysisStatus):
        return self.value >= reference_step.value
    
class FrameColor(Enum):
    """
    This Enum class contains all the possible colors of the frame
    indicating whether an origami is selected for calibration or not.
    Colors must be written as strings in the format: "rgb(r, g, b)".
    """
    GRAY = "rgb(200, 200, 200)"
    RED = "rgb(215,27,96)"
    GREEN = "rgb(5,254,4)"
    
if __name__=="__main__":
    for memb in AnalysisStatus.__members__.values():
        print(memb.value)
        print(memb.msg)
        print(memb.is_analysing)
        

