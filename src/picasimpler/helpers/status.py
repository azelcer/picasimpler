from enum import Enum, auto

class AnalysisStatus(Enum):
    """
    This Enum class contains all the possible statuses of the analysis.
    It's important that they are in order (same order as they will be done in the analysis)
    and each one has its own message in get_msg
    """
    
    msg: str
    is_analysing: bool
    
    def __new__(cls, value, msg, is_analysing):
        obj = object.__new__(cls)
        obj._value_ = cls._generate_next_value_(
            name=None,
            start=1,
            count=len(cls.__members__),
            last_values=[m.value for m in cls.__members__.values()]
        )
        obj.msg = msg
        obj.is_analysing = is_analysing
        return obj
    
    PRE_ANALYSIS = (auto(), "Ready to analyze", False)
    KIN_FILT = (auto(), "Kinetics filtering", True)
    SIMPLER_FILT = (auto(), "SIMPLER localization filtering", True)
    FILT_DONE = (auto(), "All filtering steps completed", False)
    SITE_CLUST = (auto(), "PAINT site clustering", True)
    CLUST_DONE = (auto(), "Clusterization completed", False)
    
    def passed_analysis_step(self, reference_step):
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

