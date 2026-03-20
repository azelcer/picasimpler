from enum import Enum, auto

class AnalysisStatus(Enum):
    """
    This Enum class contains all the possible statuses of the analysis.
    It's important that they are in order (same order as they will be done in the analysis)
    and each one has its own message in get_msg
    """
    PRE_ANALYSIS = auto()
    LOADING_DATA = auto()
    KIN_FILT = auto()
    SIMPLER_FILT = auto()
    SITE_CLUST = auto()
    ANALYSIS_DONE = auto()
    
    def get_msg(self):
        messages = {
            AnalysisStatus.PRE_ANALYSIS: "Ready to analyze",
            AnalysisStatus.LOADING_DATA: "Loading data from file",
            AnalysisStatus.KIN_FILT: "Kinetics filtering",
            AnalysisStatus.SIMPLER_FILT: "SIMPLER localization filtering",
            AnalysisStatus.SITE_CLUST: "PAINT site clustering",
            AnalysisStatus.ANALYSIS_DONE: "All analysis steps completed",
        }
        return messages[self]
    
    def passed_analysis_step(self, reference_step):
        return self.value >= reference_step.value