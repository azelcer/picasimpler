from enum import Enum

class AnalysisStatus(Enum):
    """
    This Enum class contains all the possible statuses of teh analysis
    """
    PRE_ANALYSIS = "Ready to analyze"
    KIN_FILT = "Kinetics filtering"
    SIMPLER_FILT = "SIMPLER localization filtering"
    SITE_CLUST = "PAINT site clustering"