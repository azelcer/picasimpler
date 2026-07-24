from pathlib import Path
import numpy as np

# SIMPLER filtering
SPAT_TOL_NM_DEF = 30.0
SPAT_TOL_NM_MIN = 5
# kinetics filtering
MAX_FIRST_FRAME_PERC = 0.33
MIN_LAST_FRAME_PERC = 0.66
FRAME_MEDIAN_PERC_RANGE = [0.33, 0.66]
MAX_ON_FRAMES_PERC = 0.8
# origami structure
HB_Z_SITES_NM = np.array([8.04,  46.17,  87.92,  144.31, 198.47])
RIFLE_Z_SITES_NM = np.array([6.78,  39.33, 73.67, 110.06])
# clustering
PRECLUST_GAMMA_DEF = 0.05
PRECLUST_EPS_DEF = 40
MAX_PRECLUST_GAMMA = 0.2
MIN_GOOD_LOC = 40
RIFLE_N_CLUST_EXP = 4
HB_N_CLUST_EXP = 5
DF_REF_VAL_NM = 100
# setup parameters
LAMDBA_EXC_DEF = 560
LAMBDA_EM_DEF = 600
LAMBDA_MIN = 0
NA_DEF = 1.45
# objetive collection efficiency simulation parameters
Z_BASELINE_NM = 0
Z_SIM_FIT_ARR = np.arange(0, 300, 0.5)
NI_DEF = 1.518
NS_DEF = 1.333
# result directory
RES_DIR = Path("results")
# initial guesses and bounds for calibration fit
ANGLE_FIXED_MIN = 62
ANGLE_FIXED_DEF = 70
ANGLE_FIXED_MAX = 75
ALPHA_GUESS = 0.9
ALPHA_MAX = np.inf
ALPHA_FIXED = 0.9
D_GUESS = 100
Z_MIN_NM = 2
# parameters for calibration plot
CALIB_PLOT_RANGE_NM = 250
CALIB_PLOT_PTS = 2000