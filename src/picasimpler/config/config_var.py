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
Z_SITES_NM = [7, 41.6, 76.9, 113.5]
# clustering
PRECLUST_GAMMA_DEF = 0.1
PRECLUST_EPS_DEF = 20
MAX_PRECLUST_GAMMA = 1/len(Z_SITES_NM)
MIN_GOOD_LOC = 100
N_CLUST_EXP = 4
DF_REF_VAL_NM = 120
# setup parameters
LAMDBA_EXC_DEF = 560
LAMBDA_EM_DEF = 570
LAMBDA_MIN = 480
NA_IDX_DEF = 2
# objetive collection efficiency simulation parameters
Z_SIM_DISCR = [5, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
Z_SIM_FIT_ARR = np.arange(5, 500, 0.5)
LAMBDA_EM_DISCR = [500, 530, 560, 590, 620, 670, 700, 720]
NI_DEF = 1.517
NS_DEF = 1.333
# result directory
RES_DIR = Path("results")
# initial guesses and bounds for calibration fit
ALPHA_GUESS = 0.8
ALPHA_MAX = 1
ALPHA_FIXED = 0.9
D_GUESS = 100
# parameters for calibration plot
CALIB_PLOT_RANGE_NM = 150
CALIB_PLOT_PTS = 2000
# calibration method
#CALIB_MODE = 'no_appr'
CALIB_MODE = 'no_appr_fix_alpha'
#CALIB_MODE = 'exp_appr'
