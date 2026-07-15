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
HB_Z_0 = 8.3
HB_Z_DIST_NM = [38.7, 41.0, 56.4, 54.1]
HB_Z_SITES_NM = np.array([
    HB_Z_0,
    HB_Z_0 + HB_Z_DIST_NM[0],
    HB_Z_0 + HB_Z_DIST_NM[0] + HB_Z_DIST_NM[1],
    HB_Z_0 + HB_Z_DIST_NM[0] + HB_Z_DIST_NM[1] + HB_Z_DIST_NM[2],
    HB_Z_0 + HB_Z_DIST_NM[0] + HB_Z_DIST_NM[1] + HB_Z_DIST_NM[2] + HB_Z_DIST_NM[3],
])
RIFLE_Z_SITES_NM = np.array([7, 41.6, 76.9, 113.5])
# clustering
PRECLUST_GAMMA_DEF = 0.05
PRECLUST_EPS_DEF = 40
MAX_PRECLUST_GAMMA = 0.2
MIN_GOOD_LOC = 40
RIFLE_N_CLUST_EXP = 4
HB_N_CLUST_EXP = 5
DF_REF_VAL_NM = 120
# setup parameters
LAMDBA_EXC_DEF = 560
LAMBDA_EM_DEF = 590
LAMBDA_MIN = 480
NA_IDX_DEF = 2
# objetive collection efficiency simulation parameters
Z_BASELINE_NM = 0
Z_SIM_DISCR = [0, 5, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
Z_SIM_FIT_ARR = np.arange(0, 300, 0.5)
LAMBDA_EM_DISCR = [500, 530, 560, 590, 620, 670, 700, 720]
NI_DEF = 1.517
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
SPACER_GUESS = 0
D_LONG_GUESS = 1500
# parameters for calibration plot
CALIB_PLOT_RANGE_NM = 250
CALIB_PLOT_PTS = 2000
# calibration method
#CALIB_MODE = 'no_appr'
#CALIB_MODE = 'no_appr_fix_angle'
CALIB_MODE = 'no_appr_fix_angle_each_orig'
#CALIB_MODE = 'no_appr_fix_angle_biexp'
#CALIB_MODE = 'no_appr_fix_angle_biexp_each_orig'
#CALIB_MODE = 'no_appr_spacer'
#CALIB_MODE = 'no_appr_fix_alpha'
#CALIB_MODE = 'exp_appr'
