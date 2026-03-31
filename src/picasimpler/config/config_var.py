from pathlib import Path

# SIMPLER filtering
SPAT_TOL_NM = 30.0
# kinetics filtering
MAX_FIRST_FRAME_PERC = 0.33
MIN_LAST_FRAME_PERC = 0.66
FRAME_MEDIAN_PERC_RANGE = [0.33, 0.66]
MAX_ON_FRAMES_PERC = 0.8
# origami structure
Z_SITES_NM = [9.3, 43.7, 79.9, 116.3]
# clustering
PRECLUST_GAMMA_DEF = 0.1
PRECLUST_EPS_DEF = 20
MAX_PRECLUST_GAMMA = 1/len(Z_SITES_NM)
MIN_GOOD_LOC = 100
N_CLUST_EXP = 4
LAMBDA_REF_VAL_NM = 100
# result directory
RES_DIR = Path("results")
# initial guesses for calibration fit
ALPHA_GUESS = 0.9
D_GUESS = 100
SPACER_GUESS = 10