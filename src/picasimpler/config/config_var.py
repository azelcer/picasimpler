from pathlib import Path

# SIMPLER filtering
SPAT_TOL_NM = 30.0
# kinetics filtering
MAX_FIRST_FRAME_PERC = 0.33
MIN_LAST_FRAME_PERC = 0.66
FRAME_MEDIAN_PERC_RANGE = [0.33, 0.66]
MAX_ON_FRAMES_PERC = 0.8
# clustering
MIN_PERC_LOC_INCLUST = 0.1
MIN_GOOD_LOC = 100
N_CLUST_EXP = 4
# origami structure
Z_SITES_NM = [9.3, 43.7, 79.9, 116.3]
# result directory
RES_DIR = Path("results")