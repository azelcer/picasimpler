from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from picasimpler.main.analysis import AnalysisWorker, SpatialFit

r"X:\messdaten\Giovanni_A\SIMPLER\260313\Rifle_4pts_R2_40gain_500pMCy3B_200mW_100ms_23TIRF\R2\R2_2_MMStack_Pos0.ome_locs_picked_standing.hdf5"

res_dir_path = Path(r"C:\Users\giova\Documents\University\Software\picasimpler\results")
filename = r"rifle_R2_250pMCy3B_300mW_100ms_30gain_tirf23_1_MMStack_Pos0.ome_locs_picked_standing_clusters.npy"
clust_res_filepath = res_dir_path / Path(filename)
NA_path = Path(r"src\picasimpler\resources\DF_NA145.txt")

lambda_exc = 560
lambda_em = 570
n_s = 1.333
n_i = 1.516

analysis_worker: AnalysisWorker = AnalysisWorker()

analysis_worker.params.lambda_exc = lambda_exc
analysis_worker.params.lambda_em = lambda_em
analysis_worker.params.n_s = n_s
analysis_worker.params.n_i = n_i
analysis_worker.params.coll_fl_tab = np.loadtxt(NA_path)
analysis_worker.params.should_do_res_analysis = False

analysis_worker.share_params()

d_exc_list = []
alpha_exc_list = []
for alpha_exc in np.arange(0.8, 1.01, 0.01):
    analysis_worker.fit.alpha_fixed = alpha_exc
    analysis_worker.do_calib_fromfile(clust_res_filepath)
    d_exc_list.append(analysis_worker.fit.d_exc)
    alpha_exc_list.append(alpha_exc)
    print(f"d_exc = {analysis_worker.fit.d_exc}")
    print(f"alpha_exc = {analysis_worker.fit.alpha_exc}")
    print(f"theta_TIRF = {analysis_worker.fit.tirf_angle}")
    
    plt.plot(analysis_worker.fit.z_ax_forplot, analysis_worker.fit.fit_func_forplot.ravel())
    plt.ylabel("Relative Intensity")
    plt.xlabel("z")
    plt.grid()
    
plt.plot(analysis_worker.fit.z_real.ravel(), analysis_worker.fit.N_renorm_arr.ravel(), ".", ms=8)
plt.show()
    
    

