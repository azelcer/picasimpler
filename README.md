Simpler plugin for Picasso software

Run the following line in a cmd prompt after activating the environment to get the .py file of the GUI:
pyuic6 -o .\src\picasimpler\UI\calibration_ui.py .\src\picasimpler\UI\calibration_ui.ui

The software follows an MVP pattern.

All the heavy-duty logic is performed by 3 classes: AnalysisWorker, SIMPLER and Clusterization (picasimpler.main.analysis).
AnalysisWorker posseses an instance of the other two classes, which has to be instatiated inside __init__, because their respective pyqtSignals have to be instantiated from the main thread, before AnalysisWorker is moved from there.
AnalysisWorker posseses also an instance of two dataclasses: Data and Params, used as containers mainly for dataframes and metadata, respectively.
The analysis follows this basic logic:
1) AnalysisWorker.load_data: loading data from .hdf5 file to AnalysisWorker.data.df_raw (pandas.DataFrame), and saving metadata from .yaml file.
2) AnalysisWorker.filter_kin_orig: filtering picks based on kinetics and saving them to AnalysisWorker.data.df_orig (pandas.DataFrame).
3) SIMPLER.filter_data: filter localizations of AnalysisWorker.data.df_orig based on SIMPLER criteria and save them to SIMPLER.locs (list of arrays, each array contains the coordinates x, y and N of all SIMPLER localizations of an origami). The function SIMPLER.filter_locs_inpick is called in a loop for each origami.
4) Clusterization.pre_clust_denoise: performs an HDBSCAN on every origami to get rid of the noise (meaning, localizations outside one of the main clusters). It then saves localizations into 2 separate lists: Clusterization.locs_clust and Clusterization.locs_noise, that have the same structure as SIMPLER.locs, but contain only localizations in main clusters and noise, respectively. Origamis whose total number of localizations in main clusters falls below a certain threshold are thrown away.
5) Clusterization.do_clust_xyn: performs GMM on Clusterization.locs_clust with a decreasing number of clusters, starting from the expected number. As soon as a fit with a lower number of clusters performs better (based on BIC), it discards the origami. It then updates Clusterization.locs_clust and Clusterization.locs_noise, considering only the origamis that passed the test, and saves the corresponding GMM results to numpy arrays Clusterinzation.clust_means and Clusterization.clust_covs. In these arrays, the first index is the number of the origami and the second one is the number of the cluster (decreasing number of photons, i.e., increasing z); Clusterinzation.clust_means has one index more for the coordinate (x, y, N), while Clusterization.clust_covs has 2.

Conventions:
1) Localizations in dataframes are measured in pixels, while loclaizations in list/arrays should be converted to nm using the function picasimpler.helpers.conversions.px_to_nm.
2) Signals should be stored in a separate class inheriting QObject which has to be instantiated in the main thread.
3) All connections should happen in the Presenter
4) use the decorator @check_analysis_status before functions that depend on the current status of the analysis.

Input parameters from UI:
1) γ: minimum (as proportion of the total number of localizations for an origami) size of a cluster to be considered as such by the pre-clustering de-noising HDBSCAN step. Clusters smaller than this threshold will be considered noise and not used in the next steps. Hence, a higher value correspond to a more stringent de-noising.
2) ε: minimum (absolute) distance between two points to be considered as elements of two different clusters by the pre-clustering de-noising HDBSCAN step. Clusters closer than this threshold value will be merged. Higher values will tend to incorporate more noise localizations into the main clusters found by the algorithm; hence, smaller values correspond to a more stringent de-noising.