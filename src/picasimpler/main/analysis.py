import numpy as np
import pandas as pd
import logging as _lgn
import yaml
import time as _time
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pathlib import Path
from scipy.spatial import distance
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
from sklearn.cluster import HDBSCAN
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
    
from picasimpler.helpers.status import AnalysisStatus, MessageType
from picasimpler.helpers.utils import px_to_nm
from picasimpler.config.config_var import (
    MAX_FIRST_FRAME_PERC,
    MIN_LAST_FRAME_PERC,
    FRAME_MEDIAN_PERC_RANGE,
    MAX_ON_FRAMES_PERC,
    MIN_GOOD_LOC,
    N_CLUST_EXP,
    Z_SITES_NM,
    DF_REF_VAL_NM,
    RES_DIR,
    ALPHA_GUESS,
    D_GUESS,
    CALIB_PLOT_RANGE_NM,
    CALIB_PLOT_PTS,
    Z_SIM_DISCR,
    Z_SIM_FIT_ARR,
    LAMBDA_EM_DISCR
)

_lgn.basicConfig()
_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.INFO)

class SIMPLERSignals(QObject):
    tell_analysis_elem_done = pyqtSignal(int)
    send_msg_toprint = pyqtSignal(object, str)

class SIMPLER:
    """
    Class containing all the SIMPLER methods and results
    """
    def __init__(self, signals: SIMPLERSignals):
        self.signals: SIMPLERSignals = signals
        self.locs: list | None = None
        self.params = Params()
    
    def get_loc_x(self, orig_num):
        return self.locs[orig_num][:, 0]
    
    def get_loc_y(self, orig_num):
        return self.locs[orig_num][:, 1]
    
    def get_loc_n(self, orig_num):
        return self.locs[orig_num][:, 2]
    
    def get_loc_log_n(self, orig_num):
        return np.log10(self.locs[orig_num][:, 2])
    
    def save_locs(self, df: pd.DataFrame, px_size_nm: int):
        """
        This function saves all SIMPLER localizations in a list of arrays (one for each origami)
        """
        all_orig_loc_list = []
        # helper array to find fast all localization pertaining to an origami
        groups = np.array(df['group'])
        groupjump = np.nonzero(np.diff(groups, prepend=-np.inf, append=np.inf) != 0)[0]
        for pick_idx in range(len(groupjump) - 1):
            pick_locs_arr = np.asarray(
                df.iloc[
                    groupjump[pick_idx]:groupjump[pick_idx + 1],
                    df.columns.get_indexer(['x', 'y', 'photons'])
                ], dtype=float
            )
            pick_locs_arr_nm = px_to_nm(pick_locs_arr, px_size_nm)
            all_orig_loc_list.append(pick_locs_arr_nm)
        self.locs = all_orig_loc_list
    
    def filter_locs_inorig(self, df_pick, r_th_sq):
        """
        this function filters all localization inside a pick, keeping only events with at least 3 frames and throwing
        away the first and last frame.
        """
        frames = np.array(df_pick['frame'])
        xy = np.column_stack((np.transpose(df_pick['x']), np.transpose(df_pick['y']))) # array of xy couples for each loc
        discard_yn = np.ones((len(frames),), dtype=np.uint64)
        framejump = np.nonzero(np.diff(frames, prepend=-np.inf, append=np.inf) != 0)[0] # indices (in the dataframe) where frame changes
        distance_next = None
        for idxframe in range(1, len(framejump) - 2):
            prevframe = frames[framejump[idxframe - 1]]
            nextframe = frames[framejump[idxframe + 1]]
            frame = frames[framejump[idxframe]]
            if (frame + 1 != nextframe):
                distance_next = None
                continue
            if (frame - 1 != prevframe):
                continue
            f_slice = slice(framejump[idxframe], framejump[idxframe + 1])
            prev_slice = slice(framejump[idxframe - 1], framejump[idxframe])
            next_slice = slice(framejump[idxframe + 1], framejump[idxframe + 2])
            if distance_next is not None:
                distance_prev = distance_next.T
            else:
                distance_prev = distance.cdist(xy[f_slice], xy[prev_slice], 'sqeuclidean')
            distance_next = distance.cdist(xy[f_slice], xy[next_slice], 'sqeuclidean')
            has_prev = np.any(distance_prev < r_th_sq, axis=1)
            has_next = np.any(distance_next < r_th_sq, axis=1)
            discard_yn[f_slice][np.logical_and(has_prev, has_next)] = 0
        rel_idx_to_discard = np.where(discard_yn == 1)[0]
        abs_idx_to_discard = np.array(df_pick.index)[rel_idx_to_discard]
        return abs_idx_to_discard
            
    def filter_locs(self, df: pd.DataFrame, px_size_nm: int, r_th_sq: float):
        """
        This function filters all data, pick by pick, using SIMPLER criteria
        """
        start = _time.time()
        n_loc_initial = len(df['frame'])
        idx_to_discard = np.array([])
        groups = np.array(df['group'])
        groupjump = np.nonzero(np.diff(groups, prepend=-np.inf, append=np.inf) != 0)[0]
        for pick_idx in range(len(groupjump) - 1):
            idx_to_discard = np.concatenate(
                (idx_to_discard, self.filter_locs_inorig(df.iloc[groupjump[pick_idx]:groupjump[pick_idx + 1]], r_th_sq)),
                axis=0
            )
            self.signals.tell_analysis_elem_done.emit(pick_idx + 1)
        df_filtered = df.drop(labels=idx_to_discard, axis=0)
        df_filtered = df_filtered.reset_index(level=None, drop=True, inplace=False,
                                              col_level=0)
        end = _time.time()
        _lgr.info("Time of filtering step: {0:.2f} s. {1} of {2} ({3:.1f}) localizations discarded".format(
            end - start, len(idx_to_discard), n_loc_initial, 100 * len(idx_to_discard) / n_loc_initial
        ))
        self.signals.send_msg_toprint.emit(MessageType.INFO, "Time of filtering step: {0:.2f} s. {1} of {2} ({3:.1f}%) localizations discarded".format(
            end - start, len(idx_to_discard), n_loc_initial, 100 * len(idx_to_discard) / n_loc_initial
        ))
        self.save_locs(df_filtered, px_size_nm)
    
class ClusterizationSignals(QObject):
    tell_analysis_elem_done = pyqtSignal(int)
    send_msg_toprint = pyqtSignal(object, str)
    
class Clusterization:
    """
    Class containing all the results (means and covariances, and SIMPLER localizations) of the clusters after site clusterization.
    First index is the origami, second index is the cluster (should be alread ordered), last or last two are coordinates.
    Coordinates go from 0 to 2 and are always in this order: x, y and N (number of photons).
    """
    def __init__(self, signals: ClusterizationSignals):
        self.signals: ClusterizationSignals = signals
        self.tot_orig_kept: int = 0
        self.locs_clust: list | None = None
        self.locs_noise: list | None = None
        self.clust_means: np.ndarray | None = None
        self.clust_covs: np.ndarray | None = None
        self.selec_orig_list: list[bool] | None = None
        self.params = Params()

    def get_clust_x(self, orig_num):
        return self.locs_clust[orig_num][:, 0]
    
    def get_clust_y(self, orig_num):
        return self.locs_clust[orig_num][:, 1]
    
    def get_clust_n(self, orig_num):
        return self.locs_clust[orig_num][:, 2]
    
    def get_clust_log_n(self, orig_num):
        return np.log10(self.locs_clust[orig_num][:, 2])
    
    def get_noise_x(self, orig_num):
        return self.locs_noise[orig_num][:, 0]
    
    def get_noise_y(self, orig_num):
        return self.locs_noise[orig_num][:, 1]
    
    def get_noise_n(self, orig_num):
        return self.locs_noise[orig_num][:, 2]
    
    def get_noise_log_n(self, orig_num):
        return np.log10(self.locs_noise[orig_num][:, 2])

    @staticmethod
    def reorder_clust(means, sigmas):
        """
        This method reorders in descending order tuples of means and sigmas based on the mean of the last coordinate
        (number of photons). It is used to order clusters from bottom to top
        """
        return list(zip(*sorted(zip(means, sigmas), key=lambda pair: -pair[0][2])))

    def pre_clust_denoise_inorig(self, locs: list, preclust_gamma: float, preclust_eps: float):
        """
        This function executes pre-clustering de-noising for a single origami
        """
        min_clust_size = int(preclust_gamma*len(locs))
        loc_rescal = np.stack(
            (locs[:, 0],
            locs[:, 1],
            DF_REF_VAL_NM*np.log(locs[:, 2])), axis=1
        )
        hdbsc = HDBSCAN(
            min_cluster_size=np.max((min_clust_size, 2)),
            cluster_selection_epsilon=preclust_eps,
            allow_single_cluster=True
        ).fit(loc_rescal)
        if len(locs[hdbsc.labels_!=-1]) > MIN_GOOD_LOC:
            return hdbsc.labels_
        else:
            return

    def pre_clust_denoise(self, locs: list, preclust_gamma: float, preclust_eps: float):
        """
        This function applies HDBSCAN to separate major clusters (without mecessarily resolving them!) from scattered
        noise and unwanted smaller clusters (such as double events).
        It first rescales the N dimension (using a reference value for the penetration length) to make the clustering
        problem more isotropic.
        """
        start = _time.time()
        self.locs_clust = []
        self.locs_noise = []
        tot_orig_bf_denoise = len(locs)
        for orig_idx in range(tot_orig_bf_denoise):
            labels = self.pre_clust_denoise_inorig(locs[orig_idx], preclust_gamma, preclust_eps)
            if labels is not None:
                self.locs_clust.append(locs[orig_idx][labels!=-1])
                self.locs_noise.append(locs[orig_idx][labels==-1])
            self.signals.tell_analysis_elem_done.emit(orig_idx)
        self.tot_orig_kept = len(self.locs_clust)
        n_orig_discarded = tot_orig_bf_denoise - self.tot_orig_kept
        end = _time.time()
        _lgr.info("Time of pre-clustering de-noising step: {0:.2f} s. {1} of {2} ({3:.1f}%) origamis discarded".format(
            end - start, n_orig_discarded, tot_orig_bf_denoise, 100 * n_orig_discarded / tot_orig_bf_denoise 
        ))
        self.signals.send_msg_toprint.emit(MessageType.INFO, "Time of pre-clustering de-noising step: {0:.2f} s. {1} of {2} ({3:.1f}%) origamis discarded".format(
            end - start, n_orig_discarded, tot_orig_bf_denoise, 100 * n_orig_discarded / tot_orig_bf_denoise
        ))

    def gmm_clust_inorig(self, locs: np.ndarray):
        """
        This function use GMM to cluster data in a single origami
        """
        for n_clust in range(N_CLUST_EXP, 0, -1):
            gmm = GaussianMixture(n_components=n_clust, covariance_type='full', n_init=5, max_iter=300, init_params='k-means++')
            gmm.fit(locs)
            last_bic = gmm.bic(locs)
            if n_clust == N_CLUST_EXP: # compute BIC for the expected number of clusters
                ref_bic = last_bic
                clust_means, clust_covs = self.reorder_clust(gmm.means_, gmm.covariances_)
            # now we decrease the number of clusters and as soon as one gives better result, we discard the origami and exit the loop
            elif last_bic < ref_bic:
                return None, None
        return clust_means, clust_covs


    def do_clust_xyn(self):
        """
        This function loops over all origamis and cluster their data in 3D (x, y, N).
        """
        start = _time.time()
        tot_orig_bf_clust = len(self.locs_clust)
        # here we will store all data relative to the clusterization result
        kept_orig_loc_list = []
        kept_orig_noise_list = []
        clust_means_list = []
        clust_covs_list = []
        for orig_idx in range(tot_orig_bf_clust):
            clust_means, clust_covs = self.gmm_clust_inorig(self.locs_clust[orig_idx])
            if clust_means is not None:
                kept_orig_loc_list.append(self.locs_clust[orig_idx])
                kept_orig_noise_list.append(self.locs_noise[orig_idx])
                clust_means_list.append(clust_means)
                clust_covs_list.append(clust_covs)
            self.signals.tell_analysis_elem_done.emit(orig_idx + 1)
        self.tot_orig_kept = len(kept_orig_loc_list)
        n_orig_discarded = tot_orig_bf_clust - self.tot_orig_kept
        self.locs_clust = kept_orig_loc_list
        self.locs_noise = kept_orig_noise_list
        self.clust_means = np.asarray(clust_means_list, dtype=float)
        self.clust_covs = np.asarray(clust_covs_list, dtype=float)
        self.selec_orig_list = [True]*self.clust_means.shape[0]
        end = _time.time()
        _lgr.info("Time of clustering step: {0:.2f} s. {1} of {2} ({3:.1f}%) origamis discarded".format(
            end - start, n_orig_discarded, tot_orig_bf_clust, 100 * n_orig_discarded / tot_orig_bf_clust  
        ))
        self.signals.send_msg_toprint.emit(MessageType.INFO, "Time of clustering step: {0:.2f} s. {1} of {2} ({3:.1f}%) origamis discarded".format(
            end - start, n_orig_discarded, tot_orig_bf_clust, 100 * n_orig_discarded / tot_orig_bf_clust  
        ))

class SpatialFitSignals(QObject):
    tell_analysis_elem_done = pyqtSignal(int)
    send_msg_toprint = pyqtSignal(object, str)

class SpatialFit(QObject):
    def __init__(self, signals):
        super().__init__()
        self.signals = signals
        self.z_sites_nm = np.array(Z_SITES_NM)
        self.params = Params()
        
    def upd_data_forfit(self, clust_means_forfit):
        """
        This function takes external inputs for the variables needed for the fit (positions in nm of the sites along the origami, and 3D positions of the
        fitted clusters of localizations) and saves them as attributes for later use
        """
        self.clust_means_forfit = clust_means_forfit
        # call functions to update all data needed for fit
        self._calc_tilt_angles()
        self._calc_z_real()
        
    def _calc_tilt_angles(self):
        """
        This function computes tilt angles for all the selected origamis based on xy cluster positions and expected z positions of the sites
        """
        self.tilt_angles = np.array([self._tilts_form_xy(self.z_sites_nm[:-1], o_pos[:-1, 0:2])[0] for o_pos in self.clust_means_forfit])
        
    def _calc_z_real(self):
        """
        This function computes the real expected z positions of the sites, taking into account the tilt angle of each origami
        """
        self.z_real = self._z_from_tilt(self.tilt_angles, self.z_sites_nm)
        
    def _tilts_form_xy(self, origami_positions: np.ndarray, xy_positions: np.ndarray):
        """
        Compute tilt angles of the origami with respect to the surface, based on the
        x and y positions of the clusterized sites.

        Parameters
        ----------
        origami_positions : np.ndarray
            Site positions along the origami in nm. For example, if the origami
            has 3 sites 50 nm apart the first one at 10 nm from the link point,
            it should be [10., 60.0, 110.0]
        xy_positions : np.ndarray
            tuples of the x and y positions of the site clusters of the origami

        Returns
        -------
        theta : float
            angle with the surface in rad (0 means horizontal origami)
        phi : float
            angle with the x axis in rad
        """

        if origami_positions.shape[0] != xy_positions.shape[0]:
            raise ValueError("Length of the z and (x,y) positions of the origami sites do not coincide")

        M = np.column_stack((np.ones(origami_positions.shape[0]), origami_positions))
        coeffs, residuals, rank, s = np.linalg.lstsq(
            M, xy_positions, rcond=None
        )
        A, B = coeffs[1]  # coefs[0] holds (x, y) of the origami at z=0

        phi = np.arctan2(B, A)
        theta = np.arccos(A / np.cos(np.radians(phi)))
        return theta, phi

    def _z_from_tilt(self, theta: float, sites_distances: np.ndarray):
        """Computes z positions from a tilt angle and a set of distances.

        Parameters
        ----------
        theta: float
            Tilt angle in radians. 0 means parallel to substrate, pi/2 means vertical
        sites_distances: np.ndarray
            Distances of each site from the origami link point

        Returns
        -------
        np.ndarray holding the z positions of each site
        """
        return sites_distances * np.sin(theta)[:, np.newaxis]

    def fit_renorm(
        self,
        p0: tuple[float, float] = (ALPHA_GUESS, D_GUESS),
    ):
        """Calculate SIMPLER effective params from known z and N.

        This function fits N instead of z and should be used when N0 of each
        origami is unknown.

        Parameters
        ----------
            p0: tuple[float, float], OPTIONAL
                Initial guesses for alpha_F and d_F

        TODO: add z_0 to account for constant linker added distance.
        """
        
        N_data = self.clust_means_forfit[:, :, 2]

        flat_z = self.z_real[:, 1:].ravel()
        # Normalize datal
        F_data = (N_data / N_data[:, 0, np.newaxis])[:, 1:].ravel()
        z_0 = np.hstack(np.repeat(self.z_real[:, 0], 3))

        # Model function
        def F(z, alpha_F, d_F):
            num = alpha_F * np.exp(-z / d_F) + (1 - alpha_F)
            den = alpha_F * np.exp(-z_0 / d_F) + (1 - alpha_F)
            return num / den
        # Fit
        popt, pcov = curve_fit(F, flat_z, F_data, p0=p0, bounds=([0,0],[1, np.inf]))
        alpha_F, d_F = popt

        perr = np.sqrt(np.diag(pcov))
        self.alpha_F = alpha_F
        self.d_F = d_F
        self.alpha_F_err = perr[0]
        self.d_F_err = perr[1]
        
    def fit_N0(self):
        """
        This function should be called once alpha_F and d_F have already been fitted.
        It iterates over all selected origamis and fits N_0 (keeping alpha_F and d_F fixed!)
        for each one.
        """
        self.N_0_arr = np.zeros(len(self.clust_means_forfit), dtype=float)
        def F(z, N_0):
            return N_0*(self.alpha_F * np.exp(-z / self.d_F) + (1 - self.alpha_F))
        for orig_idx in range(len(self.clust_means_forfit)):
            z_data = self.z_real[orig_idx, :]
            N_data = self.clust_means_forfit[orig_idx, :, 2]
            N_0_val, N_0_err = curve_fit(F, z_data, N_data, p0=self.clust_means_forfit[orig_idx, 0, 2], bounds=([0],[np.inf]))
            self.N_0_arr[orig_idx] = N_0_val
        self.N_renorm_arr = self.clust_means_forfit[:, :, 2]/self.N_0_arr[:, np.newaxis]
        self.N_0_avg = np.mean(self.N_0_arr)
        self.N_0_std = np.std(self.N_0_arr)
        # variables for plot
        self.z_ax_forplot = np.linspace(0, CALIB_PLOT_RANGE_NM, CALIB_PLOT_PTS)
        self.fit_func_forplot = F(self.z_ax_forplot, 1)
        
    def backcalc_tirf_angle(self):
        """
        This function infers, from the global decay curve, the TIRF angle, using information about emission wavelength and objective NA
        """
        self.simpler_prof = self.alpha_F*np.exp(-Z_SIM_FIT_ARR/self.d_F) + (1 - self.alpha_F)
        self.exc_prof = self.simpler_prof / self.params.coll_fl_interp
        def F_exc(x, d_exc, b, c):
            return b * np.exp(-x/d_exc) + c                           
        popt, pcov = curve_fit(F_exc, Z_SIM_FIT_ARR, self.exc_prof, p0 = [200, 0.9, 0.1])
        self.d_exc = popt[0]
        self.d_exc_err = pcov[0, 0]
        self.exc_fit = F_exc(Z_SIM_FIT_ARR, popt[0], popt[1], popt[2])
        self.tirf_angle = np.arcsin(np.sqrt(((self.params.lambda_exc/(4*np.pi*popt[0]))**2 + self.params.n_s**2)/self.params.n_i**2))*180/np.pi


@dataclass
class Params:
    """
    dataclass containing the parameters for the calibration
    """

    # SIMPLER filtering parameters
    spat_tol_nm: float | None = None # how far can two locs be to be considered the same event
    # pre-clustering parameters
    preclust_gamma: float | None = None
    preclust_eps: float | None = None
    # setup parameters
    lambda_exc: float | None = None
    lambda_em: float | None = None
    n_s: float | None = None
    n_i: float | None = None
    # Collection efficiencies
    coll_fl_tab: np.ndarray | None = None
    coll_fl_interp: np.ndarray | None = None
    # movie parameters
    n_frames: int | None = None  # number of frames in movie
    exp_time_ms: float | None = None  # exposure time in ms
    px_size_nm: float | None = None # camera pixel size in nm
    # convenience parameters
    r_th_sq: float | None = None

class AnalysisSignals(QObject):
    # type of analysis step starting now, and total number of element in it
    tell_data_loaded = pyqtSignal()
    tell_analysis_step_start = pyqtSignal(AnalysisStatus, int)
    tell_analysis_elem_done = pyqtSignal(int)
    tell_filtering_done = pyqtSignal()
    tell_filt_done = pyqtSignal()
    tell_clust_done = pyqtSignal(bool)
    tell_refit_done = pyqtSignal()
    send_msg_toprint = pyqtSignal(object, str)
    tell_calib_done = pyqtSignal(str)

class AnalysisWorker(QObject):
    def __init__(self):
        super().__init__()
        # all signals must be instatiated in main thread, so inside __init__
        self.signals = AnalysisSignals()
        self.simpler_signals = SIMPLERSignals()
        self.clust_signals = ClusterizationSignals()
        self.fit_signals = SpatialFitSignals()
        self.simpler: SIMPLER = SIMPLER(self.simpler_signals)
        self.clust: Clusterization = Clusterization(self.clust_signals)
        self.fit: SpatialFit = SpatialFit(self.fit_signals)
        self.params = Params()

    def upd_params(self):
        """
        This function updates the parameters for all the other analysis classes
        """
        self.simpler.params = self.params
        self.clust.params = self.params
        self.fit.params = self.params

    @pyqtSlot(float)
    def upd_spat_tol(self, value):
        self.params.spat_tol_nm = value
        self.upd_params()

    @pyqtSlot(float)
    def upd_preclust_gamma(self, value):
        self.params.preclust_gamma = value
        self.upd_params()
        
    @pyqtSlot(float)
    def upd_preclust_eps(self, value):
        self.params.preclust_eps = value
        self.upd_params()

    @pyqtSlot(float)
    def upd_lambda_exc(self, value):
        self.params.lambda_exc = value
        self.upd_params()
        
    @pyqtSlot(float)
    def upd_lambda_em(self, value):
        self.params.lambda_em = value
        if self.params.coll_fl_tab is not None:
            self._calc_coll_fl_arr()
        self.upd_params()
            
    @pyqtSlot(float)
    def upd_n_i(self, value):
        self.params.n_i = value
        self.upd_params()
        
    @pyqtSlot(float)
    def upd_n_s(self, value):
        self.params.n_s = value
        self.upd_params()
        
    @pyqtSlot(object)
    def upd_coll_fl_tab(self, coll_fl_tab):
        """
        This function receives a table of collection efficiencies, corresponding to the value of NA chosen on UI.
        """
        self.params.coll_fl_tab = coll_fl_tab
        if self.params.lambda_em is not None:
            self._calc_coll_fl_arr()
        self.upd_params()

    def _calc_coll_fl_arr(self):
        """
        This function computes the collection efficiency of the objective depending on the emission wavelength and z.
        It extract the values corresponding to the value of simulated emission lambda which is the closest to the value
        chosen on UI; then, it interpolates such values to the full z axis and passes the result to the fit class.
        """
        idx_closest_lambda_em = np.argmin(abs(LAMBDA_EM_DISCR - np.ones(np.size(LAMBDA_EM_DISCR))*self.params.lambda_em))
        coll_fl_discr = self.params.coll_fl_tab[:, idx_closest_lambda_em]
        if len(coll_fl_discr)!=len(Z_SIM_DISCR):
            raise ValueError("Arrays of simulated z and d_F have different length!")
        self.params.coll_fl_interp = interp1d(Z_SIM_DISCR, coll_fl_discr)(Z_SIM_FIT_ARR)
                
    @pyqtSlot(Path, Path)
    def load_data(self, picks_data_path, metadata_path):
        """
        This function calls other functions to load data and metadata from file
        """
        self.picks_data_path = picks_data_path
        self.metadata_path = metadata_path
        self._load_hdf5_todf()
        if self.is_data_file_open:
            self._load_metadata()
            if self.is_metadata_file_open:
                self.params.r_th_sq = (self.params.spat_tol_nm / self.params.px_size_nm)**2
                self.signals.tell_data_loaded.emit()

    def _load_hdf5_todf(self):
        """
        This function opens the hdf5 containing all the picked structures
        """
        try:
            with pd.HDFStore(self.picks_data_path, 'r') as store:
                hdf5_node_list = [node._v_pathname for node in store._handle.walk_nodes()]
                if '/locs' not in hdf5_node_list:
                    _lgr.error('hdf5 file does not have expected structure')
                    self.signals.send_msg_toprint.emit(MessageType.ERROR, "hdf5 file does not have expected structure")
                    # FIXME: this cleans previous file is lodaded
                    self.df_raw = None
                    self.tot_picks = 0
                    self.is_data_file_open = False
                _lgr.info('hdf5 file has expected structure')
                self.signals.send_msg_toprint.emit(MessageType.INFO, "hdf5 file has expected structure")
                df_data = store['/locs']
                # count total number of picks
                tot_picks = len(set(df_data['group']))
                _lgr.info(f"Total number of picks: {tot_picks}")
                self.signals.send_msg_toprint.emit(MessageType.INFO, f"Total number of picks: {tot_picks}")
                self.df_raw = df_data
                self.tot_picks = tot_picks
                self.is_data_file_open = True
        except Exception as e:
            if isinstance(e, KeyError) and str(e) == "'group'":
                _lgr.error("Error opening hdf5 file: picks were not found in file")
                self.signals.send_msg_toprint.emit(MessageType.ERROR, "Error opening hdf5 file: picks were not found in file")
            else:
                _lgr.error(f"Error {type(e)} opening hdf5 file: {e}")
                self.signals.send_msg_toprint.emit(MessageType.ERROR, f"Error {type(e)} opening hdf5 file: {e}")
            self.df_raw = None
            self.tot_picks = 0
            self.is_data_file_open = False

    def _load_metadata(self):
        """
        this function loads the metadata from the yaml file
        """
        try:
            with open(self.metadata_path, "r") as metadata_file:
                metadata = list(yaml.load_all(metadata_file, Loader=yaml.FullLoader))
                frames = metadata[0]['Frames']
                exp_time_ms = metadata[0]['Micro-Manager Metadata']['Exposure-ms']
                px_size_nm = metadata[1]['Pixelsize']
                _lgr.info(f"Number of frames in movie: {frames}")
                _lgr.info(f"Exposure time in ms: {exp_time_ms}")
                _lgr.info(f"Pixel size in nm: {px_size_nm}")
                self.signals.send_msg_toprint.emit(MessageType.INFO, "Movie metadata readed correctly.")
                self.signals.send_msg_toprint.emit(MessageType.SIMPLE, f"Number of frames in movie: {frames}")
                self.signals.send_msg_toprint.emit(MessageType.SIMPLE, f"Exposure time in ms: {exp_time_ms}")
                self.signals.send_msg_toprint.emit(MessageType.SIMPLE, f"Pixel size in nm: {px_size_nm}")
                self.params.n_frames = frames
                self.params.exp_time_ms = exp_time_ms
                self.params.px_size_nm = px_size_nm
                self.is_metadata_file_open = True
        except Exception as e:
            _lgr.error(f"Error opening yaml file because of: {e}")
            self.signals.send_msg_toprint.emit(MessageType.ERROR, f"Error opening yaml file because of: {e}")
            self.params.n_frames = None
            self.params.exp_time_ms = None
            self.params.px_size_nm = None
            self.is_metadata_file_open = False

    def filter_kin_orig(self):
        """
        this function removes picks not following expected PAINT statistics
        """
        picks_tokeep = []
        groups = np.array(self.df_raw['group'])
        groupjump = np.nonzero(np.diff(groups, prepend=-np.inf, append=np.inf) != 0)[0]
        for pick_idx in range(self.tot_picks):
            pick_df = self.df_raw.iloc[groupjump[pick_idx]:groupjump[pick_idx + 1]]
            first_frame_perc = np.min(pick_df['frame']) / self.params.n_frames
            last_frame_perc = np.max(pick_df['frame']) / self.params.n_frames
            med_frame_perc = np.median(pick_df['frame']) / self.params.n_frames
            unique_frames = set(pick_df['frame'])
            num_on_frames_perc = len(unique_frames) / self.params.n_frames
            # to be considered an origami, the pick has to pass all following kinetics test
            if not (
                (first_frame_perc > MAX_FIRST_FRAME_PERC) or
                (last_frame_perc < MIN_LAST_FRAME_PERC) or
                (med_frame_perc < FRAME_MEDIAN_PERC_RANGE[0]) or
                (med_frame_perc > FRAME_MEDIAN_PERC_RANGE[1]) or
                (num_on_frames_perc > MAX_ON_FRAMES_PERC)
            ):
                picks_tokeep.append(pick_idx)
                self.signals.tell_analysis_elem_done.emit(pick_idx + 1)
        df_orig = self.df_raw.loc[self.df_raw['group'].isin(picks_tokeep)]
        n_orig = len(picks_tokeep)
        _lgr.info(f"{n_orig} picks out of {self.tot_picks} passed the kinetics filter")
        self.signals.send_msg_toprint.emit(MessageType.INFO, f"{n_orig} picks out of {self.tot_picks} passed the kinetics filter")
        self.df_orig = df_orig
        self.tot_orig = n_orig

    @pyqtSlot()
    def do_filt(self):
        """
        this function calls one by one all the filtering steps (filtering based on kinetics and SIMPLER localization filtering)
        """
        self.signals.tell_analysis_step_start.emit(AnalysisStatus.KIN_FILT, self.tot_picks)
        self.filter_kin_orig()
        self.signals.tell_analysis_step_start.emit(AnalysisStatus.SIMPLER_FILT, self.tot_orig)
        self.simpler.filter_locs(self.df_orig, self.params.px_size_nm, self.params.r_th_sq)
        self.signals.tell_filt_done.emit()

    @pyqtSlot()
    def do_clust(self):
        """
        This function call the clusterization function
        """
        self.signals.tell_analysis_step_start.emit(AnalysisStatus.PRE_CLUST, self.tot_orig)
        self.clust.pre_clust_denoise(
            self.simpler.locs,
            self.params.preclust_gamma,
            self.params.preclust_eps,
        )
        self.signals.tell_analysis_step_start.emit(AnalysisStatus.SITE_CLUST, self.clust.tot_orig_kept)
        self.clust.do_clust_xyn()
        if self.simpler.locs:
            self.signals.tell_clust_done.emit(True)
        else:
            self.signals.tell_clust_done.emit(False)

    def save_clust(self):
        """
        This function saves the array of clusterization results of the selected origamis only as a .npy
        """
        clust_means_res_filename = self.picks_data_path.stem + "_clusters.npy"
        clust_covs_res_filename = self.picks_data_path.stem + "_covs.npy"
        np.save(RES_DIR / Path(clust_means_res_filename), self.clust.clust_means[self.clust.selec_orig_list,:,:])
        np.save(RES_DIR / Path(clust_covs_res_filename), self.clust.clust_covs[self.clust.selec_orig_list,:,:,:])
        
    @pyqtSlot(int)
    def refit_orig(self, orig_num: int):
        """
        This function re-fits (both pre-clustering de-noising and GMM clustering) the currently displayed origami
        """
        locs_unlabel = np.concatenate((self.clust.locs_clust[orig_num], self.clust.locs_noise[orig_num]))
        new_labels = self.clust.pre_clust_denoise_inorig(
            locs_unlabel,
            self.params.preclust_gamma,
            self.params.preclust_eps
        )
        if new_labels is None:
            _lgr.warning("Re-fit failed at pre-clustering de-noising step, try changing parameters")
            self.signals.send_msg_toprint.emit(MessageType.WARNING, "Re-fit failed at pre-clustering de-noising step, try changing parameters")
            return
        else:
            new_means, new_covs = self.clust.gmm_clust_inorig(locs_unlabel[new_labels!=-1])
            if new_means is None:
                _lgr.warning("Re-fit failed at GMM clustering step, try changing parameters")
                self.signals.send_msg_toprint.emit(MessageType.WARNING, "Re-fit failed at GMM clustering step, try changing parameters")
            else:
                # if new fit passed all steps, update old results with new
                self.clust.locs_clust[orig_num] = locs_unlabel[new_labels!=-1]
                self.clust.locs_noise[orig_num] = locs_unlabel[new_labels==-1]
                self.clust.clust_means[orig_num, :, :] = new_means
                self.clust.clust_covs[orig_num, :, :, :] = new_covs
                self.signals.tell_refit_done.emit()
        
    @pyqtSlot()
    def do_calib(self):
        """
        This function performs the SIMPLER calibration using the results from clusterization and the expected
        z positions, corrected according to the origamin tilt. 
        """
        self.perform_calib_steps(self.clust.clust_means[self.clust.selec_orig_list, :, :])
        self.signals.tell_calib_done.emit('')
        
    @pyqtSlot(Path)
    def do_calib_fromfile(self, res_path):
        """
        This function performs the SIMPLER calibration using the results from a previous clusterization saved on file, and the expected
        z positions, corrected according to the origamin tilt. 
        """
        try:
            clust_fromfile = np.load(res_path)
        except Exception as e:
            self.signals.send_msg_toprint.emit(MessageType.ERROR, f"Cannot open result file because of Exception: {e}")
        if (clust_fromfile.dtype==float) and (clust_fromfile.shape[1:3]==(4, 3)) and (len(clust_fromfile.shape)==3):
            self.perform_calib_steps(clust_fromfile)
            self.signals.tell_calib_done.emit('from file')
        else:
            self.signals.send_msg_toprint.emit(MessageType.ERROR, "Result file does not have expected structure or content")

    def perform_calib_steps(self, clust_forcalib):
        self.fit.upd_data_forfit(clust_forcalib)
        self.fit.fit_renorm()
        self.fit.fit_N0()
        self.fit.backcalc_tirf_angle()

def plot_origami_fit(z_values: np.ndarray, N_values: np.ndarray, alpha_F: float, d_F: float):
    y_values = (alpha_F * np.exp(-z_values / d_F) + (1 - alpha_F)) / (alpha_F * np.exp(-z_values[:, 0, np.newaxis] / d_F) + (1 - alpha_F))
    F_values = N_values / N_values[:, 0, np.newaxis]
    plt.plot(z_values.ravel(), F_values.ravel(), ".", ms=8, label="Data")
    plt.plot(z_values.ravel(), y_values.ravel(), "x", label="Fit")
    plt.ylabel(r"$F(z) = \frac{N(z)}{N(z_1)}$")
    plt.xlabel("z")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    clus = Clusterization(None)
    # clust_means [#origami, # site, (x, y, N)]
    clus.clust_means = np.load("../../../results/R2_2_MMStack_Pos0.ome_locs_picked_standing_clusters.npy")
    clus.clust_covs = np.load("../../../results/R2_2_MMStack_Pos0.ome_locs_picked_standing_covs.npy")
    positions = np.array(Z_SITES_NM)
    angles = np.array([clus.tilts_form_xy(positions, o_pos[:, 0:2])[0] for o_pos in clus.clust_means])
    z = clus.z_from_tilt(angles, positions)
    alpha_F, d_F, errors = clus.fit_N(z, clus.clust_means[:, :, 2])
    print(alpha_F, d_F, errors)
    plot_origami_fit(z, clus.clust_means[:, :, 2], alpha_F, d_F)


if __name__ == "__main__X":
    filepath_str = r"X:\messdaten\Giovanni_A\SIMPLER\260313\Rifle_4pts_R2_40gain_500pMCy3B_200mW_100ms_23TIRF\R2\R2_2_MMStack_Pos0.ome_locs_picked_standing.hdf5"
    data_path = Path(filepath_str)
    metadata_path = data_path.parent / Path(data_path.stem + ".yaml")
    analysis_worker = AnalysisWorker(data_path, metadata_path)
    analysis_worker.load_data()
    analysis_worker.do_filt()
    analysis_worker.do_clust()
