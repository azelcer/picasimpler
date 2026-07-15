import numpy as np
import pandas as pd
import logging as _lgn
import yaml
import json
import time as _time
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pathlib import Path
from scipy.spatial import distance
from scipy.optimize import curve_fit, fsolve
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
from sklearn.cluster import HDBSCAN
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
    
from picasimpler.helpers.status import AnalysisStatus, MessageType
from picasimpler.helpers.utils import px_to_nm
from picasimpler.helpers.cf_calc import SimulQ
from picasimpler.config.config_var import (
    MAX_FIRST_FRAME_PERC,
    MIN_LAST_FRAME_PERC,
    FRAME_MEDIAN_PERC_RANGE,
    MAX_ON_FRAMES_PERC,
    MIN_GOOD_LOC,
    RIFLE_N_CLUST_EXP,
    HB_N_CLUST_EXP,
    Z_BASELINE_NM,
    RIFLE_Z_SITES_NM,
    HB_Z_SITES_NM,
    DF_REF_VAL_NM,
    RES_DIR,
    ALPHA_GUESS,
    ALPHA_MAX,
    ALPHA_FIXED,
    D_GUESS,
    SPACER_GUESS,
    D_LONG_GUESS,
    CALIB_PLOT_RANGE_NM,
    CALIB_PLOT_PTS,
    Z_SIM_DISCR,
    Z_SIM_FIT_ARR,
    LAMBDA_EM_DISCR,
    CALIB_MODE
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
    Class containing all the results (cluster means and covariances, and SIMPLER localizations) of the clusters after site clusterization.
    As for localizations, the list index is the origami, then the first array index is the localization and the second one is the coordinate.
    As for clusters, first index is the origami, second index is the cluster (should be alread ordered), last or last two are coordinates.
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
        self.tilt_angles: np.ndarray | None = None
        self.z_real: np.ndarray | None = None
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

    @staticmethod
    def reorder_clust_idx(means):
        """
        This method gives the permutation of indices used to reorder in descending order the means based on the mean of the last coordinate
        (number of photons). It is used to order clusters from bottom to top
        """
        return np.argsort(-means[:, 2])

    def upd_clust_fromfile(self, clust_fromfile, clust_locs_fromfile, clust_labels_fromfile):
        """
        This function updates the clusterization result variable using data from file
        """
        self.clust_means = clust_fromfile
        self.locs_clust = clust_locs_fromfile
        self.clust_labels = clust_labels_fromfile

    def pre_clust_denoise_inorig(self, locs: list):
        """
        This function executes pre-clustering de-noising for a single origami
        """
        if len(locs) < MIN_GOOD_LOC:
            return
        min_clust_size = int(self.params.preclust_gamma*len(locs))
        loc_rescal = np.stack(
            (locs[:, 0],
            locs[:, 1],
            DF_REF_VAL_NM*np.log(locs[:, 2])), axis=1
        )
        hdbsc = HDBSCAN(
            min_cluster_size=np.max((min_clust_size, 2)),
            cluster_selection_epsilon=self.params.preclust_eps,
            allow_single_cluster=True
        ).fit(loc_rescal)
        labels = hdbsc.labels_
        if self.params.should_use_n_bounds and all(bound is not None for bound in self.params.n_bounds):
            labels = np.where(
                np.logical_and(locs[:, 2] > np.min((self.params.n_bounds[0], self.params.n_bounds[1])),
                            locs[:, 2] < np.max((self.params.n_bounds[0], self.params.n_bounds[1]))
                ),
                labels,
                -1
            )
        if self.params.should_use_x_bounds and all(bound is not None for bound in self.params.x_bounds):
            labels = np.where(
                np.logical_and(locs[:, 0] > np.min((self.params.x_bounds[0], self.params.x_bounds[1])),
                            locs[:, 0] < np.max((self.params.x_bounds[0], self.params.x_bounds[1]))
                ),
                labels,
                -1
            )
        if self.params.should_use_y_bounds and all(bound is not None for bound in self.params.y_bounds):
            labels = np.where(
                np.logical_and(locs[:, 1] > np.min((self.params.y_bounds[0], self.params.y_bounds[1])),
                            locs[:, 1] < np.max((self.params.y_bounds[0], self.params.y_bounds[1]))
                ),
                labels,
                -1
            )
        if len(locs[labels!=-1]) > MIN_GOOD_LOC:
            return labels
        else:
            return

    def pre_clust_denoise(self, locs: list):
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
            labels = self.pre_clust_denoise_inorig(locs[orig_idx])
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
        for n_clust in range(self.params.n_clust_exp, 0, -1):
            if n_clust == self.params.n_clust_exp:
                if self.params.should_use_n_guess and all(guess is not None for guess in self.params.n_guess):
                    gmm_guess_arr = np.zeros((self.params.n_clust_exp, 3), dtype=float)
                    for clust_idx in range(self.params.n_clust_exp):
                        locs_close_ton = locs[np.logical_and(
                            locs[:, 2] > self.params.n_guess[clust_idx] - 2*np.sqrt(self.params.n_guess[clust_idx]),
                            locs[:, 2] < self.params.n_guess[clust_idx] + 2*np.sqrt(self.params.n_guess[clust_idx])
                        )]
                        if len(locs_close_ton) == 0:
                            return None, None, None
                        gmm_guess_arr[clust_idx, 0:2] = np.mean(locs_close_ton[:, 0:2], axis=0)
                        gmm_guess_arr[clust_idx, 2] = self.params.n_guess[clust_idx]
                        gmm = GaussianMixture(n_components=n_clust, covariance_type='full', n_init=5, max_iter=300, means_init=gmm_guess_arr)
                else:
                    gmm = GaussianMixture(n_components=n_clust, covariance_type='full', n_init=5, max_iter=300, init_params='k-means++')
                labels = gmm.fit_predict(locs)
                ref_bic = gmm.bic(locs)
                permut_idx = self.reorder_clust_idx(gmm.means_)
                clust_means = gmm.means_[permut_idx]
                clust_covs = gmm.covariances_[permut_idx]
                labels = np.array([np.where(permut_idx==lab)[0][0] for lab in labels])
                #clust_means, clust_covs = self.reorder_clust(gmm.means_, gmm.covariances_)
            # now we decrease the number of clusters and as soon as one gives better result, we discard the origami and exit the loop
            else:
                gmm = GaussianMixture(n_components=n_clust, covariance_type='full', n_init=5, max_iter=300, init_params='k-means++')
                gmm.fit(locs)
                last_bic = gmm.bic(locs)
                if last_bic < ref_bic:
                    return None, None, None
        return clust_means, clust_covs, labels
    
    def gmm_clust_inorig_2D(self, locs: np.ndarray):
        """
        This function use GMM to cluster data in a single origami, considering only x and y coordinates
        """
        for n_clust in range(self.params.n_clust_exp, 0, -1):
            if n_clust == self.params.n_clust_exp:
                gmm = GaussianMixture(n_components=n_clust, covariance_type='full', n_init=5, max_iter=300, init_params='k-means++')
                labels = gmm.fit_predict(locs[:, 0:2])
                ref_bic = gmm.bic(locs[:, 0:2])
                clust_means = np.concatenate((gmm.means_, (np.mean(locs[:, 2])*np.ones(self.params.n_clust_exp)).reshape(-1, 1)), axis=1)
                clust_covs = np.concatenate((np.concatenate((gmm.covariances_, np.zeros((self.params.n_clust_exp, 1, 2))), axis=1), np.zeros((self.params.n_clust_exp, 3, 1))), axis=2)
                #clust_means, clust_covs = self.reorder_clust(gmm.means_, gmm.covariances_)
            # now we decrease the number of clusters and as soon as one gives better result, we discard the origami and exit the loop
            else:
                gmm = GaussianMixture(n_components=n_clust, covariance_type='full', n_init=5, max_iter=300, init_params='k-means++')
                gmm.fit(locs[:, 0:2])
                last_bic = gmm.bic(locs[:, 0:2])
                if last_bic < ref_bic:
                    return None, None, None
        return clust_means, clust_covs, labels

    def do_clust_xyn(self):
        """
        This function loops over all origamis and cluster their data in 3D (x, y, N).
        """
        start = _time.time()
        tot_orig_bf_clust = len(self.locs_clust)
        # here we will store all data relative to the clusterization result
        kept_orig_loc_list = []
        kept_orig_clust_label_list = []
        kept_orig_noise_list = []
        clust_means_list = []
        clust_covs_list = []
        for orig_idx in range(tot_orig_bf_clust):
            if self.params.orientation=='vertical':
                clust_means, clust_covs, clust_labels = self.gmm_clust_inorig(self.locs_clust[orig_idx])
            elif self.params.orientation=='horizontal':
                clust_means, clust_covs, clust_labels = self.gmm_clust_inorig_2D(self.locs_clust[orig_idx])
            if clust_means is not None:
                kept_orig_loc_list.append(self.locs_clust[orig_idx])
                kept_orig_clust_label_list.append(clust_labels)
                kept_orig_noise_list.append(self.locs_noise[orig_idx])
                clust_means_list.append(clust_means)
                clust_covs_list.append(clust_covs)
            self.signals.tell_analysis_elem_done.emit(orig_idx + 1)
        self.tot_orig_kept = len(kept_orig_loc_list)
        n_orig_discarded = tot_orig_bf_clust - self.tot_orig_kept
        self.locs_clust = kept_orig_loc_list
        self.clust_labels = kept_orig_clust_label_list
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

    def calc_tilt_angles(self):
        """
        This function computes tilt angles for all the selected origamis based on xy cluster positions and expected z positions of the sites
        """
        self.tilt_angles = np.array([self._tilts_form_xy(self.params.z_nm_arr, o_pos[:, 0:2])[0] for o_pos in self.clust_means])
        
    def calc_z_real(self):
        """
        This function computes the real expected z positions of the sites, taking into account the tilt angle of each origami
        """
        self.z_real = self._z_from_tilt(self.tilt_angles, self.params.z_nm_arr)
        
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
        theta = np.arccos(np.min((1, np.sqrt(A**2 + B**2))))
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
        return sites_distances * np.sin(theta)[:, np.newaxis] + Z_BASELINE_NM

class SpatialFitSignals(QObject):
    tell_analysis_elem_done = pyqtSignal(int)
    send_msg_toprint = pyqtSignal(object, str)

class SpatialFit(QObject):
    def __init__(self, signals):
        super().__init__()
        self.signals = signals
        self.params = Params()
        self.alpha_max = ALPHA_MAX
        self.alpha_fixed = ALPHA_FIXED
        
    def upd_data_forfit(self, clust_means_forfit, clust_locs, clust_labels, tilt_angles, z_real):
        """
        This function takes external inputs for the variables needed for the fit and the resolution analysis (3D positions of the fitted clusters of localizations,
        the raw localizations and their clusterization labels, tilt angles and real z) and saves them as attributes for later use
        """
        self.clust_means = clust_means_forfit
        self.clust_locs = clust_locs
        self.clust_labels = clust_labels
        self.tilt_angles = tilt_angles
        self.z_real = z_real

    def _calc_coll_fl_arr_fromtable(self):
        """
        This function computes the collection efficiency of the objective depending on the emission wavelength and z.
        It extract the values corresponding to the value of simulated emission lambda which is the closest to the value
        chosen on UI; then, it interpolates such values to the full z axis and passes the result to the fit class.
        """
        idx_closest_lambda_em = np.argmin(abs(LAMBDA_EM_DISCR - np.ones(np.size(LAMBDA_EM_DISCR))*self.params.lambda_em))
        self.coll_fl_discr = self.params.coll_fl_tab[:, idx_closest_lambda_em]
        if len(self.coll_fl_discr)!=len(Z_SIM_DISCR):
            raise ValueError("Arrays of simulated z and d_F have different length!")
        self.coll_fl_interp = interp1d(Z_SIM_DISCR, self.coll_fl_discr)
        self.params.coll_fl_interp_grid = self.coll_fl_interp(Z_SIM_FIT_ARR)
        
    def _calc_coll_fl_arr_axelrod(self):
        """
        This function computes the collection efficiency of the objective depending on the emission wavelength and z.
        It simulates the collected fluorescence for the actual value of NA, ni and ns, and lambda of emission.
        It follows the theory from the Axelrod paper from 1987.
        """
        self.simul_q = SimulQ(
            self.params.lambda_em,
            self.params.n_s,
            self.params.n_i,
            self.params.na
        )
        self.coll_fl_interp = self.simul_q.calc_q()
        self.params.coll_fl_interp_grid = self.coll_fl_interp(Z_SIM_FIT_ARR)

    def fit_renorm_exp_appr(
        self,
        p0: tuple[float, float] = (ALPHA_GUESS, D_GUESS),
    ):
        """Calculate SIMPLER effective params from known z and N.

        This function fits N instead of z and should be used when N0 of each
        origami is unknown.

        In this case, the fitting function is an exponential + constant (which is an approximation)

        Parameters
        ----------
            p0: tuple[float, float], OPTIONAL
                Initial guesses for alpha_F and d_F

        TODO: add z_0 to account for constant linker added distance.
        """
        
        N_data = self.clust_means[:, :, 2]

        flat_z = self.z_real[:, 1:].ravel()
        # Normalize datal
        F_data = (N_data / N_data[:, 0, np.newaxis])[:, 1:].ravel()
        z_0 = np.hstack(np.repeat(self.z_real[:, 0], self.params.n_clust_exp - 1))

        # Model function
        def F(z, alpha_F, d_F):
            num = alpha_F * np.exp(-z / d_F) + (1 - alpha_F)
            den = alpha_F * np.exp(-z_0 / d_F) + (1 - alpha_F)
            return num / den
        # Fit
        popt, pcov = curve_fit(F, flat_z, F_data, p0=p0, bounds=([0,0],[self.alpha_max, np.inf]))
        alpha_F, d_F = popt

        perr = np.sqrt(np.diag(pcov))
        self.alpha_F = alpha_F
        self.d_F = d_F
        self.alpha_F_err = perr[0]
        self.d_F_err = perr[1]
        
    def fit_renorm_no_appr(
        self,
        p0: tuple[float, float] = (ALPHA_GUESS, D_GUESS),
    ):
        """Calculate SIMPLER effective params from known z and N.

        This function fits N instead of z and should be used when N0 of each
        origami is unknown.

        In this case, the fitting function is an exponential + constant multiplied by the CF

        Parameters
        ----------
            p0: tuple[float, float], OPTIONAL
                Initial guesses for alpha_F and d_F

        TODO: add z_0 to account for constant linker added distance.
        """
        self._calc_coll_fl_arr_axelrod()
        N_data = self.clust_means[:, :, 2]
        flat_z = self.z_real[:, 1:].ravel()
        # Normalize datal
        F_data = (N_data / N_data[:, 0, np.newaxis])[:, 1:].ravel()
        z_0 = np.hstack(np.repeat(self.z_real[:, 0], self.params.n_clust_exp - 1))
        # Model function
        def F(z, alpha_exc, d_exc):
            num = (alpha_exc * np.exp(-(z) / d_exc) + (1 - alpha_exc))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z)
            den = (alpha_exc * np.exp(-(z_0) / d_exc) + (1 - alpha_exc))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z_0)
            return num / den
        # Fit
        popt, pcov = curve_fit(F, flat_z, F_data, p0=p0, bounds=([0,0],[self.alpha_max, np.inf]))
        alpha_exc, d_exc = popt

        perr = np.sqrt(np.diag(pcov))
        self.alpha_exc = alpha_exc
        self.d_exc = d_exc
        self.alpha_exc_err = perr[0]
        self.d_exc_err = perr[1]
        
        tirf_angle_lambda_factor = self.params.lambda_exc / (4*np.pi)
        tirf_angle_sqrt_factor = np.sqrt(((tirf_angle_lambda_factor / self.d_exc)**2 + self.params.n_s**2))
        tirf_angle_sin = tirf_angle_sqrt_factor / self.params.n_i
        tirf_angle_deriv = (1 / np.sqrt(1 - tirf_angle_sin**2)) * (1 / self.params.n_i) * (1/2) * (1 / tirf_angle_sqrt_factor) * tirf_angle_lambda_factor**2 * (2/self.d_exc**3) * (180/np.pi)
        self.tirf_angle = np.arcsin(tirf_angle_sin)*180/np.pi
        self.tirf_angle_err = np.abs(tirf_angle_deriv) * self.d_exc_err
        
    def fit_renorm_no_appr_spacer(
        self,
        p0: tuple[float, float] = (ALPHA_GUESS, D_GUESS, SPACER_GUESS),
    ):
        """Calculate SIMPLER effective params from known z and N.

        This function fits N instead of z and should be used when N0 of each
        origami is unknown.

        In this case, the fitting function is an exponential + constant multiplied by the CF

        Parameters
        ----------
            p0: tuple[float, float], OPTIONAL
                Initial guesses for alpha_F and d_F

        TODO: add z_0 to account for constant linker added distance.
        """
        self._calc_coll_fl_arr_axelrod()
        N_data = self.clust_means[:, :, 2]
        flat_z = self.z_real[:, 1:].ravel()
        # Normalize datal
        F_data = (N_data / N_data[:, 0, np.newaxis])[:, 1:].ravel()
        z_0 = np.hstack(np.repeat(self.z_real[:, 0], self.params.n_clust_exp - 1))
        # Model function
        def F(z, alpha_exc, d_exc, spacer):
            num = (alpha_exc * np.exp(-(z + spacer) / d_exc) + (1 - alpha_exc))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z + spacer)
            den = (alpha_exc * np.exp(-(z_0 + spacer) / d_exc) + (1 - alpha_exc))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z_0 + spacer)
            return num / den
        # Fit
        popt, pcov = curve_fit(F, flat_z, F_data, p0=p0, bounds=([0,0,0],[self.alpha_max, np.inf, 0.1]))
        alpha_exc, d_exc, spacer = popt

        perr = np.sqrt(np.diag(pcov))
        self.alpha_exc = alpha_exc
        self.d_exc = d_exc
        self.alpha_exc_err = perr[0]
        self.d_exc_err = perr[1]
        
        print(spacer)
        
        tirf_angle_lambda_factor = self.params.lambda_exc / (4*np.pi)
        tirf_angle_sqrt_factor = np.sqrt(((tirf_angle_lambda_factor / self.d_exc)**2 + self.params.n_s**2))
        tirf_angle_sin = tirf_angle_sqrt_factor / self.params.n_i
        tirf_angle_deriv = (1 / np.sqrt(1 - tirf_angle_sin**2)) * (1 / self.params.n_i) * (1/2) * (1 / tirf_angle_sqrt_factor) * tirf_angle_lambda_factor**2 * (2/self.d_exc**3) * (180/np.pi)
        self.tirf_angle = np.arcsin(tirf_angle_sin)*180/np.pi
        self.tirf_angle_err = np.abs(tirf_angle_deriv) * self.d_exc_err
        
    def fit_renorm_no_appr_fix_angle(
        self,
        p0: tuple[float] = (ALPHA_GUESS,),
    ):
        d_exc = self.params.lambda_exc/(4*np.pi)/np.sqrt(self.params.n_i**2*np.sin(np.radians(self.params.tirf_angle))**2 - self.params.n_s**2)
        self._calc_coll_fl_arr_axelrod()
        N_data = self.clust_means[:, :, 2]
        flat_z = self.z_real[:, 1:].ravel()
        # Normalize datal
        F_data = (N_data / N_data[:, 0, np.newaxis])[:, 1:].ravel()
        z_0 = np.hstack(np.repeat(self.z_real[:, 0], self.params.n_clust_exp - 1))
        # Model function
        def F(z, alpha):
            num = (alpha * np.exp(-z / d_exc) + (1 - alpha))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z)
            den = (alpha * np.exp(-z_0 / d_exc) + (1 - alpha))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z_0)
            return num / den
        # Fit
        popt, pcov = curve_fit(F, flat_z, F_data, p0=p0, bounds=([0],[ALPHA_MAX]))
        alpha_exc = popt[0]

        perr = np.sqrt(np.diag(pcov))
        self.alpha_exc = alpha_exc
        self.d_exc = d_exc
        self.alpha_exc_err = perr[0]
        self.d_exc_err = 0

        self.tirf_angle = self.params.tirf_angle
        self.tirf_angle_err = 0 
        
    def fit_no_appr_fix_angle_each_orig(
        self
    ):
        d_exc = self.params.lambda_exc/(4*np.pi)/np.sqrt(self.params.n_i**2*np.sin(np.radians(self.params.tirf_angle))**2 - self.params.n_s**2)
        self._calc_coll_fl_arr_axelrod()
        # Model function
        def N(z, alpha, N0):
            return N0*(alpha*np.exp(-z/d_exc) + (1 - alpha))*self.coll_fl_interp(z)/self.coll_fl_interp(0)
        # Fit
        self.alpha_arr = np.zeros(len(self.clust_means))
        self.N_0_arr = np.zeros(len(self.clust_means))
        for orig_idx in range(len(self.clust_means)):
            N_data = self.clust_means[orig_idx, :, 2].ravel()
            flat_z = self.z_real[orig_idx, :].ravel()
            p0 = (ALPHA_GUESS, N_data[0])
            popt, pcov = curve_fit(N, flat_z, N_data, p0=p0, bounds=([0, 0],[ALPHA_MAX, np.inf]))
            self.alpha_arr[orig_idx] = popt[0]
            self.N_0_arr[orig_idx] = popt[1]

        self.alpha_exc = np.mean(self.alpha_arr)
        self.alpha_exc_err = np.std(self.alpha_arr)
        self.d_exc = d_exc
        self.d_exc_err = 0

        self.tirf_angle = self.params.tirf_angle
        self.tirf_angle_err = 0
        
        self.N_renorm_arr = self.clust_means[:, :, 2]/self.N_0_arr[:, np.newaxis]
        self.N_0_avg = np.mean(self.N_0_arr)
        self.N_0_std = np.std(self.N_0_arr)
        # variables for plot
        self.z_ax_forplot = np.linspace(0, CALIB_PLOT_RANGE_NM, CALIB_PLOT_PTS)
        self.fit_func_forplot = N(self.z_ax_forplot, self.alpha_exc, 1)
        
    def fit_renorm_no_appr_fix_angle_biexp(
        self,
        p0: tuple[float] = (ALPHA_GUESS, D_LONG_GUESS),
    ):
        d_exc = self.params.lambda_exc/(4*np.pi)/np.sqrt(self.params.n_i**2*np.sin(np.radians(self.params.tirf_angle))**2 - self.params.n_s**2)
        self._calc_coll_fl_arr_axelrod()
        N_data = self.clust_means[:, :, 2]
        flat_z = self.z_real[:, 1:].ravel()
        # Normalize datal
        F_data = (N_data / N_data[:, 0, np.newaxis])[:, 1:].ravel()
        z_0 = np.hstack(np.repeat(self.z_real[:, 0], self.params.n_clust_exp - 1))
        # Model function
        def F(z, alpha, d_long):
            num = (alpha * np.exp(-z / d_exc) + (1 - alpha) * np.exp(-z / d_long))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z)
            den = (alpha * np.exp(-z_0 / d_exc) + (1 - alpha) * np.exp(-z / d_long))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z_0)
            return num / den
        # Fit
        popt, pcov = curve_fit(F, flat_z, F_data, p0=p0, bounds=([0, d_exc],[ALPHA_MAX, np.inf]))

        perr = np.sqrt(np.diag(pcov))
        self.alpha_exc = popt[0]
        self.d_exc = d_exc
        self.d_long = popt[1]
        self.d_long_err = perr[1]
        self.alpha_exc_err = perr[0]
        self.d_exc_err = 0

        print(self.d_long)

        self.tirf_angle = self.params.tirf_angle
        self.tirf_angle_err = 0 
        
    def fit_no_appr_fix_angle_biexp_each_orig(
        self
    ):
        d_exc = self.params.lambda_exc/(4*np.pi)/np.sqrt(self.params.n_i**2*np.sin(np.radians(self.params.tirf_angle))**2 - self.params.n_s**2)
        self._calc_coll_fl_arr_axelrod()
        # Model function
        def N(z, alpha, N0, d_long):
            return N0*(alpha*np.exp(-z/d_exc) + (1 - alpha)*np.exp(-z/d_long))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z)/interp1d(Z_SIM_DISCR, self.coll_fl_discr)(0)
        # Fit
        self.alpha_arr = np.zeros(len(self.clust_means))
        self.N_0_arr = np.zeros(len(self.clust_means))
        self.d_long_arr = np.zeros(len(self.clust_means))
        for orig_idx in range(len(self.clust_means)):
            N_data = self.clust_means[orig_idx, :, 2].ravel()
            flat_z = self.z_real[orig_idx, :].ravel()
            p0 = (ALPHA_GUESS, flat_z[0], D_LONG_GUESS)
            popt, pcov = curve_fit(N, flat_z, N_data, p0=p0, bounds=([0, 0, d_exc],[ALPHA_MAX, np.inf, 2000]))
            self.alpha_arr[orig_idx] = popt[0]
            self.N_0_arr[orig_idx] = popt[1]
            self.d_long_arr[orig_idx] = popt[2]
            

        self.alpha_exc = np.mean(self.alpha_arr)
        self.alpha_exc_err = np.std(self.alpha_arr)
        self.d_exc = d_exc
        self.d_exc_err = 0

        self.tirf_angle = self.params.tirf_angle
        self.tirf_angle_err = 0
        
        self.N_renorm_arr = self.clust_means[:, :, 2]/self.N_0_arr[:, np.newaxis]
        self.N_0_avg = np.mean(self.N_0_arr)
        self.N_0_std = np.std(self.N_0_arr)
        
        self.d_long = np.mean(self.d_long_arr)
        self.d_long_err = np.std(self.d_long_arr)
        print(self.d_long)
        # variables for plot
        self.z_ax_forplot = np.linspace(0, CALIB_PLOT_RANGE_NM, CALIB_PLOT_PTS)
        self.fit_func_forplot = N(self.z_ax_forplot, self.alpha_exc, 1, self.d_long)
        
    def fit_renorm_no_appr_fix_alpha(
        self,
        p0: tuple[float] = (D_GUESS,),
    ):
        """Calculate SIMPLER effective params from known z and N.

        This function fits N instead of z and should be used when N0 of each
        origami is unknown.

        In this case, the fitting function is an exponential + constant multiplied by the CF,
        and alpha is fixed

        Parameters
        ----------
            p0: tuple[float, float], OPTIONAL
                Initial guesses for alpha_F and d_F

        TODO: add z_0 to account for constant linker added distance.
        """
        self._calc_coll_fl_arr_axelrod()
        N_data = self.clust_means[:, :, 2]
        flat_z = self.z_real[:, 1:].ravel()
        # Normalize datal
        F_data = (N_data / N_data[:, 0, np.newaxis])[:, 1:].ravel()
        z_0 = np.hstack(np.repeat(self.z_real[:, 0], self.params.n_clust_exp - 1))
        # Model function
        def F(z, d_exc):
            num = (self.alpha_fixed * np.exp(-z / d_exc) + (1 - self.alpha_fixed))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z)
            den = (self.alpha_fixed * np.exp(-z_0 / d_exc) + (1 - self.alpha_fixed))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z_0)
            return num / den
        # Fit
        popt, pcov = curve_fit(F, flat_z, F_data, p0=p0, bounds=([0],[np.inf]))
        d_exc = popt[0]

        perr = np.sqrt(np.diag(pcov))
        self.alpha_exc = self.alpha_fixed
        self.d_exc = d_exc
        self.alpha_exc_err = 0
        self.d_exc_err = perr[0]
        
        tirf_angle_lambda_factor = self.params.lambda_exc / (4*np.pi)
        tirf_angle_sqrt_factor = np.sqrt(((tirf_angle_lambda_factor / self.d_exc)**2 + self.params.n_s**2))
        tirf_angle_sin = tirf_angle_sqrt_factor / self.params.n_i
        tirf_angle_deriv = (1 / np.sqrt(1 - tirf_angle_sin**2)) * (1 / self.params.n_i) * (1/2) * (1 / tirf_angle_sqrt_factor) * tirf_angle_lambda_factor**2 * (2/self.d_exc**3) * (180/np.pi)
        self.tirf_angle = np.arcsin(tirf_angle_sin)*180/np.pi
        self.tirf_angle_err = np.abs(tirf_angle_deriv) * self.d_exc_err
        
    def fit_N0_exp_appr(self):
        """
        This function should be called once alpha_F and d_F have already been fitted.
        It iterates over all selected origamis and fits N_0 (keeping alpha_F and d_F fixed!)
        for each one.
        """
        self.N_0_arr = np.zeros(len(self.clust_means), dtype=float)
        def F(z, N_0):
            return N_0*(self.alpha_F * np.exp(-z / self.d_F) + (1 - self.alpha_F))
        for orig_idx in range(len(self.clust_means)):
            z_data = self.z_real[orig_idx, :]
            N_data = self.clust_means[orig_idx, :, 2]
            N_0_val, N_0_err = curve_fit(F, z_data, N_data, p0=self.clust_means[orig_idx, 0, 2], bounds=([0],[np.inf]))
            self.N_0_arr[orig_idx] = N_0_val
        self.N_renorm_arr = self.clust_means[:, :, 2]/self.N_0_arr[:, np.newaxis]
        self.N_0_avg = np.mean(self.N_0_arr)
        self.N_0_std = np.std(self.N_0_arr)
        # variables for plot
        self.z_ax_forplot = np.linspace(0, CALIB_PLOT_RANGE_NM, CALIB_PLOT_PTS)
        self.fit_func_forplot = F(self.z_ax_forplot, 1)
        
    def fit_N0_no_appr(self):
        """
        It iterates over all selected origamis and fits N_0 (keeping other parameters fixed) for each one.
        """
        self.N_0_arr = np.zeros(len(self.clust_means), dtype=float)
        def F(z, N_0):
            return N_0*(self.alpha_exc * np.exp(-z / self.d_exc) + (1 - self.alpha_exc))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z)/interp1d(Z_SIM_DISCR, self.coll_fl_discr)(0)
        for orig_idx in range(len(self.clust_means)):
            z_data = self.z_real[orig_idx, :]
            N_data = self.clust_means[orig_idx, :, 2]
            N_0_val, N_0_err = curve_fit(F, z_data, N_data, p0=self.clust_means[orig_idx, 0, 2], bounds=([0],[np.inf]))
            self.N_0_arr[orig_idx] = N_0_val
        self.N_renorm_arr = self.clust_means[:, :, 2]/self.N_0_arr[:, np.newaxis]
        self.N_0_avg = np.mean(self.N_0_arr)
        self.N_0_std = np.std(self.N_0_arr)
        # variables for plot
        self.z_ax_forplot = np.linspace(0, CALIB_PLOT_RANGE_NM, CALIB_PLOT_PTS)
        self.fit_func_forplot = F(self.z_ax_forplot, 1)
        
    def fit_N0_no_appr_biexp(self):
        """
        It iterates over all selected origamis and fits N_0 (keeping other parameters fixed) for each one.
        """
        self.N_0_arr = np.zeros(len(self.clust_means), dtype=float)
        def F(z, N_0):
            return N_0*(self.alpha_exc * np.exp(-z / self.d_exc) + (1 - self.alpha_exc) * np.exp(-z / self.d_long))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(z)/interp1d(Z_SIM_DISCR, self.coll_fl_discr)(0)
        for orig_idx in range(len(self.clust_means)):
            z_data = self.z_real[orig_idx, :]
            N_data = self.clust_means[orig_idx, :, 2]
            N_0_val, N_0_err = curve_fit(F, z_data, N_data, p0=self.clust_means[orig_idx, 0, 2], bounds=([0],[np.inf]))
            self.N_0_arr[orig_idx] = N_0_val
        self.N_renorm_arr = self.clust_means[:, :, 2]/self.N_0_arr[:, np.newaxis]
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
        self._calc_coll_fl_arr_axelrod()
        self.exc_prof = self.simpler_prof / self.params.coll_fl_interp_grid
        def F_exc(x, d_exc, b, c):
            return b * np.exp(-x/d_exc) + c                           
        popt, pcov = curve_fit(F_exc, Z_SIM_FIT_ARR, self.exc_prof, p0 = [200, 0.9, 0.1])
        self.d_exc = popt[0]
        self.d_exc_err = pcov[0, 0]
        self.exc_fit = F_exc(Z_SIM_FIT_ARR, popt[0], popt[1], popt[2])
        self.tirf_angle = np.arcsin(np.sqrt(((self.params.lambda_exc/(4*np.pi*popt[0]))**2 + self.params.n_s**2)/self.params.n_i**2))*180/np.pi

    def backcalc_glob_param(self):
        """
        This function approximates the real decay with an exponential to get the global decay parameters
        """
        self.glob_prof = (self.alpha_exc*np.exp(-Z_SIM_FIT_ARR/self.d_exc) + (1 - self.alpha_exc))*self.params.coll_fl_interp_grid
        def F_F(z, d_F, alpha_F, norm):
            return norm*(alpha_F*np.exp(-z/d_F) + (1 - alpha_F))
        popt, pcov = curve_fit(F_F, Z_SIM_FIT_ARR, self.glob_prof, p0 = [200, 0.9, 0.1], bounds=([0, 0, 0],[np.inf, np.inf, np.inf]))
        self.d_F = popt[0]
        self.d_F_err = pcov[0, 0]
        self.alpha_F = popt[1]
        self.alpha_F_err = pcov[1, 1]
        
    def backcalc_glob_param_biexp(self):
        """
        This function approximates the real decay with an exponential to get the global decay parameters
        """
        self.glob_prof = (self.alpha_exc*np.exp(-Z_SIM_FIT_ARR/self.d_exc) + (1 - self.alpha_exc)*np.exp(-Z_SIM_FIT_ARR/self.d_long))*interp1d(Z_SIM_DISCR, self.coll_fl_discr)(Z_SIM_FIT_ARR)
        def F_F(z, d_F, alpha_F, norm):
            return norm*(alpha_F*np.exp(-z/d_F) + (1 - alpha_F))
        popt, pcov = curve_fit(F_F, Z_SIM_FIT_ARR, self.glob_prof, p0 = [200, 0.9, 0.1], bounds=([0, 0, 0],[np.inf, np.inf, np.inf]))
        self.d_F = popt[0]
        self.d_F_err = pcov[0, 0]
        self.alpha_F = popt[1]
        self.alpha_F_err = pcov[1, 1]
        
    def z_from_N_appr(self, n_ph, n_ph_0):
        """
        This function computes z from N, given the global decay parameters
        """
        return self.d_F*np.log(self.alpha_F/((n_ph/n_ph_0) - (1 - self.alpha_F)))
         
    def z_from_N_no_appr(self, n_ph, n_ph_0):
        """
        This function computes z from N, from the real, not approximated, not exponential decay curve
        """
        coll_fl_fn = interp1d(Z_SIM_DISCR, self.coll_fl_discr,
                        bounds_error=False, fill_value='extrapolate')

        n_ph   = np.atleast_1d(np.squeeze(n_ph))
        n_ph_0 = np.atleast_1d(np.squeeze(n_ph_0))

        results = np.empty_like(n_ph, dtype=float)
        for i, (a, b) in enumerate(np.broadcast(n_ph, n_ph_0)):
            def f(z, a=a, b=b):   # default args capture current values
                return (a / b
                        - coll_fl_fn(z) * (self.alpha_exc * np.exp(-z / self.d_exc)
                        + (1 - self.alpha_exc)))
            results[i] = fsolve(f, x0=50)[0]

        return results
        
    def backcalc_z(self):
        """
        This function uses the calibrated parameters to convet N into z
        """
        self.spat_locs = []
        for orig_idx in range(len(self.clust_locs)):
            self.spat_locs.append(
                np.concatenate(
                    (
                        self.clust_locs[orig_idx][:, :2],
                        self.z_from_N_no_appr(self.clust_locs[orig_idx][:, 2], self.N_0_arr[orig_idx, np.newaxis])[:, np.newaxis]  
                    ), axis = 1
                )
            )
        
    def calc_spat_sigma_gmm(self):
        """
        This function recalculates the spatial 3D sigmas for the localization, converted into x, y, z.
        """
        self.spat_covs = np.zeros((len(self.clust_means), self.params.n_clust_exp, 3, 3), dtype=float)
        for orig_idx in range(len(self.clust_means)):
            for clust_idx in range(self.params.n_clust_exp):
                gmm = GaussianMixture(n_components=1, covariance_type='full', n_init=5, max_iter=300)
                gmm.fit(np.array(self.spat_locs[orig_idx][self.clust_labels[orig_idx]==clust_idx]))
                self.spat_covs[orig_idx, clust_idx, :, :] = gmm.covariances_[0]
        self.spat_sigma_avg = np.mean(np.sqrt(np.diagonal(self.spat_covs, axis1=2, axis2=3)), axis=0)
        
@dataclass
class Params:
    """
    dataclass containing the parameters for the calibration
    """
    # sample parameters
    sampletype: str | None = None
    orientation: str | None = None
    n_clust_exp: int | None = None
    z_nm_arr: np.ndarray | None = None
    # fit parameters
    tirf_angle: float | None = None
    fix_angle_choice : bool = False
    res_analysis_choice : bool = False
    # SIMPLER filtering parameters
    spat_tol_nm: float | None = None # how far can two locs be to be considered the same event
    # pre-clustering parameters
    preclust_gamma: float | None = None
    preclust_eps: float | None = None
    # photon number guesses and bounds
    n_guess: tuple | None = None
    should_use_n_guess: bool = False
    n_bounds: tuple | None = None
    should_use_n_bounds: bool = False
    x_bounds: tuple | None = None
    should_use_x_bounds: bool = False
    y_bounds: tuple | None = None
    should_use_y_bounds: bool = False
    # should cluster resolution analysis be performed or not
    should_do_res_analysis: bool = True
    # setup parameters
    lambda_exc: float | None = None
    lambda_em: float | None = None
    n_s: float | None = None
    n_i: float | None = None
    na: float | None = None
    # Collection efficiencies
    coll_fl_tab: np.ndarray | None = None
    coll_fl_interp_grid: np.ndarray | None = None
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

    def share_params(self):
        """
        This function updates the parameters for all the other analysis classes
        """
        self.simpler.params = self.params
        self.clust.params = self.params
        self.fit.params = self.params

    @pyqtSlot(str)
    def upd_sampletype(self, value):
        self.params.sampletype = value
        match self.params.sampletype:
            case '12HB (5 points)':
                self.params.n_clust_exp = HB_N_CLUST_EXP
                self.params.z_nm_arr = HB_Z_SITES_NM
            case 'Rifle (4 points)':
                self.params.n_clust_exp = RIFLE_N_CLUST_EXP
                self.params.z_nm_arr = RIFLE_Z_SITES_NM
        self.share_params()
        
    @pyqtSlot(str)
    def upd_orientation(self, value):
        self.params.orientation = value
        self.share_params()

    @pyqtSlot(float)
    def upd_fix_angle(self, value):
        self.params.tirf_angle = value
        self.share_params()
        
    @pyqtSlot(bool)
    def upd_fix_angle_choice(self, value):
        self.params.fix_angle_choice = value
        self.share_params()
        
    @pyqtSlot(bool)
    def upd_res_analysis_choice(self, value):
        self.params.res_analysis_choice = value
        self.share_params()

    @pyqtSlot(float)
    def upd_spat_tol(self, value):
        self.params.spat_tol_nm = value
        self.share_params()

    @pyqtSlot(float)
    def upd_preclust_gamma(self, value):
        self.params.preclust_gamma = value
        self.share_params()
        
    @pyqtSlot(float)
    def upd_preclust_eps(self, value):
        self.params.preclust_eps = value
        self.share_params()
        
    @pyqtSlot(object)
    def upd_n_guess(self, value):
        self.params.n_guess = value
        self.share_params()
        
    @pyqtSlot(object)
    def upd_n_bounds(self, value):
        self.params.n_bounds = value
        self.share_params()
        
    @pyqtSlot(object)
    def upd_x_bounds(self, value):
        self.params.x_bounds = value
        self.share_params()
        
    @pyqtSlot(object)
    def upd_y_bounds(self, value):
        self.params.y_bounds = value
        self.share_params()
        
    @pyqtSlot(bool)
    def upd_n_guess_choice(self, value):
        self.params.should_use_n_guess = value
        self.share_params()
        
    @pyqtSlot(bool)
    def upd_n_bounds_choice(self, value):
        self.params.should_use_n_bounds = value
        self.share_params()
        
    @pyqtSlot(bool)
    def upd_x_bounds_choice(self, value):
        self.params.should_use_x_bounds = value
        self.share_params()
        
    @pyqtSlot(bool)
    def upd_y_bounds_choice(self, value):
        self.params.should_use_y_bounds = value
        self.share_params()
        
    @pyqtSlot(float)
    def upd_lambda_exc(self, value):
        self.params.lambda_exc = value
        self.share_params()
        
    @pyqtSlot(float)
    def upd_lambda_em(self, value):
        self.params.lambda_em = value
        self.share_params()
            
    @pyqtSlot(float)
    def upd_n_i(self, value):
        self.params.n_i = value
        self.share_params()
        
    @pyqtSlot(float)
    def upd_n_s(self, value):
        self.params.n_s = value
        self.share_params()
        
    @pyqtSlot(float)
    def upd_na(self, value):
        self.params.na = value
        self.share_params()
        
    @pyqtSlot(object)
    def upd_coll_fl_tab(self, coll_fl_tab):
        """
        This function receives a table of collection efficiencies, corresponding to the value of NA chosen on UI.
        """
        self.params.coll_fl_tab = coll_fl_tab
        self.share_params()
                
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
                roi = metadata[0]['Micro-Manager Metadata']['ROI']
                _lgr.info(f"Number of frames in movie: {frames}")
                _lgr.info(f"Exposure time in ms: {exp_time_ms}")
                _lgr.info(f"Pixel size in nm: {px_size_nm}")
                self.signals.send_msg_toprint.emit(MessageType.INFO, "Movie metadata readed correctly.")
                self.signals.send_msg_toprint.emit(MessageType.SIMPLE, f"Number of frames in movie: {frames}")
                self.signals.send_msg_toprint.emit(MessageType.SIMPLE, f"Exposure time in ms: {exp_time_ms}")
                self.signals.send_msg_toprint.emit(MessageType.SIMPLE, f"Pixel size in nm: {px_size_nm}")
                self.signals.send_msg_toprint.emit(MessageType.SIMPLE, f"ROI in px: {roi}")
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
        self.clust.pre_clust_denoise(self.simpler.locs)
        self.signals.tell_analysis_step_start.emit(AnalysisStatus.SITE_CLUST, self.clust.tot_orig_kept)
        self.clust.do_clust_xyn()
        self.clust.calc_tilt_angles()
        self.clust.calc_z_real()
        if self.simpler.locs:
            self.signals.tell_clust_done.emit(True)
        else:
            self.signals.tell_clust_done.emit(False)

    def save_clust(self):
        """
        This function saves the array of clusterization results of the selected origamis only as a .npy
        """
        locs_res_filename = self.picks_data_path.stem + "_locs.json"
        clust_labels_res_filename = self.picks_data_path.stem + "_labels.json"
        clust_means_res_filename = self.picks_data_path.stem + "_clusters.npy"
        clust_covs_res_filename = self.picks_data_path.stem + "_covs.npy"
        with open(RES_DIR / Path(locs_res_filename), "w") as f:
            json.dump([self.clust.locs_clust[idx].tolist() for idx, truth_val in enumerate(self.clust.selec_orig_list) if truth_val], f)
        with open(RES_DIR / Path(clust_labels_res_filename), "w") as f:
            json.dump([self.clust.clust_labels[idx].tolist() for idx, truth_val in enumerate(self.clust.selec_orig_list) if truth_val], f)  
        np.save(RES_DIR / Path(clust_means_res_filename), self.clust.clust_means[self.clust.selec_orig_list,:,:])
        np.save(RES_DIR / Path(clust_covs_res_filename), self.clust.clust_covs[self.clust.selec_orig_list,:,:,:])
        
    @pyqtSlot(int)
    def refit_orig(self, orig_num: int):
        """
        This function re-fits (both pre-clustering de-noising and GMM clustering) the currently displayed origami
        """
        locs_unlabel = np.concatenate((self.clust.locs_clust[orig_num], self.clust.locs_noise[orig_num]))
        new_labels = self.clust.pre_clust_denoise_inorig(locs_unlabel)
        if new_labels is None:
            _lgr.warning("Re-fit failed at pre-clustering de-noising step, try changing parameters")
            self.signals.send_msg_toprint.emit(MessageType.WARNING, "Re-fit failed at pre-clustering de-noising step, try changing parameters")
            return
        else:
            if self.params.orientation=='vertical':
                new_means, new_covs, new_clust_labels = self.clust.gmm_clust_inorig(locs_unlabel[new_labels!=-1])
            elif self.params.orientation=='horizontal':
                new_means, new_covs, new_clust_labels = self.clust.gmm_clust_inorig_2D(locs_unlabel[new_labels!=-1])
            if new_means is None:
                _lgr.warning("Re-fit failed at GMM clustering step, try changing parameters")
                self.signals.send_msg_toprint.emit(MessageType.WARNING, "Re-fit failed at GMM clustering step, try changing parameters")
            else:
                # if new fit passed all steps, update old results with new
                self.clust.locs_clust[orig_num] = locs_unlabel[new_labels!=-1]
                self.clust.clust_labels[orig_num] = new_clust_labels
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
        if self.params.n_clust_exp==self.clust.clust_means.shape[1]:
            if self.params.res_analysis_choice:
                self.perform_calib_steps(
                    True,
                    self.clust.clust_means[self.clust.selec_orig_list, :, :],
                    [np.array(self.clust.locs_clust[idx]) for idx, truth_val in enumerate(self.clust.selec_orig_list) if truth_val],
                    [np.array(self.clust.clust_labels[idx]) for idx, truth_val in enumerate(self.clust.selec_orig_list) if truth_val]
                )
            else:
                self.perform_calib_steps(
                    False,
                    self.clust.clust_means[self.clust.selec_orig_list, :, :],
                    None,
                    None
                )
        else:
            self.signals.send_msg_toprint(MessageType.ERROR, 'Mismatch between number of expected and detected clusters, change origami type')
        self.signals.tell_calib_done.emit('')
        
    @pyqtSlot(Path)
    def do_calib_fromfile(self, clust_path: Path):
        """
        This function performs the SIMPLER calibration using the results from a previous clusterization saved on file, and the expected
        z positions, corrected according to the origamin tilt. 
        """
        try:
            clust_fromfile = np.load(clust_path)
            _lgr.info(f"Result array shape: {clust_fromfile.shape}")
        except Exception as e:
            self.signals.send_msg_toprint.emit(MessageType.ERROR, f"Cannot open result file because of Exception: {e}")
            return
        if self.params.res_analysis_choice:
            try:
                clust_locs_filename = clust_path.stem[:clust_path.stem.rfind("_clusters")] + "_locs.json"
                clust_labels_filename = clust_path.stem[:clust_path.stem.rfind("_clusters")] + "_labels.json"
                clust_locs_path = clust_path.parent / Path(clust_locs_filename)
                clust_label_path = clust_path.parent / Path(clust_labels_filename)
                with open(clust_locs_path, "r") as f:
                    clust_locs_fromfile = [np.array(a) for a in json.load(f)]
                with open(clust_label_path, "r") as f:
                    clust_labels_fromfile = [np.array(a) for a in json.load(f)]
                self.params.should_do_res_analysis = True
                self.share_params()
            except Exception as e:
                self.signals.send_msg_toprint.emit(MessageType.WARNING, f"Cannot open localization and/or label files because of Exception: {e}. Calibration will be performed, but resolution analysis will be omitted.")
                clust_locs_fromfile = None
                clust_labels_fromfile = None
                self.params.should_do_res_analysis = False
                self.share_params()
        else:
            clust_locs_fromfile = None
            clust_labels_fromfile = None
            self.params.should_do_res_analysis = False
            self.share_params()
        if (clust_fromfile.dtype==float) and (clust_fromfile.shape[1:]==(self.params.n_clust_exp, 3)) and (len(clust_fromfile.shape)==3):
            self.clust.upd_clust_fromfile(clust_fromfile, clust_locs_fromfile, clust_labels_fromfile)
            self.perform_calib_steps(self.params.should_do_res_analysis, clust_fromfile, clust_locs_fromfile, clust_labels_fromfile)
            self.signals.tell_calib_done.emit('from file')
        else:
            self.signals.send_msg_toprint.emit(MessageType.ERROR, "Result file does not have expected structure or content")

    def perform_calib_steps(self, should_do_res_analysis, clust_forcalib, clust_locs, clust_labels):
        self.clust.calc_tilt_angles()
        self.clust.calc_z_real()
        if should_do_res_analysis:
            self.fit.upd_data_forfit(clust_forcalib, clust_locs, clust_labels, self.clust.tilt_angles, self.clust.z_real)
        else:
            self.fit.upd_data_forfit(clust_forcalib, None, None, self.clust.tilt_angles, self.clust.z_real)
        if self.params.fix_angle_choice and self.params.tirf_angle is not None:
            self.fit.fit_no_appr_fix_angle_each_orig()
            self.fit.backcalc_glob_param()
        else:
            self.fit.fit_renorm_no_appr()
            self.fit.fit_N0_no_appr()
            self.fit.backcalc_glob_param()
        '''
        match CALIB_MODE:
            case 'no_appr':
                self.fit.fit_renorm_no_appr()
                self.fit.fit_N0_no_appr()
                self.fit.backcalc_glob_param()
            case 'no_appr_fix_angle':
                self.fit.fit_renorm_no_appr_fix_angle()
                self.fit.fit_N0_no_appr()
                self.fit.backcalc_glob_param()
            case 'no_appr_fix_angle_each_orig':
                self.fit.fit_no_appr_fix_angle_each_orig()
                self.fit.backcalc_glob_param()
            case 'no_appr_fix_angle_biexp':
                self.fit.fit_renorm_no_appr_fix_angle_biexp()
                self.fit.fit_N0_no_appr_biexp()
                self.fit.backcalc_glob_param()   
            case 'no_appr_fix_angle_biexp_each_orig':
                self.fit.fit_no_appr_fix_angle_biexp_each_orig()
                self.fit.backcalc_glob_param()  
            case 'no_appr_spacer':
                self.fit.fit_renorm_no_appr_spacer()
                self.fit.fit_N0_no_appr()
                self.fit.backcalc_glob_param()
            case 'no_appr_fix_alpha':
                self.fit.fit_renorm_no_appr_fix_alpha()
                self.fit.fit_N0_no_appr()
                self.fit.backcalc_glob_param()
            case 'exp_appr':
                self.fit.fit_renorm_exp_appr()
                self.fit.fit_N0_exp_appr()
                self.fit.backcalc_tirf_angle()
    '''
        if should_do_res_analysis and (clust_locs is not None) and (clust_labels is not None):
            self.fit.backcalc_z()
            self.fit.calc_spat_sigma_gmm()

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
    positions = np.array(RIFLE_Z_SITES_NM)
    angles = np.array([clus.tilts_form_xy(positions, o_pos[:, 0:2])[0] for o_pos in clus.clust_means])
    z = clus.z_from_tilt(angles, positions)
    alpha_F, d_F, errors = clus.fit_N(z, clus.clust_means[:, :, 2])
    plot_origami_fit(z, clus.clust_means[:, :, 2], alpha_F, d_F)


if __name__ == "__main__":
    filepath_str = r"X:\messdaten\Giovanni_A\SIMPLER\260313\Rifle_4pts_R2_40gain_500pMCy3B_200mW_100ms_23TIRF\R2\R2_2_MMStack_Pos0.ome_locs_picked_standing.hdf5"
    data_path = Path(filepath_str)
    metadata_path = data_path.parent / Path(data_path.stem + ".yaml")
    analysis_worker = AnalysisWorker(data_path, metadata_path)
    analysis_worker.load_data()
    analysis_worker.do_filt()
    analysis_worker.do_clust()
