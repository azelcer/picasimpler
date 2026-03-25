import numpy as np
import pandas as pd
import logging as _lgn
import yaml
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture
from sklearn.cluster import HDBSCAN
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from picasimpler.helpers.status import AnalysisStatus
from picasimpler.helpers.conversions import px_to_nm
from picasimpler.config.config_var import (
    SPAT_TOL_NM,
    MAX_FIRST_FRAME_PERC,
    MIN_LAST_FRAME_PERC,
    FRAME_MEDIAN_PERC_RANGE,
    MAX_ON_FRAMES_PERC,
    MIN_PERC_LOC_INCLUST,
    MIN_GOOD_LOC,
    N_CLUST_EXP,
    H_SITES_NM
)

_lgn.basicConfig()
_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.INFO)

class SIMPLERSignals(QObject):
    tell_analysis_elem_done = pyqtSignal(int)

class SIMPLER:
    """
    Class containing all the SIMPLER methods and results
    """
    def __init__(self, signals: SIMPLERSignals):
        self.signals: SIMPLERSignals = signals
        self.locs: list | None = None
    
    def get_loc_x(self, orig_num):
        return self.locs[orig_num][:, 0]
    
    def get_loc_y(self, orig_num):
        return self.locs[orig_num][:, 1]
    
    def get_loc_n(self, orig_num):
        return self.locs[orig_num][:, 2]
    
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
    
    def filter_locs_inpick(self, df_pick, r_th_sq):
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
            
    def filter_data(self, df: pd.DataFrame, px_size_nm: int, r_th_sq: float):
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
                (idx_to_discard, self.filter_locs_inpick(df.iloc[groupjump[pick_idx]:groupjump[pick_idx + 1]], r_th_sq)),
                axis=0
            )
            self.signals.tell_analysis_elem_done.emit(pick_idx + 1)
        df_filtered = df.drop(labels=idx_to_discard, axis=0)
        df_filtered = df_filtered.reset_index(level=None, drop=True, inplace=False,
                                              col_level=0)
        end = _time.time()
        _lgr.info('Time of filtering step: %s s. %s of %s (%.2f%%) localizations discarded',
                end - start, len(idx_to_discard), n_loc_initial, 100 * len(idx_to_discard) / n_loc_initial)
        self.save_locs(df_filtered, px_size_nm)
    
class ClusterizationSignals(QObject):
    tell_analysis_elem_done = pyqtSignal(int)
    
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
        self.selec_orig_list: list | None = None

    def get_clust_x(self, orig_num):
        return self.locs_clust[orig_num][:, 0]
    
    def get_clust_y(self, orig_num):
        return self.locs_clust[orig_num][:, 1]
    
    def get_clust_n(self, orig_num):
        return self.locs_clust[orig_num][:, 2]
    
    def get_noise_x(self, orig_num):
        return self.locs_noise[orig_num][:, 0]
    
    def get_noise_y(self, orig_num):
        return self.locs_noise[orig_num][:, 1]
    
    def get_noise_n(self, orig_num):
        return self.locs_noise[orig_num][:, 2]

    @staticmethod
    def reorder_clust(means, sigmas):
        """
        This method reorders in descending order tuples of means and sigmas based on the mean of the last coordinate
        (number of photons). It is used to order clusters from bottom to top
        """
        return list(zip(*sorted(zip(means, sigmas), key=lambda pair: -pair[0][2])))

    def pre_clust_denoise(self, locs: list, min_perc_loc_insite: float, min_good_loc: int):
        """
        This function applies HDBSCAN to separate major clusters (without mecessarily resolving them!) from scattered
        noise and unwanted smaller clusters (such as double events)
        """
        self.locs_clust = []
        self.locs_noise = []
        for orig_idx in range(len(locs)):
            min_clust_size = int(min_perc_loc_insite*len(locs[orig_idx]))
            hdbsc = HDBSCAN(min_cluster_size=min_clust_size).fit(locs[orig_idx])
            if len(locs[orig_idx][hdbsc.labels_!=-1]) > min_good_loc:
                self.locs_clust.append(locs[orig_idx][hdbsc.labels_!=-1])
                self.locs_noise.append(locs[orig_idx][hdbsc.labels_==-1])
            self.signals.tell_analysis_elem_done.emit(orig_idx)
        self.tot_orig_kept = len(self.locs_clust)

    def do_clust_xyn(self, n_clust_exp: int):
        """
        This function loops over all origamis and cluster their data in 3D (x, y, N).
        """
        start = _time.time()
        tot_orig_bf_clust = len(self.locs_clust)
        # variables needed to discard origamis not passing the clusterzation test
        n_orig_discarded = 0
        # here we will store all data relative to the clusterization result
        kept_orig_loc_list = []
        kept_orig_noise_list = []
        clust_means_list = []
        clust_covs_list = []
        for orig_idx in range(tot_orig_bf_clust):
            pick_kept = True
            for n_clust in range(n_clust_exp, 0, -1):
                gmm = GaussianMixture(n_components=n_clust, covariance_type='full', n_init=1, init_params='k-means++')
                gmm.fit(self.locs_clust[orig_idx])
                last_bic = gmm.bic(self.locs_clust[orig_idx])
                if n_clust == n_clust_exp: # compute BIC for the expected number of clusters
                    ref_bic = last_bic
                    clust_means, clust_covs = self.reorder_clust(gmm.means_, gmm.covariances_)
                # now we decrease the number of clusters and as soon as one gives better result, we discard the origami and exit the loop
                elif last_bic < ref_bic:
                    n_orig_discarded += 1
                    pick_kept = False
                    break
            if pick_kept:
                kept_orig_loc_list.append(self.locs_clust[orig_idx])
                kept_orig_noise_list.append(self.locs_noise[orig_idx])
                clust_means_list.append(clust_means)
                clust_covs_list.append(clust_covs)
            self.signals.tell_analysis_elem_done.emit(orig_idx + 1)
        self.tot_orig_kept = len(kept_orig_loc_list)
        self.locs_clust = kept_orig_loc_list
        self.locs_noise = kept_orig_noise_list
        self.clust_means = np.asarray(clust_means_list, dtype=float)
        self.clust_covs = np.asarray(clust_covs_list, dtype=float)
        self.selec_orig_list = [True]*self.clust_means.shape[0]
        end = _time.time()
        _lgr.info('Time of clustering step: %s s. %s of %s (%.2f%%) origamis discarded',
                end - start, n_orig_discarded, tot_orig_bf_clust, 100 * n_orig_discarded / tot_orig_bf_clust)

    def tilts_form_xy(self, origami_positions: np.ndarray, xy_positions: np.ndarray):
        """
        Compute tilt angles of the origami with respect to the surface, based on the
        x and y positions of the clusterized sites.

        Parameters
        ----------
        origami_positions : np.ndarray
            Site positions along the origami in nm. For example, if the origami has 3 sites
            50 nm apart, t should be [0, 50.0, 100.0]
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
            raise ValueError("El largo de la lista de distancias y de posiciones no coinciden")

        M = np.column_stack((np.ones(origami_positions.shape[0]), origami_positions))
        coeffs, residuals, rank, s = np.linalg.lstsq(
            M, xy_positions, rcond=None
        )
        A, B = coeffs[1]  # coefs[0] tiene (x, y) del punto de unión de origami

        # recupero los ángulos (en grados)
        phi = np.arctan2(B, A)
        theta = np.arccos(A / np.cos(np.radians(phi)))
        return theta, phi


@dataclass
class Params:
    """
    dataclass containing the parameters for the calibration
    """
    
    # kinetics filtering parameters
    max_first_frame_perc: float
    min_last_frame_perc: float
    frame_median_perc_range: list
    max_on_frames_perc: float
    
    # SIMPLER filtering parameters
    spat_tol_nm: float # how far can two locs be to be considered the same event
    
    # clustering parameters
    min_perc_loc_inclust: float
    min_good_loc: int
    n_clust_exp: int
    h_sites_nm: list
    
    # movie parameters
    n_frames: int = field(init=False) # number of frames in movie
    exp_time_ms: float = field(init=False) # exposure time in ms
    px_size_nm: float = field(init=False) # camera pixel size in nm
    
    # convenience parameters
    r_th_sq: float = field(init=False)
    
@dataclass
class Data:
    """
    dataclass containing the localization data
    """
    picks_data_path: Path
    metadata_path: Path
    
    is_data_file_open: bool = field(default=False)
    is_metadata_file_open: bool = field(default=False)
    
    tot_picks: int = field(init=False)
    tot_orig: int = field(init=False)
    
    df_raw: pd.DataFrame = field(init=False) # dataframe with all data
    df_orig: pd.DataFrame = field(init=False) # dataframe with picks filtered by PAINT kinetics
    
class AnalysisSignals(QObject):
    # type of analysis step starting now, and total number of element in it
    tell_analysis_step_start = pyqtSignal(AnalysisStatus, int)
    tell_analysis_elem_done = pyqtSignal(int)
    tell_filtering_done = pyqtSignal()
    tell_filt_done = pyqtSignal()
    tell_clust_done = pyqtSignal(bool)
    
class AnalysisWorker(QObject):
    def __init__(self, picks_data_path, metadata_path):
        super().__init__()
        # all signals must be instatiated in main thread, so inside __init__
        self.signals = AnalysisSignals()
        self.simpler_signals = SIMPLERSignals()
        self.clust_signals = ClusterizationSignals()
        self.params = Params(
            MAX_FIRST_FRAME_PERC,
            MIN_LAST_FRAME_PERC,
            FRAME_MEDIAN_PERC_RANGE,
            MAX_ON_FRAMES_PERC,
            SPAT_TOL_NM,
            MIN_PERC_LOC_INCLUST,
            MIN_GOOD_LOC,
            N_CLUST_EXP,
            H_SITES_NM
        )
        self.data = Data(picks_data_path, metadata_path)
        self.simpler: SIMPLER = SIMPLER(self.simpler_signals)
        self.clust: Clusterization = Clusterization(self.clust_signals)

    def load_data(self):
        """
        This function calls other functions to load data and metadata from file
        """
        self.load_hdf5_todf()
        if self.data.is_data_file_open:
            self.load_metadata()
            self.params.r_th_sq = (self.params.spat_tol_nm / self.params.px_size_nm)**2

    def load_hdf5_todf(self):
        """
        This function opens the hdf5 containing all the picked structures
        """
        try:
            with pd.HDFStore(self.data.picks_data_path, 'r') as store:
                hdf5_node_list = [node._v_pathname for node in store._handle.walk_nodes()]
                if '/locs' not in hdf5_node_list:
                    _lgr.error('hdf5 file does not have expected structure')
                    # FIXME: this cleans previous file is lodaded
                    self.data.df_raw = None
                    self.data.tot_picks = 0
                    self.data.is_data_file_open = False
                _lgr.info('hdf5 file has expected structure')
                df_data = store['/locs']
                # count total number of picks
                tot_picks = df_data['group'].iloc[-1] + 1
                _lgr.info(f"Total number of picks: {tot_picks}")
                self.data.df_raw = df_data
                self.data.tot_picks = tot_picks
                self.data.is_data_file_open = True
        except Exception as e:
            if isinstance(e, KeyError) and str(e) == "'group'":
                _lgr.error(f"Error opening hdf5 file: picks were not found in file")
            else:
                _lgr.error(f"Error {type(e)} opening hdf5 file: {e}")
            self.data.df_raw = None
            self.data.tot_picks = 0
            self.data.is_data_file_open = False

    def load_metadata(self):
        """
        this function loads the metadata from the yaml file
        """
        try:
            with open(self.data.metadata_path, "r") as metadata_file:
                metadata = list(yaml.load_all(metadata_file, Loader=yaml.FullLoader))
                frames = metadata[0]['Frames']
                exp_time_ms = metadata[0]['Micro-Manager Metadata']['Exposure-ms']
                px_size_nm = metadata[1]['Pixelsize']
                _lgr.info(f"Number of frames in movie: {frames}")
                _lgr.info(f"Exposure time in ms: {exp_time_ms}")
                _lgr.info(f"Pixel size in nm: {px_size_nm}")
                self.params.n_frames = frames
                self.params.exp_time_ms = exp_time_ms
                self.params.px_size_nm = px_size_nm
                self.data.is_metadata_file_open = True
        except Exception as e:
            _lgr.error(f"Error opening yaml file because of: {e}")
            self.params.n_frames = None
            self.params.exp_time_ms = None
            self.params.px_size_nm = None
            self.data.is_metadata_file_open = False            
    
    def filter_kin_orig(self):
        """
        this function removes picks not following expected PAINT statistics
        """
        picks_tokeep = []
        groups = np.array(self.data.df_raw['group'])
        groupjump = np.nonzero(np.diff(groups, prepend=-np.inf, append=np.inf) != 0)[0]
        for pick_idx in range(self.data.tot_picks):
            pick_df = self.data.df_raw.iloc[groupjump[pick_idx]:groupjump[pick_idx + 1]]
            first_frame_perc = np.min(pick_df['frame'])/self.params.n_frames
            last_frame_perc = np.max(pick_df['frame'])/self.params.n_frames
            med_frame_perc = np.median(pick_df['frame'])/self.params.n_frames
            unique_frames = set(pick_df['frame'])
            num_on_frames_perc = len(unique_frames)/self.params.n_frames
            # to be considered an origami, the pick has to pass all following kinetics test
            if not (
                (first_frame_perc>self.params.max_first_frame_perc) or
                (last_frame_perc<self.params.min_last_frame_perc) or
                (med_frame_perc<self.params.frame_median_perc_range[0]) or
                (med_frame_perc>self.params.frame_median_perc_range[1]) or
                (num_on_frames_perc>self.params.max_on_frames_perc)
            ):
                picks_tokeep.append(pick_idx)
                self.signals.tell_analysis_elem_done.emit(pick_idx + 1)
        df_orig = self.data.df_raw.loc[self.data.df_raw['group'].isin(picks_tokeep)]
        n_orig = len(picks_tokeep)
        _lgr.info(f"Kept {n_orig} picks out of {self.data.tot_picks}, considered to be individual origamis")
        self.data.df_orig = df_orig
        self.data.tot_orig = n_orig

    @pyqtSlot()
    def do_filt(self):
        """
        this function calls one by one all the filtering steps (filtering based on kinetics and SIMPLER localization filtering)
        """
        self.signals.tell_analysis_step_start.emit(AnalysisStatus.KIN_FILT, self.data.tot_picks)
        self.filter_kin_orig()
        self.signals.tell_analysis_step_start.emit(AnalysisStatus.SIMPLER_FILT, self.data.tot_orig)
        self.simpler.filter_data(self.data.df_orig, self.params.px_size_nm, self.params.r_th_sq)
        self.signals.tell_filt_done.emit()
        
    @pyqtSlot()
    def do_clust(self):
        """
        This function call the clusterization function
        """
        self.signals.tell_analysis_step_start.emit(AnalysisStatus.PRE_CLUST, self.data.tot_orig)
        self.clust.pre_clust_denoise(self.simpler.locs, self.params.min_perc_loc_inclust, self.params.min_good_loc)
        self.signals.tell_analysis_step_start.emit(AnalysisStatus.SITE_CLUST, self.clust.tot_orig_kept)
        self.clust.do_clust_xyn(self.params.n_clust_exp)
        if self.simpler.locs:
            self.signals.tell_clust_done.emit(True)
        else:
            self.signals.tell_clust_done.emit(True)
        
if __name__=="__main__":
    filepath_str = r"X:\messdaten\Giovanni_A\SIMPLER\260313\Rifle_4pts_R2_40gain_500pMCy3B_200mW_100ms_23TIRF\R2\R2_2_MMStack_Pos0.ome_locs_picked_standing.hdf5"
    data_path = Path(filepath_str)
    metadata_path = data_path.parent / Path(data_path.stem + ".yaml")
    analysis_worker = AnalysisWorker(data_path, metadata_path)
    analysis_worker.do_filt()
    analysis_worker.do_clust()
    