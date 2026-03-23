import numpy as np
import pandas as pd
import logging as _lgn
import yaml
import time as _time

from dataclasses import dataclass, field
from pathlib import Path
from scipy.spatial import distance
from sklearn.mixture import GaussianMixture
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from picasimpler.helpers.status import AnalysisStatus
from picasimpler.config.config_var import (
    SPAT_TOL_NM,
    MAX_FIRST_FRAME_PERC,
    MIN_LAST_FRAME_PERC,
    FRAME_MEDIAN_PERC_RANGE,
    MAX_ON_FRAMES_PERC,
    N_CLUST_EXP,
    H_SITES_NM
)

_lgn.basicConfig()
_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.INFO)

class AnalysisSignals(QObject):
    # type of analysis step starting now, and total number of element in it
    tell_analysis_step_start = pyqtSignal(AnalysisStatus, int)
    tell_analysis_elem_done = pyqtSignal(int)
    tell_filtering_done = pyqtSignal()
    tell_filt_done = pyqtSignal()
    tell_clust_done = pyqtSignal()

@dataclass
class SIMPLERLocalizations:
    """
    Dataclass containing all the filtered SIMPLER localizations and some methods to access them
    """
    all_orig_loc_list: list
    
    def get_loc_x(self, orig_num):
        return self.all_orig_loc_list[orig_num][:, 0]
    
    def get_loc_y(self, orig_num):
        return self.all_orig_loc_list[orig_num][:, 1]
    
    def get_loc_n(self, orig_num):
        return self.all_orig_loc_list[orig_num][:, 2]
    
@dataclass
class ClusterResults:
    """
    dataclass containing all the results (means and covariances) of the clusters after site clusterization.
    Coordinates go from 0 to 2 and are always in this order: x, y and N (number of photons)
    """
    clust_means: np.ndarray
    clust_covs: np.ndarray
    
    def __post_init__(self):
        self.clust_means = np.asarray(self.clust_means, dtype=float)
        self.clust_covs = np.asarray(self.clust_covs, dtype=float)

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
    tot_orig_after_clust: int = field(init=False)
    
    df_raw: pd.DataFrame = field(init=False) # dataframe with all data
    df_orig: pd.DataFrame = field(init=False) # dataframe with picks filtered by PAINT kinetics
    df_filt: pd.DataFrame = field(init=False) # dataframe after SIMPLER localization filter
    df_after_clust: pd.DataFrame = field(init=False) # dataframe after clusterization
    
    simpler_locs: SIMPLERLocalizations = field(init=False) # dataclass containing the SIMPLER localizations 
    cluster_res: ClusterResults = field(init=False) # dataclass containing the clusterization data 
    
class AnalysisWorker(QObject):
    def __init__(self, picks_data_path, metadata_path):
        super().__init__()
        self.signals = AnalysisSignals()
        self.params = Params(
            MAX_FIRST_FRAME_PERC,
            MIN_LAST_FRAME_PERC,
            FRAME_MEDIAN_PERC_RANGE,
            MAX_ON_FRAMES_PERC,
            SPAT_TOL_NM,
            N_CLUST_EXP,
            H_SITES_NM
        )
        self.data = Data(picks_data_path, metadata_path)

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
    
    def filter_orig(self):
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
    
    def filter_locs_inpick(self, df_pick):
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
            has_prev = np.any(distance_prev < self.params.r_th_sq, axis=1)
            has_next = np.any(distance_next < self.params.r_th_sq, axis=1)
            discard_yn[f_slice][np.logical_and(has_prev, has_next)] = 0
        rel_idx_to_discard = np.where(discard_yn == 1)[0]
        abs_idx_to_discard = np.array(df_pick.index)[rel_idx_to_discard]
        return abs_idx_to_discard
            
    def filter_data(self):
        """
        This function filters all data, pick by pick, using SIMPLER criteria
        """
        start = _time.time()
        n_loc_initial = len(self.data.df_orig['frame'])
        idx_to_discard = np.array([])
        groups = np.array(self.data.df_orig['group'])
        groupjump = np.nonzero(np.diff(groups, prepend=-np.inf, append=np.inf) != 0)[0]
        for pick_idx in range(self.data.tot_orig):
            idx_to_discard = np.concatenate(
                (idx_to_discard, self.filter_locs_inpick(self.data.df_orig.iloc[groupjump[pick_idx]:groupjump[pick_idx + 1]])),
                axis=0
            )
            self.signals.tell_analysis_elem_done.emit(pick_idx + 1)
        df_filtered = self.data.df_orig.drop(labels=idx_to_discard, axis=0)
        df_filtered = df_filtered.reset_index(level=None, drop=True, inplace=False,
                                              col_level=0)
        end = _time.time()
        _lgr.info('Time of filtering step: %s s. %s of %s (%.2f%%) localizations discarded',
                end - start, len(idx_to_discard), n_loc_initial, 100 * len(idx_to_discard) / n_loc_initial)
        self.data.df_filt =  df_filtered
        
    def save_simpler_locs(self):
        """
        This function saves all SIMPLER localizations in a list of arrays (one for each origami)
        """
        all_orig_loc_list = []
        # helper array to find fast all localization pertaining to an origami
        groups = np.array(self.data.df_filt['group'])
        groupjump = np.nonzero(np.diff(groups, prepend=-np.inf, append=np.inf) != 0)[0]
        for pick_idx in range(self.data.tot_orig):
            pick_locs_arr = np.asarray(
                self.data.df_filt.iloc[
                    groupjump[pick_idx]:groupjump[pick_idx + 1],
                    self.data.df_filt.columns.get_indexer(['x', 'y', 'photons'])
                ], dtype=float
            )
            pick_locs_arr_nm = self.px_to_nm(pick_locs_arr, self.params.px_size_nm)
            all_orig_loc_list.append(pick_locs_arr_nm)
        self.data.simpler_locs = SIMPLERLocalizations(all_orig_loc_list)
        
    def clustering_xyn(self):
        """
        This function use GMM to cluster localizations in 3D (x, y and N space).
        It repeats the clustering with decreasing number of clusters and compares the result through BIC.
        If an origami is better clusterized by a number of cluster different from the expected one,
        it is discarded.
        """
        start = _time.time()
        # variables needed to discard origamis not passing the clusterzation test
        n_orig_discarded = 0
        idx_todiscard = []
        # here we will store all data relative to the clusterization result
        all_orig_loc_list = []
        clust_means_list = []
        clust_covs_list = []
        # helper array to find fast all localization pertaining to an origami
        groups = np.array(self.data.df_filt['group'])
        groupjump = np.nonzero(np.diff(groups, prepend=-np.inf, append=np.inf) != 0)[0]
        for pick_idx in range(self.data.tot_orig):
            pick_kept = True
            for n_clust in range(self.params.n_clust_exp, 0, -1):
                gmm = GaussianMixture(n_components=n_clust, covariance_type='full', n_init=1, init_params='k-means++')
                gmm.fit(self.data.simpler_locs.all_orig_loc_list[pick_idx])
                last_bic = gmm.bic(self.data.simpler_locs.all_orig_loc_list[pick_idx])
                if n_clust == self.params.n_clust_exp: # compute BIC for the expected number of clusters
                    ref_bic = last_bic
                    clust_means, clust_covs = self.reorder_clust(gmm.means_, gmm.covariances_)
                # now we decrease the number of clusters and as soon as one gives better result, we discard the origami and exit the loop
                elif last_bic < ref_bic:
                    idx_todiscard += [idx for idx in range(groupjump[pick_idx], groupjump[pick_idx + 1])]
                    n_orig_discarded += 1
                    pick_kept = False
                    break
            if pick_kept:
                all_orig_loc_list.append(self.data.simpler_locs.all_orig_loc_list[pick_idx])
                clust_means_list.append(clust_means)
                clust_covs_list.append(clust_covs)
            self.signals.tell_analysis_elem_done.emit(pick_idx + 1)
            
        self.data.tot_orig_after_clust = self.data.tot_orig - n_orig_discarded
        df_after_clust = self.data.df_filt.drop(labels=idx_todiscard, axis=0)
        df_after_clust = df_after_clust.reset_index(level=None, drop=True, inplace=False,
                                              col_level=0)
        
        self.data.simpler_locs = SIMPLERLocalizations(all_orig_loc_list)
        self.data.cluster_res = ClusterResults(clust_means_list, clust_covs_list)
        end = _time.time()
        _lgr.info('Time of clustering step: %s s. %s of %s (%.2f%%) origamis discarded',
                end - start, n_orig_discarded, self.data.tot_orig, 100 * n_orig_discarded / self.data.tot_orig)
        self.data.df_after_clust = df_after_clust
        
    @staticmethod
    def reorder_clust(means, sigmas):
        """
        This method reorders in descending order tuples of means and sigmas based on the mean of the last coordinate
        (number of photons). It is used to order clusters from bottom to top
        """
        return list(zip(*sorted(zip(means, sigmas), key=lambda pair: -pair[0][2])))
    
    @staticmethod
    def px_to_nm(arr_toconv, px_size_nm):
        """
        This function converts x and y coordinates of a given array from px to nm
        """
        arr_toconv[:, 0] = arr_toconv[:, 0]*px_size_nm
        arr_toconv[:, 1] = arr_toconv[:, 1]*px_size_nm
        return arr_toconv
    
    @pyqtSlot()
    def do_filt(self):
        """
        this function calls one by one all the filtering steps (filtering based on kinetics and SIMPLER localization filtering)
        """
        self.signals.tell_analysis_step_start.emit(AnalysisStatus.KIN_FILT, self.data.tot_picks)
        self.filter_orig()
        self.signals.tell_analysis_step_start.emit(AnalysisStatus.SIMPLER_FILT, self.data.tot_orig)
        self.filter_data()
        self.save_simpler_locs()
        self.signals.tell_filt_done.emit()
        
    @pyqtSlot()
    def do_clust(self):
        """
        This function call the clusterization function
        """
        self.signals.tell_analysis_step_start.emit(AnalysisStatus.SITE_CLUST, self.data.tot_orig)
        self.clustering_xyn()
        self.signals.tell_clust_done.emit()
        
if __name__=="__main__":
    filepath_str = r"X:\messdaten\Giovanni_A\SIMPLER\260313\Rifle_4pts_R2_40gain_500pMCy3B_200mW_100ms_23TIRF\R2\R2_2_MMStack_Pos0.ome_locs_picked_standing.hdf5"
    data_path = Path(filepath_str)
    metadata_path = data_path.parent / Path(data_path.stem + ".yaml")
    analysis_worker = AnalysisWorker(data_path, metadata_path)
    analysis_worker.do_filt()