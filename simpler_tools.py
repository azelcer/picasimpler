# -*- coding: utf-8 -*-
"""
"""
import time as _time
import pathlib as _pathlib
from dataclasses import dataclass
import h5py
import numpy as np
from scipy.spatial import distance as _distance, KDTree
from scipy.cluster import hierarchy
import yaml
import logging as _lgn
import warnings as _warnings
from sklearn import cluster 
from sklearn.cluster import DBSCAN as _DBSCAN, KMeans as _KMeans
from scipy.ndimage import map_coordinates


_lgn.basicConfig()
_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.INFO)


class FluoEvent:
    def __init__(self, idx: int, run_list: tuple[int, int]):  # frame, loc
        self._idx = idx
        chk_diff = [_[0] for _ in run_list]
        if not np.all(np.diff(chk_diff) == 1):
            raise ValueError(f"Frames no consecutivos: {run_list}")
        self._initial_frame = run_list[0][0]
        self._final_frame = run_list[-1][0]
        self._localization_list = [_[1] for _ in run_list]
        # self._center = np.average()

    def __repr__(self):
        return f"{self.__class__.__name__}(f{self._initial_frame}-f{self._final_frame})"

    def calculate_center(self, data: np.ndarray):
        """Calculate center."""
        df = data[self._localization_list]
        self._center = (np.average(df["x"]), np.average(df["y"]),)
        self._desv = (np.std(df["x"]), np.std(df["y"]),)

    @property
    def idx(self):
        return self._idx

    @property
    def center(self):
        return self._center

    @property
    def std(self):
        return self._desv

    @property
    def length(self):
        return self._final_frame - self._initial_frame + 1


# No fuzz about strings
def _h5py_dataset2ndarray(ds: h5py.Dataset) -> np.ndarray:
    # new_dtype_list = arr.dtype.descr + [('score', 'f4')]
    dt = ds.dtype
    rv = np.array(ds)
    fields_to_add = []
    if 'z' in dt.names:
        _lgr.info("ya tiene z")
    else:
        fields_to_add.append(("z", '<f4', np.nan,))
    # fields_to_add.append(("valid", '?', False,))
    if fields_to_add:
        # names, dtypes, fill_v = zip(*fields_to_add)
        # print(names, dtypes, fill_v)
        # rv = append_fields(rv, names, [[],]*len(names), dtypes, fill_value=fill_v, usemask=False)
        # https://stackoverflow.com/questions/25427197/numpy-how-to-add-a-column-to-an-existing-structured-array
        n_dt = dt.descr + [(name, dtype) for name, dtype, _ in fields_to_add]
        n_rv = np.empty((rv.shape[0],), dtype=n_dt)
        for name in dt.names:
            n_rv[name] = rv[name]
        for name, _, value in fields_to_add:
            n_rv[name] = value
        rv = n_rv
    return rv


def filter_data(
    data: np.ndarray, radius_threshold: float, px_size: float
) -> np.ndarray:
    """Filter localizations according to SIMPLER criteria.

    Parameters
    ----------
        data: numpy.ndarray
            Structured array as obtained from picasso
        radius_threshold: float
            Maximun radius in nm for two succesive localizations to be considered
            the same
        px_size: float
            Pixel size, in nm

    Returns
    -------
        Array of indices of records to discard
    """
    r_th_sq = (radius_threshold / px_size) ** 2
    start = _time.time()
    frames = np.array(data["frame"])
    xy = np.column_stack((np.transpose(data["x"]), np.transpose(data["y"])))
    discard_yn = np.ones((len(xy),), dtype=np.uint64)
    framejump = np.nonzero(
        np.diff(frames, prepend=-np.inf, append=np.inf) != 0
    )[0]
    distance_next = None
    for idxframe in range(1, len(framejump) - 2):
        prevframe = frames[framejump[idxframe - 1]]
        nextframe = frames[framejump[idxframe + 1]]
        frame = frames[framejump[idxframe]]
        if frame + 1 != nextframe:
            distance_next = None
            continue
        if frame - 1 != prevframe:
            continue
        f_slice = slice(framejump[idxframe], framejump[idxframe + 1])
        prev_slice = slice(framejump[idxframe - 1], framejump[idxframe])
        next_slice = slice(framejump[idxframe + 1], framejump[idxframe + 2])
        if distance_next is not None:
            distance_prev = distance_next.T
        else:
            distance_prev = _distance.cdist(
                xy[f_slice], xy[prev_slice], "sqeuclidean"
            )
        distance_next = _distance.cdist(
            xy[f_slice], xy[next_slice], "sqeuclidean"
        )
        has_prev = np.any(distance_prev < r_th_sq, axis=1)
        has_next = np.any(distance_next < r_th_sq, axis=1)
        discard_yn[f_slice][np.logical_and(has_prev, has_next)] = 0
    idx_to_discard = np.where(discard_yn == 1)[0]
    end = _time.time()
    _lgr.info(
        "Time of filtering step: %s s. %s of %s (%.2f%%) localizations discarded",
        end - start,
        len(idx_to_discard),
        len(xy),
        100 * len(idx_to_discard) / len(xy),
    )
    return idx_to_discard


def group_events(
    data: np.ndarray, radius_threshold: float, px_size: float
) -> list[FluoEvent]:
    """Group localizations by events.

    Discards events of lenght 1

    Parameters
    ----------
        data: numpy.ndarray
            Structured array as obtained from picasso
        radius_threshold: float
            Maximun radius in nm for two succesive localizations to be considered
            the same
        px_size: float
            Pixel size, in nm

    Returns
    -------
        List of events
    """
    r_th_sq = (radius_threshold / px_size) ** 2
    start = _time.time()
    frames = np.array(data["frame"])
    xy = np.column_stack((np.transpose(data["x"]), np.transpose(data["y"])))
    framejump = np.nonzero(
        np.diff(frames, prepend=-np.inf, append=np.inf) != 0
    )[0]
    runs: list[FluoEvent] = []
    n_evt = 0
    prev_run = [[] for _ in range(framejump[1] - framejump[0])]
    for idxframe in range(0, len(framejump) - 2):
        nextframe = frames[framejump[idxframe + 1]]
        start_frame_idx = framejump[idxframe]
        frame = frames[start_frame_idx]
        # FIXME: implementar esto
        if frame + 1 != nextframe:  # there are empty frames!
            # print("No hay frame", frame+1)
            next_slice = slice(0, 0, 1)
        else:
            next_slice = slice(framejump[idxframe + 1], framejump[idxframe + 2])
        f_slice = slice(start_frame_idx, framejump[idxframe + 1])
        next_run = [[] for _ in range(next_slice.stop - next_slice.start)]
        distance_next = _distance.cdist(
            xy[f_slice], xy[next_slice], "sqeuclidean"
        ) < r_th_sq
        # next_neighbours = distance_next.sum(axis=0)
        this_neighbours = distance_next.sum(axis=1)
        if any(this_neighbours > 1):
            print("Ambiguedad en frame", idxframe)
        # primero cerremos las anteriores
        for idx, lista in enumerate(prev_run):  # los indices de prev_run
            if lista:
                if this_neighbours[idx]:
                    next_run[np.argmax(distance_next[idx])] = lista
                else:
                    lista.append((frame, start_frame_idx + idx,))  # pegar loc actual
                    runs.append(FluoEvent(n_evt, lista))
                    n_evt += 1
        for idx, s in enumerate(this_neighbours):
            # print(f"la localización {start_frame_idx+idx} ({data[f_slice][idx]})"
            #       f"{'si' if s else 'no'} tiene vecinos")
            if s:
                next_run[np.argmax(distance_next[idx])].append((frame, start_frame_idx + idx,))
        prev_run = next_run
    for idx, lista in enumerate(prev_run):  # los indices de prev_run
        if lista:
            lista.append((nextframe, framejump[idxframe + 1] + idx,))  # pegar loc actual
            runs.append(FluoEvent(n_evt, lista))
            n_evt += 1
    for r in runs:
        r.calculate_center(data)
    end = _time.time()
    _lgr.info(
        "Time of filtering step: %s s. %s runs found with %s (%.2f%%) localizations",
        end - start,
        n_evt,
        sum(_.length for _ in runs),
        sum(_.length for _ in runs) / len(data) * 100
    )
    return runs


def get_intensity(int_map: np.ndarray, locations: np.ndarray, px_size: float):
    """Return excitation intensity (N0) at location point.

    Parameters
    ----------
    int_map: 2D numpy.ndarray
        Excitation intensity at each point
    locations: numpy.ndarray[n, 2]
        list of x,y locations
        WARNING: ver localización del 0, 0  en las localizaciones y el mapa
    px_size: float
        pixel size to map locations to indices
    """
    px_locations = (
        locations / px_size
    ) - 0.5  # Ojo extrapolación y localización del 0
    return map_coordinates(
        int_map, [px_locations[:, 0], px_locations[:, 1]], mode="nearest"
    )


def calculate_z(
    data: np.ndarray, alpha: float, df: float, N0: int
) -> np.ndarray:
    """Calculate z according to SIMPLER criteria.

    We use fixed a N0 for all the image now.
    """
    start = _time.time()
    # N0 = _calculate_N0(data, params)
    max_photons = np.max(data["photons"])
    min_alpha = 1 - (np.min(data["photons"]) / max_photons)
    if alpha < min_alpha:
        _lgr.warning(
            "Some location intensities are below non-evanescent excitation expected intensity"
        )
    if N0 < max_photons:
        _lgr.warning("Some location intensities are above z=0 intensity")
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        rv = df * np.log(alpha / (data["photons"] / N0 - (1 - alpha)))
    _lgr.info("Time of Z calculation step: %s s", _time.time() - start)
    return rv


def cluster_xy_positions(
    data: np.ndarray,
    dist_threshold: float,
    px_size: float,
    min_N=15,
) -> (_DBSCAN, np.ndarray):
    """Clusters locations for origami calibration.

    So far all results are contained on the DBscan ibject
    """
    start = _time.time()
    eps = (dist_threshold / px_size) ** 2
    xy = np.column_stack((np.transpose(data["x"]), np.transpose(data["y"])))
    rv = _DBSCAN(
        eps=eps, min_samples=min_N, metric="sqeuclidean", n_jobs=-1
    ).fit(xy)
    _lgr.info("clusters found: %s", len(set(rv.labels_) - {-1}))
    _lgr.info(
        "locations assigned: %s out of %s",
        len(rv.core_sample_indices_),
        len(xy),
    )
    _lgr.info("Time of clustering: %s s", _time.time() - start)
    return rv, xy


def N_clusters(origamis: _DBSCAN, data: np.ndarray) -> list[_KMeans]:
    """Subcluster each cluster by N.

    Uses k-means
    """
    t0 = _time.time()
    N_FLUO = 4  # fluoroforos por origami
    labels = set(origamis.labels_) - {-1}
    markers = [
        _KMeans(N_FLUO).fit(  # TODO: avoid convertion to array
            np.array(data["photons"][cluster.labels_ == _]).reshape(-1, 1)
        )
        for _ in labels
    ]
    _lgr.info(
        "Intensity clustered %s XY clusters in %s seconds",
        len(markers),
        _time.time() - t0,
    )
    return markers


def xy_from_N(
    clusters: list[_KMeans], positions: list[np.ndarray]
) -> np.ndarray:
    """Calculate subclusters xy mean positions."""
    N_FLUO = 4
    centers = np.ndarray((len(clusters), N_FLUO, 2))
    for idx, (c, p) in enumerate(zip(clusters, positions)):
        labels = set(c.labels_) - {-1}
        centers[idx] = [np.average(p[c.labels_ == _], axis=0) for _ in labels]
    return centers


def calibrate_origami(data):
    # encontrar muestras colocalizadas (con un radio apto angulos)
    origamis, all_positions = cluster_xy_positions(
        data, dist_threshold=30, px_size=133
    )
    n_clus = N_clusters(origamis, data)
    # Verificar calidad de clusters y filtrar
    ...
    cluster_labels = set(cluster.labels_) - {-1}
    cluster_filters = [(cluster.labels_ == _) for _ in cluster_labels]
    xy = np.column_stack((np.transpose(data["x"]), np.transpose(data["y"])))
    clustered_xy = [np.array(xy[cf]) for cf in cluster_filters]

    # Ver el ángulo y dirección en XY con ese corrimiento sacar el ángulo.
    ...


@dataclass
class SimplerAnalysisParameters:
    alpha: float
    df: float
    N0: int  # Ojo cuando hagamos mapeado


class SIMPLERData:
    """Manages and processes SIMPLER data obtained from PICASSO.

    Meter callbacks despues para que todo lo que sea mas o menos lento (hasta
    la carga es lenta)
    """

    def __init__(
        self,
        file_name: str | _pathlib.Path,
        yaml_file_name: str | _pathlib.Path | None = None,
    ):
        file_name = _pathlib.Path(file_name)
        with h5py.File(file_name, "r") as store:
            self.data = _h5py_dataset2ndarray(store["locs"])
        if not yaml_file_name:
            yaml_file_name = file_name.with_suffix(".yaml")
        yaml_file_name = _pathlib.Path(yaml_file_name)
        with open(yaml_file_name, "r") as info_file:
            self.info = list(yaml.load_all(info_file, Loader=yaml.FullLoader))
        self._orig_fname = file_name
        self._orig_yaml_name = yaml_file_name
        self.pixel_size = self.info[1]["Pixelsize"]
        self._runs: list[FluoEvent] = None
        self._filtered_runs: list[FluoEvent] = None
        self._sites: list[list[FluoEvent]] = None
        self._custom_info = {}  # info to add to the YAML file

    def save(self, fname: str | _pathlib.Path):
        fname = _pathlib.Path(fname)
        with h5py.File(fname, "w") as store:  # "x"
            locs = store.create_dataset("locs", data=self.data, dtype=self.data.dtype)
        yaml_file_name = fname.with_suffix(".yaml")
        with open(yaml_file_name, "w") as info_file:
            yaml.dump_all(self.info, info_file, default_flow_style=False)

    # def filter_data(self, params: SimplerAnalysisParameters):
    #     """Viejo, ahora usamos otro"""
    #     idx_to_discard = filter_data(self.data, params.max_dist, self.pixel_size)
    #     data_filter = np.ones((self.data.shape[0],), dtype=bool)
    #     data_filter[idx_to_discard] = False
    #     self._out_idx = idx_to_discard
    #     self.data["valid"] = data_filter
    #     self._filtered_data = self.data[data_filter]

    def group_events(self, distance: float):
        self._runs = group_events(self.data, abs(distance), self.pixel_size)
        self._filtered_runs = self._runs
        self._sites = None
        self._analyze_events()

    def _analyze_events(self):
        self._events_positions = np.array([_.center for _ in self._filtered_runs])
        self._events_size = np.array([(_.std[0]**2 + _.std[1]**2)**.5 for _ in self._filtered_runs])

    def filter_events(self, min_lenght: int = 2, max_length: int = None):
        if not self._runs:
            return
        self._filtered_runs = [_ for _ in self._runs if
                               min_lenght <= _.length <= (max_length or np.inf)]
        self._analyze_events()
        self._sites = None
        _lgr.info("Filtered by lengths between %s and %s. %s events remaining",
                  min_lenght, max_length or np.inf, len(self._filtered_runs)
                  )

    def group_sites(self, distance: float, method: str = "single"):
        """Indices of runs belonging to the same site."""
        if not self._runs:
            return []
        # scipy first
        t0 = _time.time()
        if hasattr(hierarchy, method):
            func = getattr(hierarchy, method)
            pd = _distance.pdist(self._events_positions)
            Z = func(pd)
            clst = hierarchy.fcluster(Z, distance / self.pixel_size, criterion='distance')
            nclust = clst.max()
            sites = []
            # https://stackoverflow.com/questions/30003068/how-to-get-a-list-of-all-indices-of-repeated-elements-in-a-numpy-array
            for c in range(1, nclust + 1):  # cluster numbering starts at 1
                sites.append(np.nonzero(clst == c)[0])
            self._sites = [[self._filtered_runs[_] for _ in s] for s in sites]
        else:
            if method == "OPTICS":  # TODO: use match
                clust = cluster.OPTICS(min_samples=5, max_eps=distance / self.pixel_size,)
                clust.fit(self._events_positions)
                nclust = clust.labels_.max()
                sites = []
                # TODO: acceder via labels_
                # https://stackoverflow.com/questions/30003068/how-to-get-a-list-of-all-indices-of-repeated-elements-in-a-numpy-array
                for c in range(nclust):  # cluster numbering starts at 0
                    sites.append(np.nonzero(clust.labels_ == c)[0])
                self._sites = [[self._filtered_runs[_] for _ in s] for s in sites]
            elif method == "DBSCAN":  # TODO: use match
                clust = cluster.DBSCAN(min_samples=5, eps=distance / self.pixel_size,)
                clust.fit(self._events_positions)
                nclust = clust.labels_.max()
                sites = []
                # TODO: acceder via labels_
                # https://stackoverflow.com/questions/30003068/how-to-get-a-list-of-all-indices-of-repeated-elements-in-a-numpy-array
                for c in range(nclust):  # cluster numbering starts at 0
                    sites.append(np.nonzero(clust.labels_ == c)[0])
                self._sites = [[self._filtered_runs[_] for _ in s] for s in sites]
        _lgr.info("Clustering with %s took %s s and found %s sites.",
                  method, _time.time() - t0, len(self._sites))

    def get_events(self):
        if self._runs is None:
            raise ValueError("No grouping has been performed yet")
        return self._filtered_runs

    def get_grouped_indices(self):
        if self._runs is None:
            raise ValueError("No grouping has been performed yet")
        if not self._filtered_runs:
            return np.empty((0,), dtype=int)
        return np.concatenate([_._localization_list for _ in self._filtered_runs])

    def get_grouped_locations(self):
        if self._runs is None:
            return np.array([], dtype=self.data.dtype)
        return self.data[self.get_grouped_indices()]

    def get_events_locations(self):
        if self._runs is None:
            return np.empty((0, 2,))
        return self._events_positions

    def get_events_sizes(self):
        if self._runs is None:
            return np.empty((0,))
        return self._events_size

    def get_sites(self) -> list[list[FluoEvent]]:
        if self._sites is None:
            return []
        return self._sites

    def get_ungrouped_filter(self):
        rv = np.ones_like(self.data, dtype=bool)
        if self._runs:
            rv[self.get_grouped_indices()] = False
        return rv

    def get_ungrouped_locations(self):
        return self.data[self.get_ungrouped_filter()]

    def get_unfilterred_data(self):
        return self.data

    # def get_filterred_data(self):
    #     return self._filtered_data

    def get_column_names(self):
        return self.data.dtype.names

    def calculate_z(self, params: SimplerAnalysisParameters):
        if not self._runs:
            _lgr.warning("No hay data para calcular Z")
            return
        # trim ends
        # TODO: ver si aplicar a TODOS los runs o sólo a los agrupados en sites
        # for r in self._runs:
        #     print(r._localization_list)
        #     for l in r._localization_list: 
        #         if type(l) is not np.int64:
        #             print(type(l))
        # print([_._localization_list[1:-1] for _ in self._runs])
        valid_loc = np.concatenate([_._localization_list[1:-1] for _ in self._runs if _._localization_list[1:-1]], dtype=np.int64)
        # print(valid_loc)
        z = calculate_z(self.data[valid_loc], params.alpha, params.df, params.N0)
        self.data["z"][valid_loc] = z
        # print(self.data[valid_loc]["z"])
        # print(z)

    def cluster_origamis(self, max_dist: float):
        ...
        # max_dist podría calcularse con un helper que de la distancia maxima en funcion del angulo.
        cluster, xy = cluster_xy_positions(
            data_filtered, cluster_threshold, px_size
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection, EllipseCollection

    filename = _pathlib.Path(
        "/home/azelcer/Dropbox/2024/simpler/example_spectrin_large.hdf5"
    )
    # filename = _pathlib.Path(
    #     "/home/azelcer/Dropbox/2025/simpler/rifleSIMPLER_3ptsR3_1ptR4_200pM_Cy3B_100mW_bufferC_gain100_1_MMStack_Pos0.ome_locs.hdf5"
    # )
    filename = _pathlib.Path(
        "/home/azelcer/Dropbox/2025/simpler/rifleSIMPLER_3ptsR3_1ptR4_200pM_Cy3B_300mW_bufferC_gain50_50ms_highTIRF_2_MMStack_Pos0.ome_locs.hdf5"
    )

    start = _time.time()
    with h5py.File(filename, "r") as store:
        data = _h5py_dataset2ndarray(store["locs"])
    yaml_file = filename.with_suffix(".yaml")
    with open(yaml_file, "r") as info_file:
        info = list(yaml.load_all(info_file, Loader=yaml.FullLoader))
    
    xxx = SIMPLERData(filename)
    xxx.save("/tmp/kkk.hdf5")
    fff = SIMPLERData("/tmp/kkk.hdf5")
    
    
    px_size = info[1]["Pixelsize"]
    radius_threshold = 75  # nm
    runs = group_events(data, 5, px_size)
    # idx_to_discard = filter_data(data, radius_threshold, px_size)
    # data_filter = np.ones((data.shape[0],), dtype=bool)
    # data_filter[idx_to_discard] = False
    # data_filtered = data[data_filter]
    # _lgr.info(
    #     "minimum alpha is: %s",
    #     1 - (np.min(data["photons"]) / np.max(data["photons"])),
    # )
    # z = calculate_z(data_filtered, 0.95, 100, np.max(data["photons"]))

    # plt.scatter(*zip(*(_.center for _ in runs)))
    # plt.scatter(*zip(*((_["x"], _["y"]) for _ in data)), marker='.')
