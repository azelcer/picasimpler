# -*- coding: utf-8 -*-
"""
Functions to process and calibrate SIMPLER measurements.

The functions are expected to interface with the sofware Picasso, as it is
widely used for SML. Nevertheless, the functions are general enough to be
used with other software with minumum effort.


@author: aszalai, azelcer
"""
from dataclasses import dataclass
import h5py
import numpy as np
# from numpy.lib.recfunctions import append_fields
from scipy.spatial import distance, ConvexHull, KDTree
from scipy.cluster import hierarchy
import yaml
import logging as _lgn
import warnings as _warnings
from sklearn.cluster import DBSCAN as _DBSCAN, KMeans as _KMeans
from scipy.ndimage import map_coordinates

# The following imports are used only for development.
import pathlib as _pathlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import time as _time


_lgn.basicConfig()
_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.INFO)

filename = _pathlib.Path(
    "/home/azelcer/Dropbox/2024/simpler/example_spectrin_large.hdf5"
)
# filename = _pathlib.Path(
#     "/home/azelcer/Dropbox/2025/simpler/rifleSIMPLER_3ptsR3_1ptR4_200pM_Cy3B_100mW_bufferC_gain100_1_MMStack_Pos0.ome_locs.hdf5"
# )
filename = _pathlib.Path(
    "/home/azelcer/Dropbox/2025/simpler/rifleSIMPLER_3ptsR3_1ptR4_200pM_Cy3B_300mW_bufferC_gain50_50ms_highTIRF_2_MMStack_Pos0.ome_locs.hdf5"
)


class FluoEvent:
    def __init__(self, run_list: tuple[int, int]):  # frame, loc
        chk_diff = [_[0] for _ in run_list]
        if not np.all(np.diff(chk_diff) == 1):
            raise ValueError(f"Frames no consecutivos: {run_list}")
            # print(ValueError(f"Frames no consecutivos: {run_list}"))
        self._initial_frame = run_list[0][0]
        self._final_frame = run_list[-1][0]
        self._localization_list = [_[1] for _ in run_list]
        # self._center = np.average()

    def __repr__(self):
        # print(self.__dir__())
        return f"{self.__class__.__name__}(f{self._initial_frame}-f{self._final_frame})"

    def calculate_center(self, data: np.ndarray):
        """Calculate center."""
        df = data[self._localization_list]
        self._center = (np.average(df["x"]), np.average(df["y"]),)
        self._desv = (np.std(df["x"]), np.std(df["y"]),)

    @property
    def center(self):
        return self._center

    @property
    def std(self):
        return self._desv

    @property
    def length(self):
        return self._final_frame - self._initial_frame


# No fuzz aboutstrings
def _h5py_dataset2ndarray(ds: h5py.Dataset) -> np.ndarray:
    # new_dtype_list = arr.dtype.descr + [('score', 'f4')]
    dt = ds.dtype
    rv = np.array(ds)
    fields_to_add = []
    if 'z' in dt.names:
        _lgr.info("ya tiene z")
    else:
        fields_to_add.append(("z", '<f4', np.nan,))
    fields_to_add.append(("valid", '?', False,))
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
            distance_prev = distance.cdist(
                xy[f_slice], xy[prev_slice], "sqeuclidean"
            )
        distance_next = distance.cdist(
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


def remove_unespecific(
    data: np.ndarray, radius_threshold: float, px_size: float
) -> np.ndarray:
    """Filter localizations that seem to be non-specific adsorption.

    Parameters
    ----------
        data: numpy.ndarray
            Structured array as obtained from picasso
        radius_threshold: float
            Maximun radius in nm for two localizations to be considered
            the same
        px_size: float
            Pixel size, in nm

    Returns
    -------
        Array of booleans with indices of records to keep in True
    """
    r_th = radius_threshold / px_size
    start = _time.time()
    xy = np.column_stack((np.transpose(data["x"]), np.transpose(data["y"])))
    kdt = KDTree(xy)
    rv = kdt.query(xy, [2,], distance_upper_bound=r_th)[0]
    rv = rv.reshape(rv.shape[0]) != np.inf
    end = _time.time()
    n_discarded = len(rv) - rv.sum()
    _lgr.info(
        "Time of unespecific filtering step: %s s. %s of %s (%.2f%%) localizations discarded",
        end - start,
        n_discarded,
        len(xy),
        100 * n_discarded / len(xy),
    )
    return rv


def group_events(
    data: np.ndarray, radius_threshold: float, px_size: float
) -> np.ndarray:
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
    runs = []
    r_th_sq = (radius_threshold / px_size) ** 2
    start = _time.time()
    frames = np.array(data["frame"])
    xy = np.column_stack((np.transpose(data["x"]), np.transpose(data["y"])))
    framejump = np.nonzero(
        np.diff(frames, prepend=-np.inf, append=np.inf) != 0
    )[0]
    runs = []
    prev_run = [[] for _ in range(framejump[1] - framejump[0])]
    for idxframe in range(0, len(framejump) - 2):
        nextframe = frames[framejump[idxframe + 1]]
        start_frame_idx = framejump[idxframe]
        frame = frames[start_frame_idx]
        # FIXME: implementar esto
        # if frame + 1 != nextframe:  # there are empty frames
        #     # flush
        #     continue
        f_slice = slice(start_frame_idx, framejump[idxframe + 1])
        next_slice = slice(framejump[idxframe + 1], framejump[idxframe + 2])
        next_run = [[] for _ in range(next_slice.stop - next_slice.start)]
        distance_next = distance.cdist(
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
                    runs.append(FluoEvent(lista))
        for idx, s in enumerate(this_neighbours):
            # print(f"la localización {start_frame_idx+idx} ({data[f_slice][idx]})"
            #       f"{'si' if s else 'no'} tiene vecinos")
            if s:
                next_run[np.argmax(distance_next[idx])].append((frame, start_frame_idx + idx,))
        prev_run = next_run
    for idx, lista in enumerate(prev_run):  # los indices de prev_run
        if lista:
            lista.append((nextframe, framejump[idxframe + 1] + idx,))  # pegar loc actual
            runs.append(FluoEvent(lista))
    for r in runs:
        r.calculate_center(data)
    end = _time.time()
    _lgr.info(
        "Time of filtering step: %s s. %s runs found with %s (%.2f%%) localizations",
        end - start,
        len(runs),
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


# def clusters_centers(cluster: _DBSCAN, data):
#     """Calculate clusters means."""
#     labels = set(cluster.labels_) - {-1}
#     centers = np.ndarray((len(labels), 2))
#     xy = np.column_stack((np.transpose(data['x']), np.transpose(data['y'])))
#     for idx, l in enumerate(labels):
#         centers[idx] = np.average(xy[cluster.labels_ == l], axis=0)
#     return centers


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
    max_dist: float
    alpha: float
    df: float
    N0: int  # Ojo cuando hagamos mapeado


class SIMPLERData:
    """Manages and processes SIMPLER data obtained from PICASSO.

    Meter threads despues para que todo lo que sea mas o menos lento (hasta la carga es lenta)
    funcione en bckg y con callbacks
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
        with open(yaml_file_name, "r") as info_file:
            self.info = list(yaml.load_all(info_file, Loader=yaml.FullLoader))
        self.pixel_size = self.info[1]["Pixelsize"]
        self._filtered_data = np.empty_like(self.data)

    def filter_data(self, params: SimplerAnalysisParameters):
        idx_to_discard = filter_data(self.data, params.max_dist, self.pixel_size)
        data_filter = np.ones((self.data.shape[0],), dtype=bool)
        data_filter[idx_to_discard] = False
        self._out_idx = idx_to_discard
        self.data["valid"] = data_filter
        self._filtered_data = self.data[data_filter]

    def get_unfilterred_data(self):
        return self.data

    def get_filterred_data(self):
        return self._filtered_data

    def get_column_names(self):
        return self.data.dtype.names

    def calculate_z(self, params: SimplerAnalysisParameters):
        if len(self._filtered_data) == 0:
            _lgr.warning("No hay data para calcular Z")
            return
        z = calculate_z(self._filtered_data, params.alpha, params.df, params.N0)
        self._filtered_data["z"] = z 

    def cluster_origamis(self, max_dist: float):
        ...
        # max_dist podría calcularse con un helper que de la distancia maxima en funcion del angulo.
        cluster, xy = cluster_xy_positions(
            data_filtered, cluster_threshold, px_size
        )


if __name__ == "__main__":
    start = _time.time()
    with h5py.File(filename, "r") as store:
        data = _h5py_dataset2ndarray(store["locs"])
    yaml_file = filename.with_suffix(".yaml")
    with open(yaml_file, "r") as info_file:
        info = list(yaml.load_all(info_file, Loader=yaml.FullLoader))
    px_size = info[1]["Pixelsize"]
    radius_threshold = 75  # nm
    # idx_unes = remove_unespecific(data, radius_threshold, px_size)
    runs = group_events(data, 2, px_size)
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
    t0 = _time.time()
    positions = np.array([_.center for _ in runs])
    pd = distance.pdist(positions)
    Z = hierarchy.single(pd)
    # Z = hierarchy.ward(pd)
    clst = hierarchy.fcluster(Z, .5, criterion='distance')
    nclust = clst.max()
    origamis = []
    for c in range(1, nclust+1):  # cluster numbering starts at 0
        origamis.append(np.nonzero(clst == c)[0])
    print("Creados ", nclust, "grupos en ", _time.time()-t0, "s")
    plt.figure("grouped runs")
    plt.scatter(*positions.T)
    patches = []
    for origami in origamis:
        or_points = np.array([positions[_] for _ in origami])
        if len(origami) < 3:
            vertex = or_points
        else:
            ch = ConvexHull(or_points)
            vertex = ch.points[ch.vertices]
        patches.append(Polygon(vertex, closed=True, color="r"))
    # colors = 100 * np.random.rand(len(patches))
    p = PatchCollection(patches, alpha=0.3)
    p.set_color("r")
    plt.gca().add_collection(p)


if False:
    positions = np.array([_.center for _ in runs])
    juntos = np.nonzero(distance.pdist(positions) < 20)[0]
    rows, cols = np.triu_indices(len(positions), 1)
    rows = rows[juntos]
    rows = np.concatenate((rows, [-1],))
    cols = cols[juntos]
    sets = [set() for _ in range(len(positions))]  # set o list?
    last_idx = 0
    for idx in range(len(positions)):
        origami = sets[idx]
        origami.add(idx)
        while rows[last_idx] == idx:
            origami |= sets[cols[last_idx]]
            sets[cols[last_idx]] = origami
            last_idx += 1
    stop()
    sets = set(tuple(_) for _ in sets)  # squash
    # from scipy.cluster.hierarchy import dendrogram, linkage

    # from matplotlib import pyplot as plt
    # Z = linkage(dist, 'ward')

    # fig = plt.figure(figsize=(25, 10))
    # dn = dendrogram(Z)

if False:
    # data_filtered['z'] = z
    cluster_threshold = 30  # la distancia si está 100% acostado es 30
    cluster, xy = cluster_xy_positions(
        data_filtered, cluster_threshold, px_size
    )
    plt.figure("coloreados")
    cluster_lbls = set(set(cluster.labels_) - {-1})
    centers = np.empty(
        (
            2,
            len(cluster_lbls),
        )
    )
    for idx, lbl in enumerate(cluster_lbls):
        _x, _y = tuple(zip(*xy[cluster.labels_ == lbl]))
        plt.scatter(_x, _y)
        centers[:, idx] = (np.average(_x), np.average(_y))
    out_file = filename.with_stem(filename.stem + "_frames_filtered")
    sa, saType = df_to_sarray(data_filtered)
    try:
        with h5py.File(out_file, "a") as f:
            f.create_dataset("locs", data=sa, dtype=saType)
        with open(
            filename.with_suffix(".yaml").with_stem(
                filename.stem + "_frames_filtered"
            ),
            "w",
        ) as file:
            yaml.dump_all(info, file, default_flow_style=False)
    except ValueError:
        _lgr.error("No puedo grabar, el archivo ya existe o algo así")
        # raise
    end = _time.time()
    _lgr.info("Script total time: %s s", end - start)
    # centros = clusters_centers(cluster, data_filtered)
    # plt.figure("dos")
    # plt.scatter(centros[:, 0], centros[:, 1], s=1)
    n_clus = N_clusters(cluster, data_filtered)
    plt.figure("centros")
    plt.scatter(*centers, s=1)

    plt.figure("todos")
    x = data_filtered["x"][cluster.core_sample_indices_]
    y = data_filtered["y"][cluster.core_sample_indices_]
    plt.scatter(x, y, s=1)
