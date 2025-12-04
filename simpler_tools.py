# -*- coding: utf-8 -*-
"""
Functions to process and calibrate SIMPLER measurements.

The functions are expected to interface with the sofware Picasso, as it is
widely used for SML. Nevertheless, the functions are general enough to be
used with other software with minumum effort.


@author: aszalai, azelcer
"""
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy.spatial import distance
import h5py
import yaml
import logging as _lgn
import warnings as _warnings
from sklearn.cluster import DBSCAN as _DBSCAN, KMeans as _KMeans
from scipy.ndimage import map_coordinates

# The following imports are used only for development.
import pathlib as _pathlib
import matplotlib.pyplot as plt
import time as _time


_lgn.basicConfig()
_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.INFO)

filename = _pathlib.Path(
    "/home/azelcer/Dropbox/2024/simpler/example_spectrin_large.hdf5"
)
filename = _pathlib.Path(
    "/home/azelcer/Dropbox/2025/simpler/rifleSIMPLER_3ptsR3_1ptR4_200pM_Cy3B_100mW_bufferC_gain100_1_MMStack_Pos0.ome_locs.hdf5"
)
filename = _pathlib.Path(
    "/home/azelcer/Dropbox/2025/simpler/rifleSIMPLER_3ptsR3_1ptR4_200pM_Cy3B_300mW_bufferC_gain50_50ms_highTIRF_2_MMStack_Pos0.ome_locs.hdf5"
)


def df_to_sarray(df):
    """
    Convert a pandas DataFrame object to a numpy structured array.
    Also, for every column of a str type, convert it into
    a 'bytes' str literal of length = max(len(col)).

    :param df: the data frame to convert
    :return: a numpy structured array representation of df
    
    TODO: Heredado. Revisar y ver si hace falta
    """

    def make_col_type(col_type, col):
        try:
            if "numpy.object_" in str(col_type.type):
                maxlens = col.dropna().str.len()
                if maxlens.any():
                    maxlen = maxlens.max().astype(int)
                    col_type = ("S%s" % maxlen, 1)
                else:
                    col_type = "f2"
            return col.name, col_type
        except:
            print(col.name, col_type, col_type.type, type(col))
            raise

    v = df.values
    types = df.dtypes
    numpy_struct_types = [
        make_col_type(types[col], df.loc[:, col]) for col in df.columns
    ]
    dtype = np.dtype(numpy_struct_types)
    z = np.zeros(v.shape[0], dtype)
    for i, k in enumerate(z.dtype.names):
        # This is in case you have problems with the encoding, remove the if branch if not
        try:
            if dtype[i].str.startswith("|S"):
                z[k] = df[k].str.encode("latin").astype("S")
            else:
                z[k] = v[:, i]
        except:
            print(k, v[:, i])
            raise

    return z, dtype


def filter_data(
    data: pd.DataFrame, radius_threshold: float, px_size: float
) -> np.ndarray:
    """Filter localizations according to SIMPLER criteria.

    Parameters
    ----------
        data: pandas.DataFrame
            Data as obtained from picasso
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
    data: pd.DataFrame, alpha: float, df: float, N0: int
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
    data: pd.DataFrame,
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


def N_clusters(origamis: _DBSCAN, data: pd.DataFrame) -> list[_KMeans]:
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
        with pd.HDFStore(file_name, "r") as store:
            self.data = store["locs"]
        if not yaml_file_name:
            yaml_file_name = file_name.with_suffix(".yaml")
        with open(yaml_file_name, "r") as info_file:
            self.info = list(yaml.load_all(info_file, Loader=yaml.FullLoader))
        self.pixel_size = self.info[1]["Pixelsize"]
        self._filtered_data = np.empty_like(self.data)

    def filter_data(self, params: SimplerAnalysisParameters):
        idx_to_discard = filter_data(self.data, params.max_dist, self.pixel_size)
        self._out_idx = idx_to_discard
        self._filtered_data = self.data.drop(labels=idx_to_discard, axis=0)

    def get_unfilterred_data(self):
        return self.data

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

    with pd.HDFStore(filename, "r") as store:
        data = store["locs"]
    yaml_file = filename.with_suffix(".yaml")
    with open(yaml_file, "r") as info_file:
        info = list(yaml.load_all(info_file, Loader=yaml.FullLoader))
    px_size = info[1]["Pixelsize"]
    radius_threshold = 75  # nm
    idx_to_discard = filter_data(data, radius_threshold, px_size)
    data_filtered = data.drop(labels=idx_to_discard, axis=0)
    data_filtered = data_filtered.reset_index(
        level=None, drop=True, inplace=False, col_level=0
    )
    _lgr.info(
        "minimum alpha is: %s",
        1 - (np.min(data["photons"]) / np.max(data["photons"])),
    )
    z = calculate_z(data_filtered, 0.95, 100, np.max(data["photons"]))
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
