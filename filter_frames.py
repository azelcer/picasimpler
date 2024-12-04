# -*- coding: utf-8 -*-
"""
Functions to process and calibrate SIMPLER measurements.

The functions are expected to interface with the sofware Picasso, as it is
widely used for SML. Nevertheless, the functions are general enough to be
used with other software with minumum effort.


@author: aszalai, azelcer
"""

import pandas as pd
import numpy as np
from scipy.spatial import distance
import h5py
import yaml
import logging as _lgn
import warnings as _warnings
from sklearn.cluster import DBSCAN as _DBSCAN, KMeans as _KMeans
# The following imports are used only for development.
import pathlib as _pathlib
import matplotlib.pyplot as plt
import time as _time
from tqdm import tqdm
from tools import fake_origami_data


_lgn.basicConfig()
_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.INFO)

filename = _pathlib.Path("/home/azelcer/Dropbox/2024/simpler/example_spectrin_large.hdf5")


def df_to_sarray(df):
    """
    Convert a pandas DataFrame object to a numpy structured array.
    Also, for every column of a str type, convert it into
    a 'bytes' str literal of length = max(len(col)).

    :param df: the data frame to convert
    :return: a numpy structured array representation of df
    """

    def make_col_type(col_type, col):
        try:
            if 'numpy.object_' in str(col_type.type):
                maxlens = col.dropna().str.len()
                if maxlens.any():
                    maxlen = maxlens.max().astype(int)
                    col_type = ('S%s' % maxlen, 1)
                else:
                    col_type = 'f2'
            return col.name, col_type
        except:
            print(col.name, col_type, col_type.type, type(col))
            raise

    v = df.values
    types = df.dtypes
    numpy_struct_types = [make_col_type(types[col], df.loc[:, col]) for col in df.columns]
    dtype = np.dtype(numpy_struct_types)
    z = np.zeros(v.shape[0], dtype)
    for (i, k) in enumerate(z.dtype.names):
        # This is in case you have problems with the encoding, remove the if branch if not
        try:
            if dtype[i].str.startswith('|S'):
                z[k] = df[k].str.encode('latin').astype('S')
            else:
                z[k] = v[:, i]
        except:
            print(k, v[:, i])
            raise

    return z, dtype


# angle = 67.5
# alpha = 0.9
# lambdaex = 532
# lambdaem = 582
# N0 = 16000 #R1 13500, R2 16000
# ni = 1.51
# ns = 1.33
# NA = 1.42


def filter_data(data: pd.DataFrame, radius_threshold: float, px_size: float) -> np.ndarray:
    """Filter localizations according to SIMPLER criteria.

    Returns
    -------
        Array of indices of records to discard
    """
    r_th_sq = (radius_threshold/px_size)**2
    start = _time.time()
    frames = np.array(data['frame'])
    xy = np.column_stack((np.transpose(data['x']), np.transpose(data['y'])))
    discard_yn = np.ones((len(xy),), dtype=np.uint64)
    framejump = np.nonzero(np.diff(frames, prepend=-np.inf, append=np.inf) != 0)[0]
    distance_next = None
    for idxframe in tqdm(range(1, len(framejump)-2)):
        prevframe = frames[framejump[idxframe-1]]
        nextframe = frames[framejump[idxframe+1]]
        frame = frames[framejump[idxframe]]
        if (frame + 1 != nextframe):
            distance_next = None
            continue
        if (frame - 1 != prevframe):
            continue
        f_slice = slice(framejump[idxframe], framejump[idxframe+1])
        prev_slice = slice(framejump[idxframe-1], framejump[idxframe])
        next_slice = slice(framejump[idxframe+1], framejump[idxframe+2])
        if distance_next is not None:
            distance_prev = distance_next.T
        else:
            distance_prev = distance.cdist(xy[f_slice], xy[prev_slice], 'sqeuclidean')
        distance_next = distance.cdist(xy[f_slice], xy[next_slice], 'sqeuclidean')
        has_prev = np.any(distance_prev < r_th_sq, axis=1)
        has_next = np.any(distance_next < r_th_sq, axis=1)
        discard_yn[f_slice][np.logical_and(has_prev, has_next)] = 0
    idx_to_discard = np.where(discard_yn == 1)[0]
    end = _time.time()
    _lgr.info('Time of filtering step: %s s. %s of %s (%.2f%%) localizations discarded',
              end-start, len(idx_to_discard), len(xy), 100*len(idx_to_discard)/len(xy))
    return idx_to_discard


def calculate_z(data: pd.DataFrame, alpha: float, df: float, N0: int) -> np.ndarray:
    """Calculate z according to SIMPLER criteria.

    We use fixed a N0 for all the image now.
    """
    start = _time.time()
    # N0 = _calculate_N0(data, params)
    max_photons = np.max(data['photons'])
    min_alpha = 1 - (np.min(data['photons']) / max_photons)
    if alpha < min_alpha:
        _lgr.warning("Some location intensities are below non-evanescent excitation expected intensity")
    if N0 < max_photons:
        _lgr.warning("Some location intensities are above z=0 intensity")
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        rv = df * np.log(alpha/(data['photons']/N0 - (1 - alpha)))
    _lgr.info('Time of Z calculation step: %s s', _time.time()-start)
    return rv


def cluster_xy_positions(data: pd.DataFrame, dist_threshold: float,
                         px_size: float) -> (_DBSCAN, np.ndarray):
    """Clusters locations for origami calibration."""
    start = _time.time()
    eps = (dist_threshold/px_size)**2
    xy = np.column_stack((np.transpose(data['x']), np.transpose(data['y'])))
    rv = _DBSCAN(eps=eps, min_samples=4, metric='sqeuclidean', n_jobs=-1).fit(xy)
    _lgr.info('clusters found: %s', len(set(rv.labels_) - {-1}))
    _lgr.info('locations assigned: %s out of %s', len(rv.core_sample_indices_), len(xy))
    _lgr.info('Time of clustering: %s s', _time.time()-start)
    return rv, xy


def N_clusters(origamis: _DBSCAN, data: pd.DataFrame) -> list[_KMeans]:
    """Subcluster each cluster by N.

    Uses k-means
    """
    N_FLUO = 4  # fluoroforos por origami
    labels = set(origamis.labels_) - {-1}
    markers = [_KMeans(N_FLUO).fit(  # TODO: avoid convertion to array
        np.array(data['photons'][cluster.labels_ == l]).reshape(-1, 1))
               for l in labels]
    return markers


# def clusters_centers(cluster: _DBSCAN, data):
#     """Calculate clusters means."""
#     labels = set(cluster.labels_) - {-1}
#     centers = np.ndarray((len(labels), 2))
#     xy = np.column_stack((np.transpose(data['x']), np.transpose(data['y'])))
#     for idx, l in enumerate(labels):
#         centers[idx] = np.average(xy[cluster.labels_ == l], axis=0)
#     return centers


def xy_from_N(clusters: list[_KMeans], positions: list[np.ndarray]) -> np.ndarray:
    """Calculate subclusters xy mean positions."""
    N_FLUO = 4
    centers = np.ndarray((len(clusters), N_FLUO, 2))
    for idx, (c, p) in enumerate(zip(clusters, positions)):
        labels = set(c.labels_) - {-1}
        centers[idx] = [np.average(p[c.labels_ == l], axis=0) for l in labels]
    return centers


def calibrate_origami(data):
    # encontrar muestras colocalizadas (con un radio apto angulos)
    origamis, all_positions = cluster_xy_positions(data, dist_threshold=30, px_size=133)
    n_clus = N_clusters(origamis, data)
    # Verificar calidad de clusters y filtrar
    ...
    cluster_labels = set(cluster.labels_) - {-1}
    cluster_filters = [(cluster.labels_ == l) for l in cluster_labels]
    xy = np.column_stack((np.transpose(data['x']), np.transpose(data['y'])))
    clustered_xy = [np.array(xy[cf]) for cf in cluster_filters]

    # Ver el ángulo y dirección en XY con ese corrimiento sacar el ángulo.
    ...


if __name__ == '__main__':
    data = fake_origami_data()
    t = _time.time()
    cluster_threshold = 30/5  # la distancia si está 100% acostado es 30
    px_size = 133
    # Encontrar todas las localizaciones del mismo origami
    cluster, xy = cluster_xy_positions(data, cluster_threshold, px_size)
    # Ahora encontrar sublocalizaciones con el mismo N (z)
    n_clus = N_clusters(cluster, data)
    # Esto debería hacerse dentro de alguna funcion
    cluster_labels = set(cluster.labels_) - {-1}
    cluster_filters = [(cluster.labels_ == l) for l in cluster_labels]
    clustered_xy = [np.array(xy[cf]) for cf in cluster_filters]
    # for origami, points in zip(n_clus, clustered_xy):
    centers = xy_from_N(n_clus, clustered_xy)  # centers es array(n_origami, 4, 2)
    plt.figure("Falsos origamis")
    for c, clus in zip(centers, n_clus):
        c = c[np.argsort(clus.cluster_centers_, None)]
        # dist = difernecia entre centros proyextados en la linea
        # X = np.column_stack((np.ones_like(c[:, 0]), c[:, 0]))
        # _, pend = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), c[:, 1])
        # print(pend)
        plt.plot(*zip(*c))
    # print(centers)
    # print(n_clus[0].)
    # print([len(set(n_cl.labels_)) for n_cl in n_clus])
    # print([n_cl.cluster_centers_ for n_cl in n_clus])
    print(_time.time() - t)
# if False:
    start = _time.time()

    with pd.HDFStore(filename, 'r') as store:
        data = store['locs']
    yaml_file = filename.with_suffix('.yaml')
    with open(yaml_file, "r") as info_file:
        info = list(yaml.load_all(info_file, Loader=yaml.FullLoader))
    px_size = info[1]['Pixelsize']
    radius_threshold = 75  # nm
    idx_to_discard = filter_data(data, radius_threshold, px_size)
    data_filtered = data.drop(labels=idx_to_discard, axis=0)
    data_filtered = data_filtered.reset_index(level=None, drop=True, inplace=False,
                                              col_level=0)
    _lgr.info("minimum alpha is: %s", 1 - (np.min(data['photons']) / np.max(data['photons'])))
    z = calculate_z(data_filtered, 0.95, 100, np.max(data['photons']))
    # data_filtered['z'] = z
    cluster_threshold = 30/5  # la distancia si está 100% acostado es 30
    cluster = cluster_xy_positions(data_filtered, cluster_threshold, px_size)
    out_file = filename.with_stem(filename.stem + '_frames_filtered')
    sa, saType = df_to_sarray(data_filtered)
    try:
        with h5py.File(out_file, 'a') as f:
            f.create_dataset('locs', data=sa, dtype=saType)
        with open(filename.with_suffix('.yaml').with_stem(filename.stem + '_frames_filtered'), "w") as file:
            yaml.dump_all(info, file, default_flow_style=False)
    except ValueError:
        _lgr.error("No puedo grabar, el archivo ya existe o algo así")
        # raise
    end = _time.time()
    _lgr.info('Script total time: %s s', end - start)
    x = data_filtered['x'][cluster[0].core_sample_indices_]
    y = data_filtered['y'][cluster[0].core_sample_indices_]
    plt.figure("centros")
    plt.scatter(x, y, s=1)
    # centros = clusters_centers(cluster, data_filtered)
    # plt.figure("dos")
    # plt.scatter(centros[:, 0], centros[:, 1], s=1)
