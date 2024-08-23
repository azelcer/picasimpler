# -*- coding: utf-8 -*-
"""

@author: aszalai, azelcer
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from scipy.spatial import distance
import h5py
import yaml
import time as _time
from tqdm import tqdm
import logging as _lgn
import pathlib as _pathlib
import warnings as _warnings
from sklearn.cluster import DBSCAN as _DBSCAN
import matplotlib.pyplot as plt


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
                         px_size: float) -> np.ndarray:
    """Clusters locations for origami claibration."""
    start = _time.time()
    eps = (dist_threshold/px_size)**2
    xy = np.column_stack((np.transpose(data['x']), np.transpose(data['y'])))
    rv = _DBSCAN(eps=eps, min_samples=4, metric='sqeuclidean', n_jobs=-1).fit(xy)
    _lgr.info('clusters found: %s', len(set(rv.labels_) - {-1}))
    _lgr.info('locations assigned: %s out of %s', len(rv.core_sample_indices_), len(xy))
    _lgr.info('Time of clustering: %s s', _time.time()-start)
    return rv


def clusters_centers(cluster: _DBSCAN, data):
    """Calculate clusters means."""
    labels = set(cluster.labels_) - {-1}
    centers = np.ndarray((len(labels), 2))
    xy = np.column_stack((np.transpose(data['x']), np.transpose(data['y'])))
    for idx, l in enumerate(labels):
        centers[idx] = np.average(xy[cluster.labels_ == l], axis=0)
    return centers


def fake_origami_data():
    """Distribuye los origamis en un cuadrado (para probar)"""
    D = 100  # distancia entre origamis
    D_F = 30  # distancia entre fluoroforos
    L = 10  # número de origamis por lado
    N = L*L  # number of origamis
    N_FLUO = 4
    pos_noise = 5./133
    ANG_MAX = np.radians(25)
    # Angle of each origami to the normal
    angles = np.random.random(N) * ANG_MAX
    # Direction of tilt with X axe
    rotations = np.random.random(N) * np.pi * 2
    # Posiciones
    vertices = np.linspace(D, D * L, L)
    # centros de cada origami
    x_c, y_c = np.meshgrid(vertices, vertices, indexing='ij')
    x_c = x_c.ravel()
    y_c = y_c.ravel()
    # posicion relativa de cada fluoroforo, debería ser 1-3 y 2-4
    pos_vec = np.arange(1, N_FLUO+1, dtype=np.float64) * D_F
    # rotación de cada origami
    rot_x = np.cos(rotations)[:, np.newaxis] * pos_vec * np.sin(angles)[:, np.newaxis]
    rot_y = np.sin(rotations)[:, np.newaxis] * pos_vec * np.sin(angles)[:, np.newaxis]
    z = np.cos(angles)[:, np.newaxis] * pos_vec
    x = rot_x + x_c[:, np.newaxis]
    y = rot_y + y_c[:, np.newaxis]

    d = 100  # nm
    alpha = 0.95
    N0 = 15000
    # Todo: meter probabilidad acá
    rng = np.random.default_rng()
    # Acá hay una sola localización pero con ruido: hacer unas 50 con ruido en x y en y
    I = N0 * (np.exp(-(z.ravel()+rng.normal(0, 5, z.size))/d) + (1-alpha))
    # I = N0*(alpha * rng.exponential(d/z.ravel()) + (1 - alpha))
    # Esto da N0 * z/d mustras promedio
    # n_frames = 15000
    # I = rng.exponential(d/z.ravel(), (n_frames, z.size)).sum(axis=0)#+ (1-alpha))
    # I = rng.poisson(d/z.ravel(), (n_frames, z.size)).sum(axis=0)
    # print(I.max())
    # plt.scatter(x.ravel(), y.ravel(), s=1)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(x.ravel(), y.ravel(), #z.ravel(), c =
    #            I/max(I)*255)
    # plt.figure()
    # plt.scatter(z.ravel(), I)
    loc_per_fluo = 50
    total_fluorophores = L * L * N_FLUO  # despues hacer dos pares
    n_registers = total_fluorophores* loc_per_fluo
    # Index(['frame', 'x', 'y', 'photons', 'sx', 'sy', 'bg', 'lpx', 'lpy',
           # 'ellipticity', 'net_gradient', 'x_pick_rot', 'y_pick_rot', 'group'],
    xdf = np.ndarray((n_registers,), dtype=np.float32)
    ydf = np.ndarray((n_registers,), dtype=np.float32)
    zdf = np.ndarray((n_registers,), dtype=np.float32)
    photons = np.ndarray((n_registers,), dtype=np.float32)
    bg = np.zeros(n_registers, dtype=np.float32)
    frames = np.ndarray((n_registers,), dtype=np.uint32)
    sx = 5.
    sy = 5.
    for i in range(loc_per_fluo):
        slc = slice(i * total_fluorophores, (i+1) * total_fluorophores)
        xdf[slc] = (x.ravel() + rng.normal(0, sx, total_fluorophores))/133
        ydf[slc] = (y.ravel() + rng.normal(0, sy, total_fluorophores))/133
        zdf[slc] = (z.ravel() + rng.normal(0, 5, total_fluorophores))/133
        photons[slc] = N0 * (np.exp(-(z.ravel()+rng.normal(0, 5, z.size))/d) + (1-alpha))
        frames[slc] = i
    sx = np.full((n_registers,), sx, dtype=np.float32)
    sy = np.full((n_registers,), sy, dtype=np.float32)
    lp_ = np.full((n_registers,), 0.01, dtype=np.float32)
    rv = pd.DataFrame({'frame': frames, 'x': xdf, 'y': ydf, 'photons': photons,
                       'sx': sx, 'sy':sy, 'bg': bg, 'lpx': lp_, 'lpy': lp_,
                       'ellipticity': lp_, 'net_gradient': lp_, 'x_pick_rot': lp_,
                       'y_pick_rot': lp_, 'group': np.zeros(n_registers, dtype=np.int32),
                       'z': zdf})
    sa, saType = df_to_sarray(rv)
    with h5py.File('/tmp/data.hdf5', 'a') as f:
        f.create_dataset('locs', data=sa, dtype=saType)
    print(n_registers, i)


if __name__ == '__main__':
    fake_origami_data()
if False:
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
        raise
    end = _time.time()
    _lgr.info('Script total time: %s s', end - start)
    x = data_filtered['x'][cluster.core_sample_indices_]
    y = data_filtered['y'][cluster.core_sample_indices_]
    import matplotlib.pyplot as plt
    plt.scatter(x, y, s=1)
    centros = clusters_centers(cluster, data_filtered)
    plt.figure("dos")
    plt.scatter(centros[:, 0], centros[:, 1], s=1)
    # fake_origami_data()