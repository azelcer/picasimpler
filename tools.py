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
import logging as _lgn
# The following imports are used only for development.


_lgn.basicConfig()
_lgr = _lgn.getLogger(__name__)
_lgr.setLevel(_lgn.INFO)


def fake_origami_data():
    """Crea info falsa con origamis en un cuadrado (para probar)"""
    # Estos son parametros que podrian ser variables
    # Caracteristicas de los origamis
    D = 100  # distancia entre origamis
    D_F = 30  # distancia entre fluoroforos
    L = 5  # número de origamis por lado
    N = L*L  # number of origamis
    N_FLUO = 4  # fluoroforos por origami
    ANG_MAX = np.radians(25)  # Angulo maximo entre origami y sustrato
    # Parametros para simpler
    d = 100  # nm
    alpha = 0.95
    N0 = 15000
    # otros
    px_size = 133.  # en nm por pixel
    z_noise = 5.  # en nm
    # Angle of each origami to the normal
    angles = np.random.random(N) * ANG_MAX
    # Direction of tilt with X axe
    rotations = np.random.random(N) * np.pi * 2
    # print(rotations)
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

    # Todo: meter probabilidad acá
    rng = np.random.default_rng()
    # Acá hay una sola localización pero con ruido: hacer unas 50 con ruido en x y en y
    I = N0 * (np.exp(-(z.ravel() + rng.normal(0, 5, z.size))/d) + (1-alpha))
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
    n_registers = total_fluorophores * loc_per_fluo
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
        xdf[slc] = (x.ravel() + rng.normal(0, sx, total_fluorophores)) / px_size
        ydf[slc] = (y.ravel() + rng.normal(0, sy, total_fluorophores)) / px_size
        zdf[slc] = (z.ravel() + rng.normal(0, 5, total_fluorophores)) / px_size
        photons[slc] = N0 * (np.exp(-(z.ravel() + rng.normal(0, z_noise, z.size))/d)
                             + (1-alpha))
        frames[slc] = i
    sx = np.full((n_registers,), sx, dtype=np.float32)
    sy = np.full((n_registers,), sy, dtype=np.float32)
    lp_ = np.full((n_registers,), 0.01, dtype=np.float32)
    # Picasso file fields
    # 'frame', 'x', 'y', 'photons', 'sx', 'sy', 'bg', 'lpx', 'lpy',
    # 'ellipticity', 'net_gradient', 'x_pick_rot', 'y_pick_rot', 'group'
    rv = pd.DataFrame({'frame': frames, 'x': xdf, 'y': ydf, 'photons': photons,
                       'sx': sx, 'sy': sy, 'bg': bg, 'lpx': lp_, 'lpy': lp_,
                       'ellipticity': lp_, 'net_gradient': lp_,
                       'x_pick_rot': lp_, 'y_pick_rot': lp_,
                       'group': np.zeros(n_registers, dtype=np.int32),
                       'z': zdf})
    return rv


if __name__ == '__main__':
    data = fake_origami_data()
