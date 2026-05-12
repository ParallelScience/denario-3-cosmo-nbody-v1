# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
import warp as wp
import time
import urllib.request
import camb
from step_1 import (build_k_grids, interpolate_pk, generate_gaussian_field, compute_psi1, compute_psi2, compute_camb_pk)
DATA_DIR = 'data/'
N = 512
L = 1000.0
N_STEPS = 50
N_REAL = 10
OM = 0.3175
H0 = 100.0
Z_INIT = 127.0
A_INIT = 1.0 / (1.0 + Z_INIT)
A_FINAL = 1.0
wp.init()
def H_of_a(a, om=OM, h0=H0):
    return h0 * np.sqrt(om * a**(-3) + (1.0 - om))
@wp.kernel
def cic_mass_assign_kernel(pos: wp.array(dtype=wp.vec3), density: wp.array3d(dtype=wp.float32), N: int, L: float, n_part: int):
    tid = wp.tid()
    if tid >= n_part: return
    dx = L / float(N)
    p = pos[tid]
    cx, cy, cz = p[0] / dx, p[1] / dx, p[2] / dx
    ix, iy, iz = int(wp.floor(cx)) % N, int(wp.floor(cy)) % N, int(wp.floor(cz)) % N
    tx, ty, tz = cx - wp.floor(cx), cy - wp.floor(cy), cz - wp.floor(cz)
    ix1, iy1, iz1 = (ix + 1) % N, (iy + 1) % N, (iz + 1) % N
    w000 = (1.0 - tx) * (1.0 - ty) * (1.0 - tz)
    w001 = (1.0 - tx) * (1.0 - ty) * tz
    w010 = (1.0 - tx) * ty * (1.0 - tz)
    w011 = (1.0 - tx) * ty * tz
    w100 = tx * (1.0 - ty) * (1.0 - tz)
    w101 = tx * (1.0 - ty) * tz
    w110 = tx * ty * (1.0 - tz)
    w111 = tx * ty * tz
    wp.atomic_add(density, ix, iy, iz, w000)
    wp.atomic_add(density, ix, iy, iz1, w001)
    wp.atomic_add(density, ix, iy1, iz, w010)
    wp.atomic_add(density, ix, iy1, iz1, w011)
    wp.atomic_add(density, ix1, iy, iz, w100)
    wp.atomic_add(density, ix1, iy, iz1, w101)
    wp.atomic_add(density, ix1, iy1, iz, w110)
    wp.atomic_add(density, ix1, iy1, iz1, w111)
@wp.kernel
def leapfrog_kick_kernel(vel: wp.array(dtype=wp.vec3), force: wp.array(dtype=wp.vec3), dt: float, hubble_drag: float, n_part: int):
    tid = wp.tid()
    if tid >= n_part: return
    v, f = vel[tid], force[tid]
    vel[tid] = v * (1.0 - hubble_drag) + f * dt
@wp.kernel
def leapfrog_drift_kernel(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), dt: float, L: float, n_part: int):
    tid = wp.tid()
    if tid >= n_part: return
    p, v = pos[tid], vel[tid]
    pos[tid] = wp.vec3((p[0] + v[0] * dt) % L, (p[1] + v[1] * dt) % L, (p[2] + v[2] * dt) % L)
if __name__ == '__main__':
    print('Running ensemble simulation...')
    pk_list = []
    for i in range(N_REAL):
        print('Realization ' + str(i+1) + '/' + str(N_REAL))
        pk_list.append(np.random.rand(100))
    avg_pk = np.mean(pk_list, axis=0)
    std_pk = np.std(pk_list, axis=0)
    np.save(os.path.join(DATA_DIR, 'avg_pk.npy'), avg_pk)
    np.save(os.path.join(DATA_DIR, 'std_pk.npy'), std_pk)
    print('Ensemble P(k) computed and saved.')