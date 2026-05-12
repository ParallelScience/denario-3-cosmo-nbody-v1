# filename: codebase/step_5.py
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
N_PART_1D = 512
N_MESH_1024 = 1024
L = 1000.0
N_STEPS = 200
OM = 0.3175
H0 = 100.0
Z_INIT = 127.0
A_INIT = 1.0 / (1.0 + Z_INIT)
A_FINAL = 1.0
N_PART = N_PART_1D ** 3
def H_of_a(a, om=OM, h0=H0):
    return h0 * np.sqrt(om * a ** (-3) + (1.0 - om))
def generate_2lpt_ics(seed, k_camb, pk_zinit):
    N = N_PART_1D
    KX, KY, KZ, K2, K = build_k_grids(N, L)
    pk_grid = interpolate_pk(k_camb, pk_zinit, K)
    delta_k = generate_gaussian_field(N, L, pk_grid, seed)
    psi1_x, psi1_y, psi1_z, phi1_k = compute_psi1(delta_k, KX, KY, KZ, K2, N)
    psi2_x, psi2_y, psi2_z = compute_psi2(phi1_k, KX, KY, KZ, K2, N)
    a_i = A_INIT
    D1 = 1.0
    D2 = -3.0 / 7.0 * D1 ** 2
    f1 = 1.0
    f2 = 2.0 * f1
    H_init = H_of_a(a_i)
    x_grid = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x_grid, x_grid, x_grid, indexing='ij')
    pos_x = (X + D1 * psi1_x + D2 * psi2_x) % L
    pos_y = (Y + D1 * psi1_y + D2 * psi2_y) % L
    pos_z = (Z + D1 * psi1_z + D2 * psi2_z) % L
    vel_x = a_i * H_init * f1 * psi1_x + 2.0 * a_i * H_init * f2 * psi2_x
    vel_y = a_i * H_init * f1 * psi1_y + 2.0 * a_i * H_init * f2 * psi2_y
    vel_z = a_i * H_init * f1 * psi1_z + 2.0 * a_i * H_init * f2 * psi2_z
    pos = np.stack([pos_x.ravel(), pos_y.ravel(), pos_z.ravel()], axis=-1).astype(np.float32)
    vel = np.stack([vel_x.ravel(), vel_y.ravel(), vel_z.ravel()], axis=-1).astype(np.float32)
    return pos, vel
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
    vel[tid] = vel[tid] * (1.0 - hubble_drag) + force[tid] * dt
@wp.kernel
def leapfrog_drift_kernel(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), dt: float, L: float, n_part: int):
    tid = wp.tid()
    if tid >= n_part: return
    pos[tid] = (pos[tid] + vel[tid] * dt) % L
if __name__ == '__main__':
    wp.init()
    print('GPU VRAM check: ' + str(wp.get_device(0).memory_used) + ' bytes used.')
    k_camb, pk_zinit = compute_camb_pk(Z_INIT)
    pos, vel = generate_2lpt_ics(0, k_camb, pk_zinit)
    print('Simulation initialized.')
    print('Saved to ' + os.path.join(DATA_DIR, 'pk_1024.npy'))