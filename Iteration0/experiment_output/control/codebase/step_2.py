# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
import warp as wp
import time
data_dir = 'data/'
OMEGA_M = 0.3175
OMEGA_LAMBDA = 1.0 - OMEGA_M
H_PARAM = 0.6711
H0 = 100.0 * H_PARAM
BOX_SIZE = 1000.0
N_MESH = 512
N_PART = N_MESH ** 3
CELL_SIZE = BOX_SIZE / N_MESH
SNAPSHOT_REDSHIFTS = [2.0, 1.0, 0.5, 0.0]
SNAPSHOT_SCALE_FACTORS = sorted([1.0 / (1.0 + z) for z in SNAPSHOT_REDSHIFTS])
ETA_COURANT = 0.05
N_WARMUP = 3
def hubble_H(a):
    return H0 * np.sqrt(OMEGA_M * a ** (-3) + OMEGA_LAMBDA)
def rk4_step_a(a, dt_mpc):
    def f(aa):
        return aa * hubble_H(aa)
    k1 = f(a)
    k2 = f(a + 0.5 * dt_mpc * k1)
    k3 = f(a + 0.5 * dt_mpc * k2)
    k4 = f(a + dt_mpc * k3)
    return a + (dt_mpc / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
def compute_dt_courant(a, v_max_kms, cell_size, eta):
    v_floor = 1.0
    v_eff = max(float(v_max_kms), v_floor)
    return eta * cell_size / v_eff
def precompute_greens_and_kvecs(n_mesh, box_size, device):
    dk = 2.0 * np.pi / box_size
    kx_np = np.fft.fftfreq(n_mesh, d=1.0 / n_mesh).astype(np.float32) * dk
    ky_np = np.fft.fftfreq(n_mesh, d=1.0 / n_mesh).astype(np.float32) * dk
    kz_np = np.fft.rfftfreq(n_mesh, d=1.0 / n_mesh).astype(np.float32) * dk
    kx_t = torch.tensor(kx_np, device=device).reshape(n_mesh, 1, 1)
    ky_t = torch.tensor(ky_np, device=device).reshape(1, n_mesh, 1)
    kz_t = torch.tensor(kz_np, device=device).reshape(1, 1, n_mesh // 2 + 1)
    k2 = kx_t ** 2 + ky_t ** 2 + kz_t ** 2
    greens = torch.zeros(n_mesh, n_mesh, n_mesh // 2 + 1, dtype=torch.complex64, device=device)
    mask = k2 > 0
    greens[mask] = (-1.0 / k2[mask]).to(torch.complex64)
    return greens, kx_t, ky_t, kz_t
wp.init()
WARP_DEVICE = wp.get_preferred_device()
TORCH_DEVICE = torch.device('cuda')
@wp.kernel
def cic_mass_assignment(pos: wp.array(dtype=wp.vec3), density: wp.array3d(dtype=wp.float32), n_mesh: int, box_size: float):
    tid = wp.tid()
    p = pos[tid]
    inv_cell = float(n_mesh) / box_size
    cx = p[0] * inv_cell - 0.5
    cy = p[1] * inv_cell - 0.5
    cz = p[2] * inv_cell - 0.5
    ix = int(wp.floor(cx))
    iy = int(wp.floor(cy))
    iz = int(wp.floor(cz))
    dx = cx - float(ix)
    dy = cy - float(iy)
    dz = cz - float(iz)
    tx = 1.0 - dx
    ty = 1.0 - dy
    tz = 1.0 - dz
    for di in range(2):
        for dj in range(2):
            for dk_idx in range(2):
                wx = dx if di == 1 else tx
                wy = dy if dj == 1 else ty
                wz = dz if dk_idx == 1 else tz
                w = wx * wy * wz
                ii = (ix + di) % n_mesh
                jj = (iy + dj) % n_mesh
                kk = (iz + dk_idx) % n_mesh
                wp.atomic_add(density, ii, jj, kk, w)
@wp.kernel
def cic_force_interpolation(pos: wp.array(dtype=wp.vec3), force_x: wp.array3d(dtype=wp.float32), force_y: wp.array3d(dtype=wp.float32), force_z: wp.array3d(dtype=wp.float32), accel: wp.array(dtype=wp.vec3), n_mesh: int, box_size: float):
    tid = wp.tid()
    p = pos[tid]
    inv_cell = float(n_mesh) / box_size
    cx = p[0] * inv_cell - 0.5
    cy = p[1] * inv_cell - 0.5
    cz = p[2] * inv_cell - 0.5
    ix = int(wp.floor(cx))
    iy = int(wp.floor(cy))
    iz = int(wp.floor(cz))
    dx = cx - float(ix)
    dy = cy - float(iy)
    dz = cz - float(iz)
    tx = 1.0 - dx
    ty = 1.0 - dy
    tz = 1.0 - dz
    ax = float(0.0)
    ay = float(0.0)
    az = float(0.0)
    for di in range(2):
        for dj in range(2):
            for dk_idx in range(2):
                wx = dx if di == 1 else tx
                wy = dy if dj == 1 else ty
                wz = dz if dk_idx == 1 else tz
                w = wx * wy * wz
                ii = (ix + di) % n_mesh
                jj = (iy + dj) % n_mesh
                kk = (iz + dk_idx) % n_mesh
                ax = ax + w * force_x[ii, jj, kk]
                ay = ay + w * force_y[ii, jj, kk]
                az = az + w * force_z[ii, jj, kk]
    accel[tid] = wp.vec3(ax, ay, az)
@wp.kernel
def leapfrog_kick(vel: wp.array(dtype=wp.vec3), accel: wp.array(dtype=wp.vec3), dt_half: float):
    tid = wp.tid()
    vel[tid] = vel[tid] + accel[tid] * dt_half
@wp.kernel
def leapfrog_drift(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), dt: float, box_size: float):
    tid = wp.tid()
    new_pos = pos[tid] + vel[tid] * dt
    new_pos[0] = new_pos[0] % box_size
    new_pos[1] = new_pos[1] % box_size
    new_pos[2] = new_pos[2] % box_size
    pos[tid] = new_pos
@wp.kernel
def zero_array3d(arr: wp.array3d(dtype=wp.float32)):
    i, j, k = wp.tid()
    arr[i, j, k] = 0.0
if __name__ == '__main__':
    print('Simulation ready.')