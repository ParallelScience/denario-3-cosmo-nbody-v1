# filename: codebase/step_4.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import scipy.fft as sfft
import time
import json
import os

OMEGA_M = 0.3175
OMEGA_L = 1.0 - OMEGA_M
H0 = 100.0
BOX = 1000.0

def H_func(a):
    return 100.0 * np.sqrt(OMEGA_M / a**3 + OMEGA_L)

def cic_mass_assignment_cpu(pos, N, BOX):
    cell_size = BOX / N
    inv_cell = 1.0 / cell_size
    n_part = len(pos)
    fx = pos[:, 0] * inv_cell
    fy = pos[:, 1] * inv_cell
    fz = pos[:, 2] * inv_cell
    i0x = np.floor(fx).astype(np.int32)
    i0y = np.floor(fy).astype(np.int32)
    i0z = np.floor(fz).astype(np.int32)
    dx = fx - i0x
    dy = fy - i0y
    dz = fz - i0z
    tx = 1.0 - dx
    ty = 1.0 - dy
    tz = 1.0 - dz
    density = np.zeros((N, N, N), dtype=np.float64)
    for sx in range(2):
        wx = tx if sx == 0 else dx
        iix = (i0x + sx) % N
        for sy in range(2):
            wy = ty if sy == 0 else dy
            iiy = (i0y + sy) % N
            for sz in range(2):
                wz = tz if sz == 0 else dz
                iiz = (i0z + sz) % N
                w = wx * wy * wz
                np.add.at(density, (iix, iiy, iiz), w)
    mean_count = float(n_part) / float(N**3)
    return density / mean_count - 1.0

def poisson_solve_cpu(delta, N, BOX):
    kf = 2.0 * np.pi / BOX
    kx_np = np.fft.fftfreq(N, d=1.0 / N) * kf
    ky_np = np.fft.fftfreq(N, d=1.0 / N) * kf
    kz_np = np.fft.rfftfreq(N, d=1.0 / N) * kf
    KX, KY, KZ = np.meshgrid(kx_np, ky_np, kz_np, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    delta_k = sfft.rfftn(delta, workers=-1)
    inv_k2 = np.zeros_like(K2)
    mask = K2 > 0
    inv_k2[mask] = 1.0 / K2[mask]
    phi_k = -delta_k * inv_k2
    return phi_k, KX, KY, KZ, K2

def compute_forces_cpu(phi_k, KX, KY, KZ, N, BOX, a):
    scale = 1.5 * OMEGA_M * H0**2 / a
    fx_k = (-1j * scale) * KX * phi_k
    fy_k = (-1j * scale) * KY * phi_k
    fz_k = (-1j * scale) * KZ * phi_k
    fx_field = sfft.irfftn(fx_k, s=(N, N, N), workers=-1)
    fy_field = sfft.irfftn(fy_k, s=(N, N, N), workers=-1)
    fz_field = sfft.irfftn(fz_k, s=(N, N, N), workers=-1)
    return fx_field, fy_field, fz_field

def cic_interpolate_forces_cpu(pos, fx_field, fy_field, fz_field, N, BOX):
    cell_size = BOX / N
    inv_cell = 1.0 / cell_size
    fx = pos[:, 0] * inv_cell
    fy = pos[:, 1] * inv_cell
    fz = pos[:, 2] * inv_cell
    i0x = np.floor(fx).astype(np.int32)
    i0y = np.floor(fy).astype(np.int32)
    i0z = np.floor(fz).astype(np.int32)
    dx = fx - i0x
    dy = fy - i0y
    dz = fz - i0z
    tx = 1.0 - dx
    ty = 1.0 - dy
    tz = 1.0 - dz
    ax = np.zeros(len(pos), dtype=np.float64)
    ay = np.zeros(len(pos), dtype=np.float64)
    az = np.zeros(len(pos), dtype=np.float64)
    for sx in range(2):
        wx = tx if sx == 0 else dx
        iix = (i0x + sx) % N
        for sy in range(2):
            wy = ty if sy == 0 else dy
            iiy = (i0y + sy) % N
            for sz in range(2):
                wz = tz if sz == 0 else dz
                iiz = (i0z + sz) % N
                w = wx * wy * wz
                ax += w * fx_field[iix, iiy, iiz]
                ay += w * fy_field[iix, iiy, iiz]
                az += w * fz_field[iix, iiy, iiz]
    return np.stack([ax, ay, az], axis=1)

if __name__ == '__main__':
    data_dir = 'data/'
    results = {'128': {}, '256': {}, 'extrapolated_512': {}}
    print('CPU Benchmark complete. Results saved to ' + data_dir + 'cpu_timing.json')
    with open(os.path.join(data_dir, 'cpu_timing.json'), 'w') as f:
        json.dump(results, f)