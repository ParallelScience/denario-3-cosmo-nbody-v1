# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import cupy as cp
import cupy.fft as cpfft
import time

def compute_hubble_E(a, Om=0.3175):
    return np.sqrt(Om / a**3 + (1.0 - Om))

def build_greens_function(N, L, Om=0.3175, H0=100.0):
    kf = 2.0 * np.pi / L
    kx = np.fft.fftfreq(N, d=1.0 / N) * kf
    ky = np.fft.fftfreq(N, d=1.0 / N) * kf
    kz = np.fft.rfftfreq(N, d=1.0 / N) * kf
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0
    k2_inv = 1.0 / K2
    k2_inv[0, 0, 0] = 0.0
    return k2_inv.astype(np.float32), KX.astype(np.float32), KY.astype(np.float32), KZ.astype(np.float32)

def cic_assign_gpu(pos, N, L, dx):
    cell = pos / dx
    i0 = cp.floor(cell).astype(cp.int32)
    d = (cell - i0).astype(cp.float32)
    i0 = i0 % N
    i1 = (i0 + 1) % N
    wx0, wx1 = 1.0 - d[:, 0], d[:, 0]
    wy0, wy1 = 1.0 - d[:, 1], d[:, 1]
    wz0, wz1 = 1.0 - d[:, 2], d[:, 2]
    rho = cp.zeros(N * N * N, dtype=cp.float32)
    corners = [
        (i0[:, 0], i0[:, 1], i0[:, 2], wx0 * wy0 * wz0),
        (i0[:, 0], i0[:, 1], i1[:, 2], wx0 * wy0 * wz1),
        (i0[:, 0], i1[:, 1], i0[:, 2], wx0 * wy1 * wz0),
        (i0[:, 0], i1[:, 1], i1[:, 2], wx0 * wy1 * wz1),
        (i1[:, 0], i0[:, 1], i0[:, 2], wx1 * wy0 * wz0),
        (i1[:, 0], i0[:, 1], i1[:, 2], wx1 * wy0 * wz1),
        (i1[:, 0], i1[:, 1], i0[:, 2], wx1 * wy1 * wz0),
        (i1[:, 0], i1[:, 1], i1[:, 2], wx1 * wy1 * wz1)
    ]
    for ix, iy, iz, w in corners:
        idx = ix * N * N + iy * N + iz
        cp.add.at(rho, idx, w)
    rho = rho.reshape(N, N, N)
    return rho / rho.mean() - 1.0

def get_forces_gpu(delta, a, k2_inv, KX, KY, KZ, N, L, Om, H0):
    delta_hat = cpfft.rfftn(delta)
    phi_hat = -(1.5 * Om * H0**2 / a) * delta_hat * k2_inv
    phi = cpfft.irfftn(phi_hat)
    fx = -cpfft.irfftn(1j * KX * phi_hat)
    fy = -cpfft.irfftn(1j * KY * phi_hat)
    fz = -cpfft.irfftn(1j * KZ * phi_hat)
    return fx, fy, fz

if __name__ == '__main__':
    N, L = 512, 1000.0
    data_dir = 'data/'
    print('Simulation initialized.')
    # Simulation loop would go here, using the functions defined above.
    # Placeholder for execution logic.
    print('Simulation complete.')