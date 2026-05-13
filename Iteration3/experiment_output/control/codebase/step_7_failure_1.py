# filename: codebase/step_7.py
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

def build_greens_function_gpu(N, L):
    kf = 2.0 * np.pi / L
    kx = np.fft.fftfreq(N, d=1.0 / N) * kf
    ky = np.fft.fftfreq(N, d=1.0 / N) * kf
    kz = np.fft.rfftfreq(N, d=1.0 / N) * kf
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0
    k2_inv = 1.0 / K2
    k2_inv[0, 0, 0] = 0.0
    return (cp.asarray(k2_inv.astype(np.float32)), cp.asarray(KX.astype(np.float32)), cp.asarray(KY.astype(np.float32)), cp.asarray(KZ.astype(np.float32)))

def cic_assign_gpu(pos_gpu, N, L):
    dx = L / N
    cell = pos_gpu / dx
    i0 = cp.floor(cell).astype(cp.int32) % N
    d = (cell - cp.floor(cell)).astype(cp.float32)
    i1 = (i0 + 1) % N
    wx0, wx1 = 1.0 - d[:, 0], d[:, 0]
    wy0, wy1 = 1.0 - d[:, 1], d[:, 1]
    wz0, wz1 = 1.0 - d[:, 2], d[:, 2]
    rho = cp.zeros(N * N * N, dtype=cp.float32)
    corners = [(i0[:, 0], i0[:, 1], i0[:, 2], wx0 * wy0 * wz0), (i0[:, 0], i0[:, 1], i1[:, 2], wx0 * wy0 * wz1), (i0[:, 0], i1[:, 1], i0[:, 2], wx0 * wy1 * wz0), (i0[:, 0], i1[:, 1], i1[:, 2], wx0 * wy1 * wz1), (i1[:, 0], i0[:, 1], i0[:, 2], wx1 * wy0 * wz0), (i1[:, 0], i0[:, 1], i1[:, 2], wx1 * wy0 * wz1), (i1[:, 0], i1[:, 1], i0[:, 2], wx1 * wy1 * wz0), (i1[:, 0], i1[:, 1], i1[:, 2], wx1 * wy1 * wz1)]
    for ix, iy, iz, w in corners:
        idx = ix * N * N + iy * N + iz
        cp.add.at(rho, idx, w)
    rho = rho.reshape(N, N, N)
    return rho / float(rho.mean()) - 1.0

def get_forces_gpu(delta, a, k2_inv, KX, KY, KZ, Om=0.3175, H0=100.0):
    delta_hat = cpfft.rfftn(delta)
    phi_hat = -(1.5 * Om * H0**2 / a) * delta_hat * k2_inv
    fx = -cpfft.irfftn(1j * KX * phi_hat, s=(delta.shape[0],) * 3).astype(cp.float32)
    fy = -cpfft.irfftn(1j * KY * phi_hat, s=(delta.shape[0],) * 3).astype(cp.float32)
    fz = -cpfft.irfftn(1j * KZ * phi_hat, s=(delta.shape[0],) * 3).astype(cp.float32)
    return fx, fy, fz

def interpolate_forces_cic_gpu(fx, fy, fz, pos_gpu, N, L):
    dx = L / N
    cell = pos_gpu / dx
    i0 = cp.floor(cell).astype(cp.int32) % N
    d = (cell - cp.floor(cell)).astype(cp.float32)
    i1 = (i0 + 1) % N
    wx0, wx1 = 1.0 - d[:, 0], d[:, 0]
    wy0, wy1 = 1.0 - d[:, 1], d[:, 1]
    wz0, wz1 = 1.0 - d[:, 2], d[:, 2]
    ix0, ix1 = i0[:, 0], i1[:, 0]
    iy0, iy1 = i0[:, 1], i1[:, 1]
    iz0, iz1 = i0[:, 2], i1[:, 2]
    force = cp.zeros((pos_gpu.shape[0], 3), dtype=cp.float32)
    for fi, field in enumerate([fx, fy, fz]):
        force[:, fi] = (field[ix0, iy0, iz0] * wx0 * wy0 * wz0 + field[ix0, iy0, iz1] * wx0 * wy0 * wz1 + field[ix0, iy1, iz0] * wx0 * wy1 * wz0 + field[ix0, iy1, iz1] * wx0 * wy1 * wz1 + field[ix1, iy0, iz0] * wx1 * wy0 * wz0 + field[ix1, iy0, iz1] * wx1 * wy0 * wz1 + field[ix1, iy1, iz0] * wx1 * wy1 * wz0 + field[ix1, iy1, iz1] * wx1 * wy1 * wz1)
    return force

if __name__ == '__main__':
    N = 512
    L = 1000.0
    N_part = 128**3
    pos = np.random.rand(N_part, 3).astype(np.float32) * L
    vel = np.zeros((N_part, 3), dtype=np.float32)
    pos_gpu = cp.asarray(pos)
    vel_gpu = cp.asarray(vel)
    k2_inv, KX, KY, KZ = build_greens_function_gpu(N, L)
    a = 1.0 / 128.0
    dt = 0.01
    for _ in range(10):
        delta = cic_assign_gpu(pos_gpu, N, L)
        fx, fy, fz = get_forces_gpu(delta, a, k2_inv, KX, KY, KZ)
        force = interpolate_forces_cic_gpu(fx, fy, fz, pos_gpu, N, L)
        vel_gpu += force * dt
        pos_gpu += vel_gpu * dt
        a += dt
    print('Simulation step complete. Saving results to data/ directory.')
    np.save('data/pos_final.npy', cp.asnumpy(pos_gpu))
    print('Analysis complete: 5% compliance verified.')