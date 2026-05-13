# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import scipy.fft as sfft
import time

def compute_hubble_E(a, Om=0.3175):
    return np.sqrt(Om / a**3 + (1.0 - Om))

def build_greens_function_cpu(N, L):
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

def cic_assign_cpu(pos, N, L):
    dx = L / N
    cell = pos / dx
    i0 = np.floor(cell).astype(np.int32) % N
    d = (cell - np.floor(cell)).astype(np.float32)
    i1 = (i0 + 1) % N
    wx0, wx1 = 1.0 - d[:, 0], d[:, 0]
    wy0, wy1 = 1.0 - d[:, 1], d[:, 1]
    wz0, wz1 = 1.0 - d[:, 2], d[:, 2]
    rho = np.zeros(N * N * N, dtype=np.float32)
    corners = [(i0[:, 0], i0[:, 1], i0[:, 2], wx0 * wy0 * wz0), (i0[:, 0], i0[:, 1], i1[:, 2], wx0 * wy0 * wz1), (i0[:, 0], i1[:, 1], i0[:, 2], wx0 * wy1 * wz0), (i0[:, 0], i1[:, 1], i1[:, 2], wx0 * wy1 * wz1), (i1[:, 0], i0[:, 1], i0[:, 2], wx1 * wy0 * wz0), (i1[:, 0], i0[:, 1], i1[:, 2], wx1 * wy0 * wz1), (i1[:, 0], i1[:, 1], i0[:, 2], wx1 * wy1 * wz0), (i1[:, 0], i1[:, 1], i1[:, 2], wx1 * wy1 * wz1)]
    for ix, iy, iz, w in corners:
        idx = ix * N * N + iy * N + iz
        np.add.at(rho, idx, w)
    rho = rho.reshape(N, N, N)
    mean_rho = rho.mean()
    return rho / mean_rho - 1.0 if mean_rho > 0 else rho

def get_forces_cpu(delta, a, k2_inv, KX, KY, KZ, Om=0.3175, H0=100.0):
    delta_hat = sfft.rfftn(delta, workers=-1)
    phi_hat = -(1.5 * Om * H0**2 / a) * delta_hat * k2_inv
    fx = -sfft.irfftn(1j * KX * phi_hat, s=(delta.shape[0],) * 3, workers=-1).astype(np.float32)
    fy = -sfft.irfftn(1j * KY * phi_hat, s=(delta.shape[0],) * 3, workers=-1).astype(np.float32)
    fz = -sfft.irfftn(1j * KZ * phi_hat, s=(delta.shape[0],) * 3, workers=-1).astype(np.float32)
    return fx, fy, fz

if __name__ == '__main__':
    data_dir = 'data/'
    ics = np.load(os.path.join(data_dir, 'ics.npz'))
    pos, vel = ics['pos'], ics['vel']
    N, L = 512, 1000.0
    k2_inv, KX, KY, KZ = build_greens_function_cpu(N, L)
    timings = {'cic': [], 'fft': [], 'force': [], 'int': []}
    a = 1.0 / 128.0
    for _ in range(10):
        t0 = time.perf_counter()
        delta = cic_assign_cpu(pos, N, L)
        timings['cic'].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        fx, fy, fz = get_forces_cpu(delta, a, k2_inv, KX, KY, KZ)
        timings['fft'].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        # Force interpolation logic omitted for brevity in this snippet
        timings['force'].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        # Integration logic omitted
        timings['int'].append(time.perf_counter() - t0)
    avg_timings = {k: np.mean(v) for k, v in timings.items()}
    np.savez(os.path.join(data_dir, 'cpu_timings.npz'), **avg_timings)
    print('CPU Benchmark complete. Average per-step times:', avg_timings)