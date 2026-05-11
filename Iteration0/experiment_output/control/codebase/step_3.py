# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import scipy.fft as sfft
import time
import json
import os

data_dir = "data/"
OMEGA_M = 0.3175
OMEGA_LAMBDA = 1.0 - OMEGA_M
H_PARAM = 0.6711
H0 = 100.0 * H_PARAM
BOX_SIZE = 1000.0
N_MESH = 512
N_PART = N_MESH ** 3
CELL_SIZE = BOX_SIZE / N_MESH
ETA_COURANT = 0.05
N_BENCH_STEPS = 5
N_FULL_STEPS = 50

def hubble_H(a):
    return H0 * np.sqrt(OMEGA_M * a ** (-3) + OMEGA_LAMBDA)

def precompute_greens_and_kvecs(n_mesh, box_size):
    dk = 2.0 * np.pi / box_size
    kx_np = np.fft.fftfreq(n_mesh, d=1.0 / n_mesh).astype(np.float32) * dk
    ky_np = np.fft.fftfreq(n_mesh, d=1.0 / n_mesh).astype(np.float32) * dk
    kz_np = np.fft.rfftfreq(n_mesh, d=1.0 / n_mesh).astype(np.float32) * dk
    kx_grid = kx_np.reshape(n_mesh, 1, 1)
    ky_grid = ky_np.reshape(1, n_mesh, 1)
    kz_grid = kz_np.reshape(1, 1, n_mesh // 2 + 1)
    k2 = kx_grid ** 2 + ky_grid ** 2 + kz_grid ** 2
    k2_safe = np.where(k2 == 0, 1.0, k2)
    greens = (-1.0 / k2_safe).astype(np.float32)
    greens[0, 0, 0] = 0.0
    return greens, kx_grid, ky_grid, kz_grid

def cic_mass_assignment_cpu(pos, n_mesh, box_size, density_out):
    density_out[:] = 0.0
    inv_cell = n_mesh / box_size
    cx = pos[:, 0] * inv_cell - 0.5
    cy = pos[:, 1] * inv_cell - 0.5
    cz = pos[:, 2] * inv_cell - 0.5
    ix = np.floor(cx).astype(np.int32)
    iy = np.floor(cy).astype(np.int32)
    iz = np.floor(cz).astype(np.int32)
    dx = (cx - ix).astype(np.float32)
    dy = (cy - iy).astype(np.float32)
    dz = (cz - iz).astype(np.float32)
    tx = 1.0 - dx
    ty = 1.0 - dy
    tz = 1.0 - dz
    for di in range(2):
        wx = dx if di == 1 else tx
        for dj in range(2):
            wy = dy if dj == 1 else ty
            for dk_idx in range(2):
                wz = dz if dk_idx == 1 else tz
                w = (wx * wy * wz).astype(np.float32)
                ii = (ix + di) % n_mesh
                jj = (iy + dj) % n_mesh
                kk = (iz + dk_idx) % n_mesh
                flat_idx = ii * n_mesh * n_mesh + jj * n_mesh + kk
                np.add.at(density_out.ravel(), flat_idx, w)

def poisson_solve_cpu(density, greens, kx_grid, ky_grid, kz_grid, n_mesh):
    mean_density = float(N_PART) / float(n_mesh ** 3)
    delta = (density - mean_density) / mean_density
    delta_k = sfft.rfftn(delta, workers=-1)
    phi_k = delta_k * greens
    fx_k = -1j * kx_grid * phi_k
    fy_k = -1j * ky_grid * phi_k
    fz_k = -1j * kz_grid * phi_k
    fx = sfft.irfftn(fx_k, s=(n_mesh, n_mesh, n_mesh), workers=-1).astype(np.float32)
    fy = sfft.irfftn(fy_k, s=(n_mesh, n_mesh, n_mesh), workers=-1).astype(np.float32)
    fz = sfft.irfftn(fz_k, s=(n_mesh, n_mesh, n_mesh), workers=-1).astype(np.float32)
    return fx, fy, fz

def cic_force_interpolation_cpu(pos, fx_grid, fy_grid, fz_grid, n_mesh, box_size):
    inv_cell = n_mesh / box_size
    cx = pos[:, 0] * inv_cell - 0.5
    cy = pos[:, 1] * inv_cell - 0.5
    cz = pos[:, 2] * inv_cell - 0.5
    ix = np.floor(cx).astype(np.int32)
    iy = np.floor(cy).astype(np.int32)
    iz = np.floor(cz).astype(np.int32)
    dx = (cx - ix).astype(np.float32)
    dy = (cy - iy).astype(np.float32)
    dz = (cz - iz).astype(np.float32)
    tx = 1.0 - dx
    ty = 1.0 - dy
    tz = 1.0 - dz
    ax = np.zeros(len(pos), dtype=np.float32)
    ay = np.zeros(len(pos), dtype=np.float32)
    az = np.zeros(len(pos), dtype=np.float32)
    for di in range(2):
        wx = dx if di == 1 else tx
        for dj in range(2):
            wy = dy if dj == 1 else ty
            for dk_idx in range(2):
                wz = dz if dk_idx == 1 else tz
                w = (wx * wy * wz).astype(np.float32)
                ii = (ix + di) % n_mesh
                jj = (iy + dj) % n_mesh
                kk = (iz + dk_idx) % n_mesh
                ax += w * fx_grid[ii, jj, kk]
                ay += w * fy_grid[ii, jj, kk]
                az += w * fz_grid[ii, jj, kk]
    return np.stack([ax, ay, az], axis=1)

if __name__ == '__main__':
    pos = np.random.rand(N_PART, 3).astype(np.float32) * BOX_SIZE
    vel = np.zeros((N_PART, 3), dtype=np.float32)
    greens, kx, ky, kz = precompute_greens_and_kvecs(N_MESH, BOX_SIZE)
    density = np.zeros((N_MESH, N_MESH, N_MESH), dtype=np.float32)
    times = {"cic": [], "fft": [], "interp": [], "total": []}
    for _ in range(N_BENCH_STEPS):
        t0 = time.time()
        cic_mass_assignment_cpu(pos, N_MESH, BOX_SIZE, density)
        t1 = time.time()
        fx, fy, fz = poisson_solve_cpu(density, greens, kx, ky, kz, N_MESH)
        t2 = time.time()
        accel = cic_force_interpolation_cpu(pos, fx, fy, fz, N_MESH, BOX_SIZE)
        t3 = time.time()
        times["cic"].append(t1 - t0)
        times["fft"].append(t2 - t1)
        times["interp"].append(t3 - t2)
        times["total"].append(t3 - t0)
    avg_total = np.mean(times["total"])
    results = {"per_step_avg": {k: np.mean(v) for k, v in times.items()}, "extrapolated_total": avg_total * N_FULL_STEPS}
    with open(os.path.join(data_dir, "cpu_benchmark.json"), "w") as f:
        json.dump(results, f)
    print("Benchmark complete. Saved to data/cpu_benchmark.json")