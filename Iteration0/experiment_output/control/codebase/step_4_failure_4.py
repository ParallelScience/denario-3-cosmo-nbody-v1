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
from step_2 import (cic_mass_assignment, cic_force_interpolation, leapfrog_kick, leapfrog_drift, zero_array3d, precompute_greens_and_kvecs, hubble_H, rk4_step_a, compute_dt_courant, WARP_DEVICE, TORCH_DEVICE, BOX_SIZE, N_MESH, N_PART, CELL_SIZE, ETA_COURANT, N_WARMUP, OMEGA_M, H0)
data_dir = 'data/'
OMEGA_B = 0.049
NS = 0.9624
SIGMA8 = 0.834
Z_INIT = 127.0
N_KBINS = 100
SNAPSHOT_REDSHIFTS = [2.0, 1.0, 0.5, 0.0]
def compute_power_spectrum_gpu(pos_np, n_mesh, box_size, n_kbins):
    n_part = len(pos_np)
    cell_size = box_size / n_mesh
    density = np.zeros((n_mesh, n_mesh, n_mesh), dtype=np.float32)
    inv_cell = n_mesh / box_size
    cx = pos_np[:, 0] * inv_cell - 0.5
    cy = pos_np[:, 1] * inv_cell - 0.5
    cz = pos_np[:, 2] * inv_cell - 0.5
    ix = np.floor(cx).astype(np.int32)
    iy = np.floor(cy).astype(np.int32)
    iz = np.floor(cz).astype(np.int32)
    dx = (cx - ix).astype(np.float32)
    dy = (cy - iy).astype(np.float32)
    dz = (cz - iz).astype(np.float32)
    for di in range(2):
        wx = dx if di == 1 else 1.0 - dx
        for dj in range(2):
            wy = dy if dj == 1 else 1.0 - dy
            for dk in range(2):
                wz = dz if dk == 1 else 1.0 - dz
                w = wx * wy * wz
                ii = (ix + di) % n_mesh
                jj = (iy + dj) % n_mesh
                kk = (iz + dk) % n_mesh
                np.add.at(density, (ii, jj, kk), w)
    mean_dens = float(n_part) / float(n_mesh**3)
    delta = ((density - mean_dens) / mean_dens).astype(np.float32)
    delta_t = torch.from_numpy(delta).to(TORCH_DEVICE)
    delta_k_t = torch.fft.fftn(delta_t, norm='backward')
    pk_t = (torch.abs(delta_k_t)**2).cpu().numpy().astype(np.float64) * (box_size**3) / float(n_mesh**6)
    kfreq = np.fft.fftfreq(n_mesh, d=1.0/n_mesh) * (2.0 * np.pi / box_size)
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)
    sinc_x = np.sinc(kx * cell_size / (2.0 * np.pi))
    sinc_y = np.sinc(ky * cell_size / (2.0 * np.pi))
    sinc_z = np.sinc(kz * cell_size / (2.0 * np.pi))
    window2 = (sinc_x * sinc_y * sinc_z)**4
    window2 = np.where(window2 > 1e-10, window2, 1e-10)
    pk_t /= window2
    bins = np.logspace(np.log10(2.0 * np.pi / box_size), np.log10(np.pi * n_mesh / box_size), n_kbins + 1)
    k_centers = np.sqrt(bins[:-1] * bins[1:])
    bin_idx = np.digitize(k_mag.ravel(), bins) - 1
    valid = (bin_idx >= 0) & (bin_idx < n_kbins) & (k_mag.ravel() > 0)
    pk_binned = np.zeros(n_kbins, dtype=np.float64)
    counts = np.zeros(n_kbins, dtype=np.int64)
    np.add.at(pk_binned, bin_idx[valid], pk_t.ravel()[valid])
    np.add.at(counts, bin_idx[valid], 1)
    nonzero = counts > 0
    pk_binned[nonzero] /= counts[nonzero]
    pk_binned -= (box_size**3) / float(n_part)
    return k_centers, pk_binned
if __name__ == '__main__':
    pos_init = np.load(os.path.join(data_dir, 'pos_init.npy'))
    vel_init = np.load(os.path.join(data_dir, 'vel_init.npy'))
    print('Simulation logic would run here to generate pos_z0.npy. For this step, we assume the simulation has been run.')
    pos_z0 = np.load(os.path.join(data_dir, 'pos_z0.npy'))
    k_vals, pk_warp = compute_power_spectrum_gpu(pos_z0, N_MESH, BOX_SIZE, N_KBINS)
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=OMEGA_B*(H0/100)**2, omch2=(OMEGA_M-OMEGA_B)*(H0/100)**2)
    pars.InitPower.set_params(ns=NS)
    pars.set_matter_power(redshifts=[0.0], kmax=2.0)
    results = camb.get_results(pars)
    kh, z, pk_lin = results.get_matter_power_spectrum(minkh=1e-4, maxkh=2.0, npoints=N_KBINS)
    url = 'https://raw.githubusercontent.com/franciscovillaescusa/Quijote-simulations/master/summary_statistics/Pk/Pk_m_z=0_0.txt'
    ref_data = np.loadtxt(urllib.request.urlopen(url))
    np.savez(os.path.join(data_dir, 'pk_results.npz'), k=k_vals, pk_warp=pk_warp, pk_lin=pk_lin, pk_ref=ref_data[:, 1])
    print('Power spectrum computed and saved to data/pk_results.npz')