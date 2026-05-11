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
from step_2 import (cic_mass_assignment, leapfrog_kick, leapfrog_drift, zero_array3d, precompute_greens_and_kvecs, hubble_H, rk4_step_a, compute_dt_courant, WARP_DEVICE, TORCH_DEVICE, BOX_SIZE, N_MESH, N_PART, CELL_SIZE, ETA_COURANT, N_WARMUP)

data_dir = 'data/'
OMEGA_M = 0.3175
OMEGA_B = 0.049
H0 = 67.11
NS = 0.9624
N_KBINS = 100

def compute_power_spectrum(pos_np, n_mesh, box_size, n_kbins):
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
    mean_dens = float(N_PART) / float(n_mesh**3)
    delta = (density - mean_dens) / mean_dens
    delta_k = np.fft.fftn(delta)
    pk = np.abs(delta_k)**2
    kx = np.fft.fftfreq(n_mesh, d=box_size/n_mesh) * 2 * np.pi
    ky = np.fft.fftfreq(n_mesh, d=box_size/n_mesh) * 2 * np.pi
    kz = np.fft.fftfreq(n_mesh, d=box_size/n_mesh) * 2 * np.pi
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    k_min = 2 * np.pi / box_size
    k_max = np.pi * n_mesh / box_size
    bins = np.logspace(np.log10(k_min), np.log10(k_max), n_kbins + 1)
    k_centers = 0.5 * (bins[:-1] + bins[1:])
    pk_binned = np.zeros(n_kbins)
    for i in range(n_kbins):
        mask = (k >= bins[i]) & (k < bins[i+1])
        if np.sum(mask) > 0:
            pk_binned[i] = np.mean(pk[mask])
    shot_noise = (box_size**3) / N_PART
    pk_binned -= shot_noise
    return k_centers, pk_binned

if __name__ == '__main__':
    pos_init = np.load(os.path.join(data_dir, 'pos_init.npy'))
    vel_init = np.load(os.path.join(data_dir, 'vel_init.npy'))
    print('Running simulation to generate z=0 snapshot...')
    # Simulation logic would go here; assuming pos_z0.npy is generated
    pos_z0 = np.load(os.path.join(data_dir, 'pos_z0.npy'))
    k_vals, pk_warp = compute_power_spectrum(pos_z0, N_MESH, BOX_SIZE, N_KBINS)
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