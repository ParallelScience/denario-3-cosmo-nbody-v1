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
from step_1 import compute_camb_pk, compute_hubble, compute_growth_rate
from step_2 import (cic_mass_assignment, cic_force_interpolation, leapfrog_kick, leapfrog_drift, zero_array3d, precompute_greens_and_kvecs, hubble_H, rk4_step_a, compute_dt_courant, WARP_DEVICE, TORCH_DEVICE, BOX_SIZE, N_MESH, N_PART, CELL_SIZE, OMEGA_M, OMEGA_LAMBDA, H0, H_PARAM, SNAPSHOT_REDSHIFTS, SNAPSHOT_SCALE_FACTORS, ETA_COURANT, N_WARMUP)
def run_full_gpu_simulation(pos_init_np, vel_init_np, a_init, greens, kx_t, ky_t, kz_t, data_dir):
    n_part = pos_init_np.shape[0]
    pos_wp = wp.array(pos_init_np, dtype=wp.vec3, device=WARP_DEVICE)
    vel_wp = wp.array(vel_init_np, dtype=wp.vec3, device=WARP_DEVICE)
    accel_wp = wp.zeros(n_part, dtype=wp.vec3, device=WARP_DEVICE)
    density_wp = wp.zeros((N_MESH, N_MESH, N_MESH), dtype=wp.float32, device=WARP_DEVICE)
    force_x_wp = wp.zeros((N_MESH, N_MESH, N_MESH), dtype=wp.float32, device=WARP_DEVICE)
    force_y_wp = wp.zeros((N_MESH, N_MESH, N_MESH), dtype=wp.float32, device=WARP_DEVICE)
    force_z_wp = wp.zeros((N_MESH, N_MESH, N_MESH), dtype=wp.float32, device=WARP_DEVICE)
    mean_density_val = float(n_part) / float(N_MESH ** 3)
    def compute_forces(pos_w, dens_w, fx_w, fy_w, fz_w):
        wp.launch(zero_array3d, dim=(N_MESH, N_MESH, N_MESH), inputs=[dens_w], device=WARP_DEVICE)
        wp.launch(cic_mass_assignment, dim=n_part, inputs=[pos_w, dens_w, N_MESH, BOX_SIZE], device=WARP_DEVICE)
        wp.synchronize()
        dens_torch = torch.as_tensor(dens_w, device=TORCH_DEVICE)
        delta = (dens_torch - mean_density_val) / mean_density_val
        delta_k = torch.fft.rfftn(delta, norm='backward')
        phi_k = delta_k * greens
        fx_k = -1j * kx_t * phi_k
        fy_k = -1j * ky_t * phi_k
        fz_k = -1j * kz_t * phi_k
        fx_real = torch.fft.irfftn(fx_k, s=(N_MESH, N_MESH, N_MESH), norm='backward').to(torch.float32).contiguous()
        fy_real = torch.fft.irfftn(fy_k, s=(N_MESH, N_MESH, N_MESH), norm='backward').to(torch.float32).contiguous()
        fz_real = torch.fft.irfftn(fz_k, s=(N_MESH, N_MESH, N_MESH), norm='backward').to(torch.float32).contiguous()
        fx_wp_local = wp.from_torch(fx_real, dtype=wp.float32)
        fy_wp_local = wp.from_torch(fy_real, dtype=wp.float32)
        fz_wp_local = wp.from_torch(fz_real, dtype=wp.float32)
        wp.launch(cic_force_interpolation, dim=n_part, inputs=[pos_w, fx_wp_local, fy_wp_local, fz_wp_local, accel_wp, N_MESH, BOX_SIZE], device=WARP_DEVICE)
        wp.synchronize()
    a = a_init
    snapshot_scale_factors = sorted(SNAPSHOT_SCALE_FACTORS)
    snap_idx = 0
    step = 0
    compute_forces(pos_wp, density_wp, force_x_wp, force_y_wp, force_z_wp)
    vel_arr = wp.to_torch(vel_wp)
    accel_arr = wp.to_torch(accel_wp)
    vel_arr.add_(accel_arr * (-0.5 * (1.0 / (a * hubble_H(a)))))
    wp.synchronize()
    while a < 1.0:
        vel_np_mag = torch.norm(wp.to_torch(vel_wp), dim=1).max().item()
        dt_mpc = compute_dt_courant(a, vel_np_mag, CELL_SIZE, ETA_COURANT)
        a_new = rk4_step_a(a, dt_mpc)
        if a_new > 1.0:
            dt_mpc = dt_mpc * (1.0 - a) / (a_new - a)
            a_new = 1.0
        H_a = hubble_H(a)
        dt_kick = dt_mpc / (a * H_a)
        wp.launch(leapfrog_kick, dim=n_part, inputs=[vel_wp, accel_wp, dt_kick], device=WARP_DEVICE)
        wp.launch(leapfrog_drift, dim=n_part, inputs=[pos_wp, vel_wp, dt_kick, BOX_SIZE], device=WARP_DEVICE)
        wp.synchronize()
        a = a_new
        while snap_idx < len(snapshot_scale_factors) and a >= snapshot_scale_factors[snap_idx]:
            z_snap = 1.0 / snapshot_scale_factors[snap_idx] - 1.0
            pos_snap = wp.to_torch(pos_wp).cpu().numpy()
            snap_fname = os.path.join(data_dir, 'pos_z' + str(round(z_snap, 1)) + '.npy')
            np.save(snap_fname, pos_snap)
            snap_idx += 1
        compute_forces(pos_wp, density_wp, force_x_wp, force_y_wp, force_z_wp)
        H_a_new = hubble_H(a)
        dt_kick_new = dt_mpc / (a * H_a_new)
        wp.launch(leapfrog_kick, dim=n_part, inputs=[vel_wp, accel_wp, dt_kick_new], device=WARP_DEVICE)
        wp.synchronize()
        step += 1
    return wp.to_torch(pos_wp).cpu().numpy()
def cic_density_gpu(pos_np, n_mesh, box_size):
    n_part = pos_np.shape[0]
    pos_wp = wp.array(pos_np, dtype=wp.vec3, device=WARP_DEVICE)
    density_wp = wp.zeros((n_mesh, n_mesh, n_mesh), dtype=wp.float32, device=WARP_DEVICE)
    wp.launch(cic_mass_assignment, dim=n_part, inputs=[pos_wp, density_wp, n_mesh, box_size], device=WARP_DEVICE)
    wp.synchronize()
    return wp.to_torch(density_wp).cpu().numpy()
def compute_power_spectrum(pos_np, n_mesh, box_size):
    n_part = pos_np.shape[0]
    cell_size = box_size / n_mesh
    vol = box_size ** 3
    density = cic_density_gpu(pos_np, n_mesh, box_size)
    mean_n = float(n_part) / float(n_mesh ** 3)
    delta = (density - mean_n) / mean_n
    delta_k = np.fft.rfftn(delta)
    pk_raw = (np.abs(delta_k) ** 2) * (vol / float(n_part) ** 2)
    dk = 2.0 * np.pi / box_size
    kx_1d = np.fft.fftfreq(n_mesh, d=1.0 / n_mesh) * dk
    ky_1d = np.fft.fftfreq(n_mesh, d=1.0 / n_mesh) * dk
    kz_1d = np.fft.rfftfreq(n_mesh, d=1.0 / n_mesh) * dk
    kx_g, ky_g, kz_g = np.meshgrid(kx_1d, ky_1d, kz_1d, indexing='ij')
    k_mag = np.sqrt(kx_g ** 2 + ky_g ** 2 + kz_g ** 2)
    half_delta = cell_size / 2.0
    sinc_x = np.sinc(kx_g * half_delta / np.pi)
    sinc_y = np.sinc(ky_g * half_delta / np.pi)
    sinc_z = np.sinc(kz_g * half_delta / np.pi)
    window2 = (sinc_x * sinc_y * sinc_z) ** 4
    window2 = np.where(window2 > 1e-10, window2, 1.0)
    pk_deconv = pk_raw / window2
    pk_deconv -= (vol / float(n_part))
    k_bins = np.linspace(0, k_mag.max(), 50)
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
    pk_binned = np.zeros_like(k_centers)
    n_modes = np.zeros_like(k_centers)
    for i in range(len(k_bins) - 1):
        mask = (k_mag >= k_bins[i]) & (k_mag < k_bins[i+1])
        if np.sum(mask) > 0:
            pk_binned[i] = np.mean(pk_deconv[mask])
            n_modes[i] = np.sum(mask)
    return k_centers, pk_binned, n_modes
if __name__ == '__main__':
    data_dir = 'data/'
    pos_init = np.load(os.path.join(data_dir, 'pos_init.npy'))
    vel_init = np.load(os.path.join(data_dir, 'vel_init.npy'))
    a_init = 1.0 / 128.0
    greens, kx_t, ky_t, kz_t = precompute_greens_and_kvecs(N_MESH, BOX_SIZE)
    pos_final = run_full_gpu_simulation(pos_init, vel_init, a_init, greens, kx_t, ky_t, kz_t, data_dir)
    k_centers, pk_warp, n_modes = compute_power_spectrum(pos_final, N_MESH, BOX_SIZE)
    np.save(os.path.join(data_dir, 'pk_warp.npy'), pk_warp)
    np.save(os.path.join(data_dir, 'k_bins.npy'), k_centers)
    print('Power spectrum computed and saved.')