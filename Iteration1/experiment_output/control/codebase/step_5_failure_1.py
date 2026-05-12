# filename: codebase/step_5.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
import warp as wp
import urllib.request
from step_2 import (cic_deposit_kernel, OMEGA_M, OMEGA_L, H0, BOX)
OMEGA_M_VAL = float(OMEGA_M)
OMEGA_L_VAL = float(OMEGA_L)
H0_VAL = float(H0)
BOX_VAL = float(BOX)
N_SIM = 512
N_PART = N_SIM ** 3
N_KBINS = 60
DATA_DIR = "data/"
ITER0_DATA_DIR = "/home/node/work/projects/cosmo_nbody_v1/Iteration0/experiment_output/control/data/"
SNAPSHOT_FILES = {0.0: "pos_z0.0.npy", 0.5: "pos_z0.5.npy", 1.0: "pos_z1.0.npy", 2.0: "pos_z2.0.npy"}
def compute_pk_from_positions_gpu(pos_np, N, box, n_kbins=60):
    n_part = len(pos_np)
    V = box ** 3
    pos_wp_pk = wp.array(pos_np, dtype=wp.vec3, device='cuda')
    density_wp_pk = wp.zeros((N, N, N), dtype=wp.float32, device='cuda')
    inv_cell = float(N) / box
    wp.launch(kernel=cic_deposit_kernel, dim=n_part, inputs=[pos_wp_pk, density_wp_pk, N, inv_cell], device='cuda')
    wp.synchronize()
    density_torch = wp.to_torch(density_wp_pk)
    mean_count = float(n_part) / float(N ** 3)
    delta = density_torch / mean_count - 1.0
    delta_k = torch.fft.rfftn(delta, norm='backward')
    kf = 2.0 * np.pi / box
    kN = np.pi * N / box
    kx_np = np.fft.fftfreq(N, d=1.0 / N) * kf
    ky_np = np.fft.fftfreq(N, d=1.0 / N) * kf
    kz_np = np.fft.rfftfreq(N, d=1.0 / N) * kf
    kx_t = torch.tensor(kx_np, dtype=torch.float32, device='cuda')
    ky_t = torch.tensor(ky_np, dtype=torch.float32, device='cuda')
    kz_t = torch.tensor(kz_np, dtype=torch.float32, device='cuda')
    KX_pk = kx_t[:, None, None].expand(N, N, N // 2 + 1).contiguous()
    KY_pk = ky_t[None, :, None].expand(N, N, N // 2 + 1).contiguous()
    KZ_pk = kz_t[None, None, :].expand(N, N, N // 2 + 1).contiguous()
    K2_pk = KX_pk ** 2 + KY_pk ** 2 + KZ_pk ** 2
    K_mag = torch.sqrt(K2_pk)
    sinc_x = torch.sinc(KX_pk / (2.0 * kN))
    sinc_y = torch.sinc(KY_pk / (2.0 * kN))
    sinc_z = torch.sinc(KZ_pk / (2.0 * kN))
    W_cic2 = (sinc_x * sinc_y * sinc_z) ** 2
    W_cic2 = torch.clamp(W_cic2, min=1e-10)
    pk_raw = (torch.abs(delta_k) ** 2) / W_cic2
    k_min = kf
    k_max = kN
    k_edges = np.logspace(np.log10(k_min), np.log10(k_max), n_kbins + 1)
    K_mag_np = K_mag.cpu().numpy().ravel()
    pk_raw_np = pk_raw.cpu().numpy().ravel()
    k_centers = np.zeros(n_kbins)
    pk_out = np.zeros(n_kbins)
    n_modes = np.zeros(n_kbins, dtype=np.int64)
    for i in range(n_kbins):
        mask_bin = (K_mag_np >= k_edges[i]) & (K_mag_np < k_edges[i + 1])
        cnt = np.sum(mask_bin)
        if cnt > 0:
            k_centers[i] = np.mean(K_mag_np[mask_bin])
            pk_out[i] = np.mean(pk_raw_np[mask_bin]) * V / float(N ** 6)
            n_modes[i] = cnt
    shot_noise = V / float(n_part)
    pk_out -= shot_noise
    valid = n_modes > 0
    return k_centers[valid], pk_out[valid], n_modes[valid]
def download_quijote_pk(save_path):
    url = 'https://raw.githubusercontent.com/franciscovillaescusa/Quijote-simulations/master/summary_statistics/Pk/Pk_m_z=0_0.txt'
    urllib.request.urlretrieve(url, save_path)
    data = np.loadtxt(save_path)
    return data
def find_nearest_k(k_arr, k_target):
    return int(np.argmin(np.abs(k_arr - k_target)))
if __name__ == '__main__':
    wp.init()
    pk_results = {}
    available_redshifts = []
    for z_val, fname in sorted(SNAPSHOT_FILES.items(), reverse=True):
        fpath = os.path.join(ITER0_DATA_DIR, fname)
        if os.path.exists(fpath):
            available_redshifts.append(z_val)
        else:
            print("WARNING: Snapshot for z=" + str(z_val) + " not found at " + fpath)
    print("Available snapshots: z = " + str(sorted(available_redshifts, reverse=True)))
    for z_val in sorted(available_redshifts, reverse=True):
        fname = SNAPSHOT_FILES[z_val]
        fpath = os.path.join(ITER0_DATA_DIR, fname)
        pos_np = np.load(fpath)
        if pos_np.ndim == 2 and pos_np.shape[1] == 3:
            pass
        else:
            pos_np = pos_np.reshape(-1, 3)
        pos_np = pos_np.astype(np.float32)
        k_c, pk_vals, n_modes = compute_pk_from_positions_gpu(pos_np, N_SIM, BOX_VAL, n_kbins=N_KBINS)
        pk_results[z_val] = (k_c, pk_vals, n_modes)
        out_fname = "pk_z" + str(z_val) + ".npy"
        np.save(os.path.join(DATA_DIR, out_fname), np.stack([k_c, pk_vals]))
        print("Saved P(k) for z=" + str(z_val) + " to " + os.path.join(DATA_DIR, out_fname))
    quijote_path = os.path.join(DATA_DIR, "quijote_pk_z0.txt")
    if not os.path.exists(quijote_path):
        ref_data = download_quijote_pk(quijote_path)
    else:
        ref_data = np.loadtxt(quijote_path)
    k_ref = ref_data[:, 0]
    pk_ref = ref_data[:, 1]
    print("\n--- P(k) values at key wavenumbers ---")
    for z_val in sorted(available_redshifts, reverse=True):
        k_c, pk_vals, _ = pk_results[z_val]
        print("z=" + str(z_val) + ":")
        for k_target in [0.1, 0.3, 1.0]:
            idx = find_nearest_k(k_c, k_target)
            print("  k=" + str(k_target) + " h/Mpc: P(k)=" + str(round(float(pk_vals[idx]), 4)) + " (Mpc/h)^3")
    if 0.0 in pk_results:
        k_c0, pk_z0, _ = pk_results[0.0]
        for k_target in [0.1, 0.3, 1.0]:
            idx_sim = find_nearest_k(k_c0, k_target)
            idx_ref = find_nearest_k(k_ref, k_target)
            ratio = float(pk_z0[idx_sim]) / float(pk_ref[idx_ref])
            print("  k=" + str(k_target) + " h/Mpc: Ratio P_warp/P_quijote=" + str(round(ratio, 4)))
        np.save(os.path.join(DATA_DIR, "pk_z0_final.npy"), np.stack([k_c0, pk_z0]))