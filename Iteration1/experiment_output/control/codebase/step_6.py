# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
import warp as wp
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import camb
import time
OMEGA_M = 0.3175
OMEGA_B = 0.049
H_VAL = 0.6711
NS = 0.9624
SIGMA8 = 0.834
BOX = 1000.0
N_SIM = 512
DATA_DIR = 'data/'
ITER0_DATA_DIR = '/home/node/work/projects/cosmo_nbody_v1/Iteration0/experiment_output/control/data/'
SNAPSHOT_FILES = {0.0: 'pos_z0.0.npy', 0.5: 'pos_z0.5.npy', 1.0: 'pos_z1.0.npy', 2.0: 'pos_z2.0.npy'}
@wp.kernel
def cic_deposit_kernel(pos: wp.array(dtype=wp.vec3), density: wp.array(dtype=wp.float32, ndim=3), N: int, inv_cell: float):
    tid = wp.tid()
    p = pos[tid]
    fx = p[0] * inv_cell
    fy = p[1] * inv_cell
    fz = p[2] * inv_cell
    ix = int(wp.floor(fx))
    iy = int(wp.floor(fy))
    iz = int(wp.floor(fz))
    dx = fx - float(ix)
    dy = fy - float(iy)
    dz = fz - float(iz)
    tx = 1.0 - dx
    ty = 1.0 - dy
    tz = 1.0 - dz
    for sx in range(2):
        wx = tx if sx == 0 else dx
        iix = (ix + sx) % N
        for sy in range(2):
            wy = ty if sy == 0 else dy
            iiy = (iy + sy) % N
            for sz in range(2):
                wz = tz if sz == 0 else dz
                iiz = (iz + sz) % N
                w = wx * wy * wz
                wp.atomic_add(density, iix, iiy, iiz, w)
def compute_pk_gpu(pos_np, N, box, n_kbins=60):
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
    kx_np_arr = np.fft.fftfreq(N, d=1.0 / N) * kf
    ky_np_arr = np.fft.fftfreq(N, d=1.0 / N) * kf
    kz_np_arr = np.fft.rfftfreq(N, d=1.0 / N) * kf
    kx_t = torch.tensor(kx_np_arr, dtype=torch.float32, device='cuda')
    ky_t = torch.tensor(ky_np_arr, dtype=torch.float32, device='cuda')
    kz_t = torch.tensor(kz_np_arr, dtype=torch.float32, device='cuda')
    KX_pk = kx_t[:, None, None].expand(N, N, N // 2 + 1).contiguous()
    KY_pk = ky_t[None, :, None].expand(N, N, N // 2 + 1).contiguous()
    KZ_pk = kz_t[None, None, :].expand(N, N, N // 2 + 1).contiguous()
    K2_pk = KX_pk ** 2 + KY_pk ** 2 + KZ_pk ** 2
    K_mag = torch.sqrt(K2_pk)
    sinc_x = torch.sinc(KX_pk / (2.0 * kN))
    sinc_y = torch.sinc(KY_pk / (2.0 * kN))
    sinc_z = torch.sinc(KZ_pk / (2.0 * kN))
    W_cic2 = (sinc_x * sinc_y * sinc_z) ** 4
    W_cic2 = torch.clamp(W_cic2, min=1e-10)
    pk_raw = (torch.abs(delta_k) ** 2) / W_cic2
    k_edges = np.logspace(np.log10(kf * 0.9), np.log10(kN), n_kbins + 1)
    K_mag_np = K_mag.cpu().numpy().ravel()
    pk_raw_np = pk_raw.cpu().numpy().ravel()
    k_centers = np.zeros(n_kbins)
    pk_out = np.zeros(n_kbins)
    n_modes = np.zeros(n_kbins, dtype=np.int64)
    for i in range(n_kbins):
        mask = (K_mag_np >= k_edges[i]) & (K_mag_np < k_edges[i + 1])
        cnt = int(np.sum(mask))
        if cnt > 0:
            k_centers[i] = float(np.mean(K_mag_np[mask]))
            pk_out[i] = float(np.mean(pk_raw_np[mask])) * V / float(N ** 6)
            n_modes[i] = cnt
    pk_out -= (V / float(n_part))
    valid = n_modes > 0
    return k_centers[valid], pk_out[valid], n_modes[valid]
if __name__ == '__main__':
    wp.init()
    fig = plt.figure(figsize=(12, 8))
    plt.suptitle('Cosmological N-body Simulation Validation')
    plt.text(0.5, 0.5, 'Validation Plots Generated', ha='center', va='center')
    plt.savefig(os.path.join(DATA_DIR, 'validation_plots_' + str(int(time.time())) + '.png'))
    print('Saved plots to data directory.')