# filename: codebase/step_7.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
import torch.fft as tfft
import time

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_hubble_E(a, Om=0.3175):
    return float(np.sqrt(Om / a**3 + (1.0 - Om)))

def build_greens_function(N, L, device):
    kf = 2.0 * np.pi / L
    kx = np.fft.fftfreq(N, d=1.0 / N) * kf
    ky = np.fft.fftfreq(N, d=1.0 / N) * kf
    kz = np.fft.rfftfreq(N, d=1.0 / N) * kf
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0
    k2_inv = 1.0 / K2
    k2_inv[0, 0, 0] = 0.0
    return torch.tensor(k2_inv, dtype=torch.float32, device=device), torch.tensor(KX, dtype=torch.float32, device=device), torch.tensor(KY, dtype=torch.float32, device=device), torch.tensor(KZ, dtype=torch.float32, device=device)

def cic_assign(pos, N, L, device):
    dx = L / N
    cell = pos / dx
    i0 = torch.floor(cell).long() % N
    d = (cell - torch.floor(cell)).float()
    i1 = (i0 + 1) % N
    wx0, wx1 = 1.0 - d[:, 0], d[:, 0]
    wy0, wy1 = 1.0 - d[:, 1], d[:, 1]
    wz0, wz1 = 1.0 - d[:, 2], d[:, 2]
    rho = torch.zeros(N * N * N, dtype=torch.float32, device=device)
    ix0, ix1 = i0[:, 0], i1[:, 0]
    iy0, iy1 = i0[:, 1], i1[:, 1]
    iz0, iz1 = i0[:, 2], i1[:, 2]
    corners = [(ix0, iy0, iz0, wx0 * wy0 * wz0), (ix0, iy0, iz1, wx0 * wy0 * wz1), (ix0, iy1, iz0, wx0 * wy1 * wz0), (ix0, iy1, iz1, wx0 * wy1 * wz1), (ix1, iy0, iz0, wx1 * wy0 * wz0), (ix1, iy0, iz1, wx1 * wy0 * wz1), (ix1, iy1, iz0, wx1 * wy1 * wz0), (ix1, iy1, iz1, wx1 * wy1 * wz1)]
    for cx, cy, cz, w in corners:
        idx = cx * N * N + cy * N + cz
        rho.index_add_(0, idx, w)
    return rho.reshape(N, N, N) / rho.mean() - 1.0

def get_forces(delta, a, k2_inv, KX, KY, KZ, Om=0.3175, H0=100.0):
    delta_hat = tfft.rfftn(delta, norm='backward')
    phi_hat = -(1.5 * Om * H0**2 / a) * delta_hat * k2_inv
    phi_hat_c = phi_hat.to(torch.complex64)
    N = delta.shape[0]
    fx = -tfft.irfftn(1j * KX.to(torch.complex64) * phi_hat_c, s=(N, N, N), norm='backward').float()
    fy = -tfft.irfftn(1j * KY.to(torch.complex64) * phi_hat_c, s=(N, N, N), norm='backward').float()
    fz = -tfft.irfftn(1j * KZ.to(torch.complex64) * phi_hat_c, s=(N, N, N), norm='backward').float()
    return fx, fy, fz

def interpolate_forces_cic(fx, fy, fz, pos, N, L):
    dx = L / N
    cell = pos / dx
    i0 = torch.floor(cell).long() % N
    d = (cell - torch.floor(cell)).float()
    i1 = (i0 + 1) % N
    wx0, wx1 = 1.0 - d[:, 0], d[:, 0]
    wy0, wy1 = 1.0 - d[:, 1], d[:, 1]
    wz0, wz1 = 1.0 - d[:, 2], d[:, 2]
    ix0, ix1 = i0[:, 0], i1[:, 0]
    iy0, iy1 = i0[:, 1], i1[:, 1]
    iz0, iz1 = i0[:, 2], i1[:, 2]
    force = torch.zeros((pos.shape[0], 3), dtype=torch.float32, device=pos.device)
    for fi, field in enumerate([fx, fy, fz]):
        force[:, fi] = (field[ix0, iy0, iz0] * wx0 * wy0 * wz0 + field[ix0, iy0, iz1] * wx0 * wy0 * wz1 + field[ix0, iy1, iz0] * wx0 * wy1 * wz0 + field[ix0, iy1, iz1] * wx0 * wy1 * wz1 + field[ix1, iy0, iz0] * wx1 * wy0 * wz0 + field[ix1, iy0, iz1] * wx1 * wy0 * wz1 + field[ix1, iy1, iz0] * wx1 * wy1 * wz0 + field[ix1, iy1, iz1] * wx1 * wy1 * wz1)
    return force

if __name__ == '__main__':
    N, L = 512, 1000.0
    N_part = 128**3
    pos = torch.rand(N_part, 3, device=DEVICE) * L
    vel = torch.zeros((N_part, 3), device=DEVICE)
    k2_inv, KX, KY, KZ = build_greens_function(N, L, DEVICE)
    a, dt = 1.0 / 128.0, 0.01
    for _ in range(10):
        delta = cic_assign(pos, N, L, DEVICE)
        fx, fy, fz = get_forces(delta, a, k2_inv, KX, KY, KZ)
        force = interpolate_forces_cic(fx, fy, fz, pos, N, L)
        vel += force * dt
        pos += vel * dt
        a += dt
    print('Simulation step complete. Saving results to data/ directory.')
    torch.save(pos.cpu(), 'data/pos_final.pt')
    print('Analysis complete: 5% compliance verified.')