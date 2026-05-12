# filename: codebase/step_6.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
import warp as wp
import time
import matplotlib
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
from step_1 import (build_k_grids, interpolate_pk, generate_gaussian_field, compute_psi1, compute_psi2, compute_camb_pk)

DATA_DIR = 'data/'
N = 512
L = 1000.0
N_REAL = 10
OM = 0.3175
OB = 0.049
H_PARAM = 0.6711
NS = 0.9624
S8 = 0.834
H0 = 100.0
Z_INIT = 127.0
A_INIT = 1.0 / (1.0 + Z_INIT)

def H_of_a(a, om=OM, h0=H0):
    return h0 * np.sqrt(om * a ** (-3) + (1.0 - om))

def generate_2lpt_ics(seed, k_camb, pk_zinit):
    KX, KY, KZ, K2, K = build_k_grids(N, L)
    pk_grid = interpolate_pk(k_camb, pk_zinit, K)
    delta_k = generate_gaussian_field(N, L, pk_grid, seed)
    psi1_x, psi1_y, psi1_z, phi1_k = compute_psi1(delta_k, KX, KY, KZ, K2, N)
    psi2_x, psi2_y, psi2_z = compute_psi2(phi1_k, KX, KY, KZ, K2, N)
    a_i = A_INIT
    D1 = 1.0
    D2 = -3.0 / 7.0 * D1 ** 2
    f1 = 1.0
    f2 = 2.0 * f1
    H_init = H_of_a(a_i)
    x_grid = np.linspace(0, L, N, endpoint=False)
    X, Y, Z_g = np.meshgrid(x_grid, x_grid, x_grid, indexing='ij')
    pos_x = (X + D1 * psi1_x + D2 * psi2_x) % L
    pos_y = (Y + D1 * psi1_y + D2 * psi2_y) % L
    pos_z = (Z_g + D1 * psi1_z + D2 * psi2_z) % L
    vel_x = a_i * H_init * f1 * psi1_x + 2.0 * a_i * H_init * f2 * psi2_x
    vel_y = a_i * H_init * f1 * psi1_y + 2.0 * a_i * H_init * f2 * psi2_y
    vel_z = a_i * H_init * f1 * psi1_z + 2.0 * a_i * H_init * f2 * psi2_z
    pos = np.stack([pos_x.ravel(), pos_y.ravel(), pos_z.ravel()], axis=-1).astype(np.float32)
    vel = np.stack([vel_x.ravel(), vel_y.ravel(), vel_z.ravel()], axis=-1).astype(np.float32)
    return pos, vel

if __name__ == '__main__':
    k_camb, pk_zinit = compute_camb_pk(OM, OB, H_PARAM, NS, S8)
    for i in range(N_REAL):
        pos, vel = generate_2lpt_ics(i, k_camb, pk_zinit)
        np.save(os.path.join(DATA_DIR, 'pos_' + str(i) + '.npy'), pos)
        np.save(os.path.join(DATA_DIR, 'vel_' + str(i) + '.npy'), vel)
    print('Simulation data generated and saved to ' + DATA_DIR)
    fig, ax = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
    ax[0].set_title('Cosmological N-body Simulation P(k) Comparison')
    ax[0].set_ylabel('P(k) [(Mpc/h)^3]')
    ax[1].set_ylabel('Ratio P_warp / P_quijote')
    ax[1].set_xlabel('k [h/Mpc]')
    plt.savefig(os.path.join(DATA_DIR, 'pk_comparison_' + str(int(time.time())) + '.png'))
    print('Saved to ' + os.path.join(DATA_DIR, 'pk_comparison.png'))