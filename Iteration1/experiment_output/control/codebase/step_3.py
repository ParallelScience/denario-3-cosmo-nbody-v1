# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
import warp as wp
import time
from step_2 import (cic_deposit_kernel, cic_interpolate_force_kernel, build_k_grids, OMEGA_M, OMEGA_L, H0, BOX)
def H_func(a):
    return 100.0 * np.sqrt(OMEGA_M / a**3 + OMEGA_L)
def compute_density_gpu_preallocated(pos_wp, density_wp, N, BOX, n_part):
    inv_cell = float(N) / BOX
    density_wp.zero_()
    wp.launch(kernel=cic_deposit_kernel, dim=n_part, inputs=[pos_wp, density_wp, N, inv_cell], device='cuda')
    wp.synchronize()
    density_torch = wp.to_torch(density_wp)
    mean_count = float(n_part) / float(N**3)
    return density_torch / mean_count - 1.0
def poisson_solve_gpu(delta, K2):
    delta_k = torch.fft.rfftn(delta, norm='backward')
    inv_k2 = torch.zeros_like(K2)
    mask = K2 > 0
    inv_k2[mask] = 1.0 / K2[mask]
    return -delta_k * inv_k2
def compute_forces_gpu_preallocated(phi_k, KX, KY, KZ, N, BOX, a, pos_wp, accel_wp, fx_field_wp, fy_field_wp, fz_field_wp):
    scale = float(1.5 * OMEGA_M * H0**2 / a)
    phi_k_c = phi_k.to(torch.complex64)
    j = torch.tensor(1j, dtype=torch.complex64, device=phi_k.device)
    fx_k = (-j * scale) * KX.to(torch.complex64) * phi_k_c
    fy_k = (-j * scale) * KY.to(torch.complex64) * phi_k_c
    fz_k = (-j * scale) * KZ.to(torch.complex64) * phi_k_c
    fx_field = torch.fft.irfftn(fx_k, s=(N, N, N), norm='backward')
    fy_field = torch.fft.irfftn(fy_k, s=(N, N, N), norm='backward')
    fz_field = torch.fft.irfftn(fz_k, s=(N, N, N), norm='backward')
    fx_field_wp.assign(wp.from_torch(fx_field.contiguous(), dtype=wp.float32))
    fy_field_wp.assign(wp.from_torch(fy_field.contiguous(), dtype=wp.float32))
    fz_field_wp.assign(wp.from_torch(fz_field.contiguous(), dtype=wp.float32))
    n_part = pos_wp.shape[0]
    wp.launch(kernel=cic_interpolate_force_kernel, dim=n_part, inputs=[pos_wp, fx_field_wp, fy_field_wp, fz_field_wp, accel_wp, N, float(N) / BOX], device='cuda')
    wp.synchronize()
    return accel_wp
@wp.kernel
def kick_kernel(vel: wp.array(dtype=wp.vec3), accel: wp.array(dtype=wp.vec3), dt_kick: float):
    tid = wp.tid()
    vel[tid] = vel[tid] + accel[tid] * dt_kick
@wp.kernel
def drift_kernel(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), dt_drift: float, box: float):
    tid = wp.tid()
    p = pos[tid] + vel[tid] * dt_drift
    p[0] = p[0] - wp.floor(p[0] / box) * box
    p[1] = p[1] - wp.floor(p[1] / box) * box
    p[2] = p[2] - wp.floor(p[2] / box) * box
    pos[tid] = p
def vram_gb():
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return alloc, reserved
def print_vram(label):
    alloc, reserved = vram_gb()
    print('VRAM [' + label + ']: allocated=' + str(round(alloc, 3)) + ' GiB, reserved=' + str(round(reserved, 3)) + ' GiB')
def build_time_steps(a_init, a_final, n_steps):
    eta = (a_final / a_init)**(1.0 / n_steps) - 1.0
    a_vals = [a_init]
    a = a_init
    for _ in range(n_steps):
        da = eta * a
        a = a + da
        if a > a_final: a = a_final
        a_vals.append(a)
    a_vals[-1] = a_final
    return a_vals
if __name__ == '__main__':
    N = 512
    N_PART = N**3
    N_STEPS = 80
    A_INIT = 1.0 / 128.0
    A_FINAL = 1.0
    data_dir = 'data/'
    wp.init()
    pos_np = np.load(os.path.join(data_dir, 'pos.npy'))
    vel_np = np.load(os.path.join(data_dir, 'vel.npy'))
    pos_wp = wp.array(pos_np, dtype=wp.vec3, device='cuda')
    vel_wp = wp.array(vel_np, dtype=wp.vec3, device='cuda')
    density_wp = wp.zeros((N, N, N), dtype=wp.float32, device='cuda')
    accel_wp = wp.zeros(N_PART, dtype=wp.vec3, device='cuda')
    fx_field_wp = wp.zeros((N, N, N), dtype=wp.float32, device='cuda')
    fy_field_wp = wp.zeros((N, N, N), dtype=wp.float32, device='cuda')
    fz_field_wp = wp.zeros((N, N, N), dtype=wp.float32, device='cuda')
    KX, KY, KZ, K2 = build_k_grids(N, BOX, 'cuda')
    a_vals = build_time_steps(A_INIT, A_FINAL, N_STEPS)
    a = a_vals[0]
    for i in range(N_STEPS):
        a_next = a_vals[i+1]
        da = a_next - a
        dt_kick = da / (a**2 * H_func(a))
        wp.launch(kernel=kick_kernel, dim=N_PART, inputs=[vel_wp, accel_wp, dt_kick], device='cuda')
        dt_drift = da / (a**2 * H_func(a))
        wp.launch(kernel=drift_kernel, dim=N_PART, inputs=[pos_wp, vel_wp, dt_drift, BOX], device='cuda')
        delta = compute_density_gpu_preallocated(pos_wp, density_wp, N, BOX, N_PART)
        phi_k = poisson_solve_gpu(delta, K2)
        accel_wp = compute_forces_gpu_preallocated(phi_k, KX, KY, KZ, N, BOX, a_next, pos_wp, accel_wp, fx_field_wp, fy_field_wp, fz_field_wp)
        wp.launch(kernel=kick_kernel, dim=N_PART, inputs=[vel_wp, accel_wp, dt_kick], device='cuda')
        a = a_next
    print('Simulation complete.')