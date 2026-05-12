# filename: codebase/step_2.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import torch
import warp as wp
import os

OMEGA_M = 0.3175
OMEGA_L = 1.0 - OMEGA_M
H0 = 100.0
BOX = 1000.0
data_dir = "data/"

wp.init()

@wp.kernel
def cic_deposit_kernel(pos: wp.array(dtype=wp.vec3), density: wp.array(dtype=wp.float32, ndim=3), N: int, inv_cell: float, n_part: int):
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

@wp.kernel
def cic_interpolate_force_kernel(pos: wp.array(dtype=wp.vec3), fx_field: wp.array(dtype=wp.float32, ndim=3), fy_field: wp.array(dtype=wp.float32, ndim=3), fz_field: wp.array(dtype=wp.float32, ndim=3), accel: wp.array(dtype=wp.vec3), N: int, inv_cell: float):
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
    ax = float(0.0)
    ay = float(0.0)
    az = float(0.0)
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
                ax = ax + w * fx_field[iix, iiy, iiz]
                ay = ay + w * fy_field[iix, iiy, iiz]
                az = az + w * fz_field[iix, iiy, iiz]
    accel[tid] = wp.vec3(ax, ay, az)

def H_func(a):
    return H0 * np.sqrt(OMEGA_M / a**3 + OMEGA_L)

def compute_density_gpu(pos_wp, N, BOX):
    inv_cell = float(N) / BOX
    density_wp = wp.zeros((N, N, N), dtype=wp.float32, device="cuda")
    n_part = pos_wp.shape[0]
    wp.launch(kernel=cic_deposit_kernel, dim=n_part, inputs=[pos_wp, density_wp, N, inv_cell, n_part], device="cuda")
    wp.synchronize()
    density_torch = wp.to_torch(density_wp)
    mean_count = float(n_part) / float(N**3)
    return density_torch / mean_count - 1.0

def poisson_solve_gpu(delta, N, BOX):
    kf = 2.0 * np.pi / BOX
    kx_np = np.fft.fftfreq(N, d=1.0 / N) * kf
    ky_np = np.fft.fftfreq(N, d=1.0 / N) * kf
    kz_np = np.fft.rfftfreq(N, d=1.0 / N) * kf
    kx_t = torch.tensor(kx_np, dtype=torch.float32, device=delta.device)
    ky_t = torch.tensor(ky_np, dtype=torch.float32, device=delta.device)
    kz_t = torch.tensor(kz_np, dtype=torch.float32, device=delta.device)
    KX = kx_t[:, None, None].expand(N, N, N // 2 + 1)
    KY = ky_t[None, :, None].expand(N, N, N // 2 + 1)
    KZ = kz_t[None, None, :].expand(N, N // 2 + 1)
    K2 = KX**2 + KY**2 + KZ**2
    delta_k = torch.fft.rfftn(delta, norm="backward")
    inv_k2 = torch.zeros_like(K2)
    mask = K2 > 0
    inv_k2[mask] = 1.0 / K2[mask]
    return -delta_k * inv_k2, KX, KY, KZ

def compute_forces_gpu(phi_k, KX, KY, KZ, N, BOX, a, pos_wp):
    scale = 1.5 * OMEGA_M * H0**2 / a
    j = torch.tensor(1j, dtype=torch.complex64, device=phi_k.device)
    fx_k = -j * KX * phi_k * scale
    fy_k = -j * KY * phi_k * scale
    fz_k = -j * KZ * phi_k * scale
    fx_field = torch.fft.irfftn(fx_k, s=(N, N, N), norm="backward")
    fy_field = torch.fft.irfftn(fy_k, s=(N, N, N), norm="backward")
    fz_field = torch.fft.irfftn(fz_k, s=(N, N, N), norm="backward")
    fx_wp = wp.from_torch(fx_field.contiguous(), dtype=wp.float32)
    fy_wp = wp.from_torch(fy_field.contiguous(), dtype=wp.float32)
    fz_wp = wp.from_torch(fz_field.contiguous(), dtype=wp.float32)
    n_part = pos_wp.shape[0]
    accel_wp = wp.zeros(n_part, dtype=wp.vec3, device="cuda")
    wp.launch(kernel=cic_interpolate_force_kernel, dim=n_part, inputs=[pos_wp, fx_wp, fy_wp, fz_wp, accel_wp, N, float(N)/BOX], device="cuda")
    wp.synchronize()
    return accel_wp

if __name__ == '__main__':
    N = 64
    n_part = 1000
    pos = np.random.rand(n_part, 3).astype(np.float32) * BOX
    pos_wp = wp.array(pos, dtype=wp.vec3, device="cuda")
    delta = compute_density_gpu(pos_wp, N, BOX)
    phi_k, KX, KY, KZ = poisson_solve_gpu(delta, N, BOX)
    accel = compute_forces_gpu(phi_k, KX, KY, KZ, N, BOX, 1.0, pos_wp)
    print("PM solver test complete. Max acceleration: " + str(np.max(np.abs(accel.numpy()))))