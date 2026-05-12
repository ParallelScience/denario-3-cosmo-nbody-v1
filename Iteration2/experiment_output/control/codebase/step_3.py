# filename: codebase/step_3.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import time
import json

DATA_DIR = 'data/'
N = 512
L = 1000.0
N_STEPS_MEASURE = 15
N_STEPS_TOTAL = 500
OM = 0.3175
H0 = 100.0
Z_INIT = 127.0
A_INIT = 1.0 / (1.0 + Z_INIT)
A_FINAL = 1.0

def H_of_a(a, om=OM, h0=H0):
    return h0 * np.sqrt(om * a**(-3) + (1.0 - om))

def build_poisson_kernel(N, L):
    dk = 2.0 * np.pi / L
    kx = np.fft.fftfreq(N, d=1.0 / N) * dk
    ky = np.fft.fftfreq(N, d=1.0 / N) * dk
    kz = np.fft.rfftfreq(N, d=1.0 / N) * dk
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    k2_safe = np.where(K2 > 0, K2, 1.0)
    poisson_kernel = np.where(K2 > 0, -1.0 / k2_safe, 0.0)
    return KX, KY, KZ, poisson_kernel

def cic_mass_assign_cpu(pos, N, L):
    dx = L / N
    density = np.zeros((N, N, N), dtype=np.float32)
    cx, cy, cz = pos[:, 0] / dx, pos[:, 1] / dx, pos[:, 2] / dx
    ix, iy, iz = np.floor(cx).astype(np.int32) % N, np.floor(cy).astype(np.int32) % N, np.floor(cz).astype(np.int32) % N
    tx, ty, tz = cx - np.floor(cx), cy - np.floor(cy), cz - np.floor(cz)
    ix1, iy1, iz1 = (ix + 1) % N, (iy + 1) % N, (iz + 1) % N
    w000 = (1.0 - tx) * (1.0 - ty) * (1.0 - tz)
    w001 = (1.0 - tx) * (1.0 - ty) * tz
    w010 = (1.0 - tx) * ty * (1.0 - tz)
    w011 = (1.0 - tx) * ty * tz
    w100 = tx * (1.0 - ty) * (1.0 - tz)
    w101 = tx * (1.0 - ty) * tz
    w110 = tx * ty * (1.0 - tz)
    w111 = tx * ty * tz
    np.add.at(density, (ix, iy, iz), w000.astype(np.float32))
    np.add.at(density, (ix, iy, iz1), w001.astype(np.float32))
    np.add.at(density, (ix, iy1, iz), w010.astype(np.float32))
    np.add.at(density, (ix, iy1, iz1), w011.astype(np.float32))
    np.add.at(density, (ix1, iy, iz), w100.astype(np.float32))
    np.add.at(density, (ix1, iy, iz1), w101.astype(np.float32))
    np.add.at(density, (ix1, iy1, iz), w110.astype(np.float32))
    np.add.at(density, (ix1, iy1, iz1), w111.astype(np.float32))
    return density / density.mean() - 1.0

def poisson_solve_and_forces_cpu(density, KX, KY, KZ, poisson_kernel, a, N, L, om=OM, h0=H0):
    prefactor = 1.5 * om * h0**2 / a
    delta_k = np.fft.rfftn(density.astype(np.float64))
    phi_k = prefactor * poisson_kernel * delta_k
    fx = np.fft.irfftn(-1j * KX * phi_k, s=(N, N, N)).astype(np.float32)
    fy = np.fft.irfftn(-1j * KY * phi_k, s=(N, N, N)).astype(np.float32)
    fz = np.fft.irfftn(-1j * KZ * phi_k, s=(N, N, N)).astype(np.float32)
    return fx, fy, fz

def cic_force_interp_cpu(pos, fx, fy, fz, N, L):
    dx = L / N
    cx, cy, cz = pos[:, 0] / dx, pos[:, 1] / dx, pos[:, 2] / dx
    ix, iy, iz = np.floor(cx).astype(np.int32) % N, np.floor(cy).astype(np.int32) % N, np.floor(cz).astype(np.int32) % N
    tx, ty, tz = (cx - np.floor(cx)).astype(np.float32), (cy - np.floor(cy)).astype(np.float32), (cz - np.floor(cz)).astype(np.float32)
    ix1, iy1, iz1 = (ix + 1) % N, (iy + 1) % N, (iz + 1) % N
    w000, w001, w010, w011 = (1-tx)*(1-ty)*(1-tz), (1-tx)*(1-ty)*tz, (1-tx)*ty*(1-tz), (1-tx)*ty*tz
    w100, w101, w110, w111 = tx*(1-ty)*(1-tz), tx*(1-ty)*tz, tx*ty*(1-tz), tx*ty*tz
    def interp(f):
        return w000*f[ix,iy,iz] + w001*f[ix,iy,iz1] + w010*f[ix,iy1,iz] + w011*f[ix,iy1,iz1] + w100*f[ix1,iy,iz] + w101*f[ix1,iy,iz1] + w110*f[ix1,iy1,iz] + w111*f[ix1,iy1,iz1]
    return np.stack([interp(fx), interp(fy), interp(fz)], axis=-1)

if __name__ == '__main__':
    pos = np.random.rand(10000, 3).astype(np.float32) * L
    vel = np.zeros((10000, 3), dtype=np.float32)
    KX, KY, KZ, pk = build_poisson_kernel(N, L)
    times = {'cic': [], 'fft': [], 'force': [], 'int': []}
    for i in range(N_STEPS_MEASURE):
        t0 = time.time()
        rho = cic_mass_assign_cpu(pos, N, L)
        times['cic'].append(time.time() - t0)
        t0 = time.time()
        fx, fy, fz = poisson_solve_and_forces_cpu(rho, KX, KY, KZ, pk, 0.5, N, L)
        times['fft'].append(time.time() - t0)
        t0 = time.time()
        f = cic_force_interp_cpu(pos, fx, fy, fz, N, L)
        times['force'].append(time.time() - t0)
        t0 = time.time()
        vel += f * 0.01
        pos += vel * 0.01
        times['int'].append(time.time() - t0)
    results = {k: {'avg_step': np.mean(v), 'total_extrapolated': np.mean(v) * N_STEPS_TOTAL} for k, v in times.items()}
    with open(os.path.join(DATA_DIR, 'cpu_timing.json'), 'w') as f:
        json.dump(results, f)
    print('Saved timing to data/cpu_timing.json')