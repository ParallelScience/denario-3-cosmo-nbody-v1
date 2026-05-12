# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import camb
from scipy.interpolate import interp1d
import os

def H_func(a, omega_m, omega_l):
    return 100.0 * np.sqrt(omega_m / a**3 + omega_l)

def growth_rate_f(a, omega_m, omega_l):
    H_a = H_func(a, omega_m, omega_l)
    omega_m_a = omega_m / (a**3 * (H_a / 100.0)**2)
    return omega_m_a**0.55

def compute_camb_pk(omega_m, omega_b, h, ns, sigma8, z_init, n_k=2000):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=h * 100.0, ombh2=omega_b * h**2, omch2=(omega_m - omega_b) * h**2, omk=0, mnu=0.0, neutrino_hierarchy='degenerate')
    pars.InitPower.set_params(ns=ns, As=2.0e-9)
    pars.set_matter_power(redshifts=[0.0, z_init], kmax=20.0, k_per_logint=20)
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    s8_all = results.get_sigma8()
    sigma8_camb_z0 = s8_all[-1]
    rescale = (sigma8 / sigma8_camb_z0)**2
    k_arr, z_arr, pk_arr = results.get_matter_power_spectrum(minkh=1e-4, maxkh=20.0, npoints=n_k)
    pk_z0 = pk_arr[0] * rescale
    pk_zinit = pk_arr[1] * rescale
    return k_arr, pk_z0, pk_zinit, sigma8_camb_z0 * np.sqrt(rescale)

def generate_gaussian_field(N, BOX, pk_interp, seed=0):
    rng = np.random.default_rng(seed)
    V = BOX**3
    kf = 2.0 * np.pi / BOX
    kx = np.fft.fftfreq(N, d=1.0 / N) * kf
    ky = np.fft.fftfreq(N, d=1.0 / N) * kf
    kz = np.fft.rfftfreq(N, d=1.0 / N) * kf
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K_mag = np.sqrt(K2)
    pk_grid = np.zeros_like(K_mag)
    mask = K_mag > 0
    pk_grid[mask] = np.maximum(pk_interp(K_mag[mask]), 0.0)
    amplitude = (N**3 / np.sqrt(2.0 * V)) * np.sqrt(pk_grid)
    noise_real = rng.standard_normal(size=(N, N, N // 2 + 1))
    noise_imag = rng.standard_normal(size=(N, N, N // 2 + 1))
    delta_k = amplitude * (noise_real + 1j * noise_imag)
    delta_k[0, 0, 0] = 0.0
    return delta_k, KX, KY, KZ, K2

def compute_za_displacement(delta_k, KX, KY, KZ, K2, N, BOX):
    inv_k2 = np.zeros_like(K2)
    mask = K2 > 0
    inv_k2[mask] = 1.0 / K2[mask]
    psi_x_k = -1j * KX * inv_k2 * delta_k
    psi_y_k = -1j * KY * inv_k2 * delta_k
    psi_z_k = -1j * KZ * inv_k2 * delta_k
    norm = 1.0 / N**3
    psi_x = np.fft.irfftn(psi_x_k, s=(N, N, N)) * norm
    psi_y = np.fft.irfftn(psi_y_k, s=(N, N, N)) * norm
    psi_z = np.fft.irfftn(psi_z_k, s=(N, N, N)) * norm
    return psi_x, psi_y, psi_z

def make_regular_grid(N, BOX):
    dx = BOX / N
    q1d = np.arange(N, dtype=np.float32) * dx + 0.5 * dx
    qx, qy, qz = np.meshgrid(q1d, q1d, q1d, indexing='ij')
    return qx.ravel(), qy.ravel(), qz.ravel()

def cic_mass_assignment(pos_x, pos_y, pos_z, N, BOX):
    cell_size = BOX / N
    N_part = len(pos_x)
    ix = pos_x / cell_size
    iy = pos_y / cell_size
    iz = pos_z / cell_size
    i0x = np.floor(ix).astype(np.int32)
    i0y = np.floor(iy).astype(np.int32)
    i0z = np.floor(iz).astype(np.int32)
    dx = ix - i0x
    dy = iy - i0y
    dz = iz - i0z
    tx = 1.0 - dx
    ty = 1.0 - dy
    tz = 1.0 - dz
    density = np.zeros((N, N, N), dtype=np.float64)
    for sx in range(2):
        wx = tx if sx == 0 else dx
        ix_s = (i0x + sx) % N
        for sy in range(2):
            wy = ty if sy == 0 else dy
            iy_s = (i0y + sy) % N
            for sz in range(2):
                wz = tz if sz == 0 else dz
                iz_s = (i0z + sz) % N
                w = wx * wy * wz
                np.add.at(density, (ix_s, iy_s, iz_s), w)
    mean_density = N_part / N**3
    delta = density / mean_density - 1.0
    return delta

if __name__ == '__main__':
    N = 512
    BOX = 1000.0
    z_init = 127.0
    omega_m = 0.3175
    omega_b = 0.049
    h = 0.6711
    ns = 0.9624
    sigma8 = 0.834
    omega_l = 1.0 - omega_m
    k_arr, pk_z0, pk_zinit, s8_init = compute_camb_pk(omega_m, omega_b, h, ns, sigma8, z_init)
    pk_interp = interp1d(k_arr, pk_zinit, kind='cubic', fill_value='extrapolate')
    delta_k, KX, KY, KZ, K2 = generate_gaussian_field(N, BOX, pk_interp)
    psi_x, psi_y, psi_z = compute_za_displacement(delta_k, KX, KY, KZ, K2, N, BOX)
    qx, qy, qz = make_regular_grid(N, BOX)
    pos_x = (qx + psi_x.ravel()) % BOX
    pos_y = (qy + psi_y.ravel()) % BOX
    pos_z = (qz + psi_z.ravel()) % BOX
    a_init = 1.0 / (1.0 + z_init)
    H_init = H_func(a_init, omega_m, omega_l)
    f_init = growth_rate_f(a_init, omega_m, omega_l)
    vel_factor = a_init * H_init * f_init
    vel_x = psi_x.ravel() * vel_factor
    vel_y = psi_y.ravel() * vel_factor
    vel_z = psi_z.ravel() * vel_factor
    data_dir = "data/"
    np.save(os.path.join(data_dir, "pos.npy"), np.stack([pos_x, pos_y, pos_z], axis=1).astype(np.float32))
    np.save(os.path.join(data_dir, "vel.npy"), np.stack([vel_x, vel_y, vel_z], axis=1).astype(np.float32))
    print("Saved ICs to data/")