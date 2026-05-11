# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import camb
import os
import time

def compute_camb_pk(z_init, omega_m, omega_b, h, ns, sigma8, box_size, n_mesh):
    omega_cdm = omega_m - omega_b
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100.0 * h, ombh2=omega_b * h**2, omch2=omega_cdm * h**2, omk=0.0, mnu=0.0, neutrino_hierarchy='degenerate')
    pars.InitPower.set_params(ns=ns, As=2.0e-9)
    pars.set_matter_power(redshifts=[z_init, 0.0], kmax=20.0)
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    sigma8_camb = results.get_sigma8_0()
    As_rescaled = 2.0e-9 * (sigma8 / sigma8_camb)**2
    pars.InitPower.set_params(ns=ns, As=As_rescaled)
    pars.set_matter_power(redshifts=[z_init, 0.0], kmax=20.0)
    results = camb.get_results(pars)
    k_arr, _, pk_2d = results.get_matter_power_spectrum(minkh=1e-4, maxkh=20.0, npoints=2048)
    return k_arr, pk_2d[0]

def compute_hubble(z, omega_m, h):
    H0 = 100.0 * h
    omega_lambda = 1.0 - omega_m
    E_z = np.sqrt(omega_m * (1.0 + z)**3 + omega_lambda)
    return H0 * E_z

def compute_growth_rate(z, omega_m):
    omega_lambda = 1.0 - omega_m
    E2_z = omega_m * (1.0 + z)**3 + omega_lambda
    omega_m_z = omega_m * (1.0 + z)**3 / E2_z
    return omega_m_z**0.55

def generate_gaussian_field(n_mesh, k_arr, pk_arr, box_size, seed=0):
    rng = np.random.default_rng(seed)
    dk = 2.0 * np.pi / box_size
    kfreq_1d = np.fft.fftfreq(n_mesh, d=1.0 / n_mesh) * dk
    kfreq_1d_r = np.fft.rfftfreq(n_mesh, d=1.0 / n_mesh) * dk
    kx_grid, ky_grid, kz_grid = np.meshgrid(kfreq_1d.astype(np.float32), kfreq_1d.astype(np.float32), kfreq_1d_r.astype(np.float32), indexing='ij')
    k_grid_mag = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2).astype(np.float32)
    pk_interp = np.interp(k_grid_mag.ravel(), k_arr, pk_arr, left=0.0, right=0.0).reshape(k_grid_mag.shape).astype(np.float32)
    amplitude = np.sqrt(pk_interp / (2.0 * box_size**3)).astype(np.float32)
    noise_real = rng.standard_normal(k_grid_mag.shape).astype(np.float32)
    noise_imag = rng.standard_normal(k_grid_mag.shape).astype(np.float32)
    delta_k = (amplitude * noise_real + 1j * amplitude * noise_imag).astype(np.complex64)
    delta_k[k_grid_mag == 0] = 0.0
    return delta_k, k_grid_mag, kx_grid, ky_grid, kz_grid

def compute_za_displacement(delta_k, kx_grid, ky_grid, kz_grid, k_grid_mag, n_mesh):
    k2 = k_grid_mag**2
    k2_safe = np.where(k2 == 0, 1.0, k2)
    psi_kx = (-1j * kx_grid / k2_safe * delta_k).astype(np.complex64)
    psi_ky = (-1j * ky_grid / k2_safe * delta_k).astype(np.complex64)
    psi_kz = (-1j * kz_grid / k2_safe * delta_k).astype(np.complex64)
    psi_kx[k_grid_mag == 0] = 0.0
    psi_ky[k_grid_mag == 0] = 0.0
    psi_kz[k_grid_mag == 0] = 0.0
    psi_x = np.fft.irfftn(psi_kx, s=(n_mesh, n_mesh, n_mesh)).astype(np.float32)
    psi_y = np.fft.irfftn(psi_ky, s=(n_mesh, n_mesh, n_mesh)).astype(np.float32)
    psi_z = np.fft.irfftn(psi_kz, s=(n_mesh, n_mesh, n_mesh)).astype(np.float32)
    return psi_x, psi_y, psi_z

def generate_particle_grid(n_mesh, box_size):
    cell_size = box_size / n_mesh
    coords_1d = (np.arange(n_mesh, dtype=np.float32) + 0.5) * cell_size
    qx, qy, qz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing='ij')
    return qx.ravel(), qy.ravel(), qz.ravel()

if __name__ == '__main__':
    data_dir = "data/"
    OMEGA_M, OMEGA_B, H_PARAM, NS, SIGMA8, BOX_SIZE, N_MESH, Z_INIT, SEED = 0.3175, 0.049, 0.6711, 0.9624, 0.834, 1000.0, 512, 127.0, 0
    k_arr, pk_arr = compute_camb_pk(Z_INIT, OMEGA_M, OMEGA_B, H_PARAM, NS, SIGMA8, BOX_SIZE, N_MESH)
    a_init = 1.0 / (1.0 + Z_INIT)
    H_init = compute_hubble(Z_INIT, OMEGA_M, H_PARAM)
    f_init = compute_growth_rate(Z_INIT, OMEGA_M)
    delta_k, k_grid_mag, kx_grid, ky_grid, kz_grid = generate_gaussian_field(N_MESH, k_arr, pk_arr, BOX_SIZE, seed=SEED)
    psi_x, psi_y, psi_z = compute_za_displacement(delta_k, kx_grid, ky_grid, kz_grid, k_grid_mag, N_MESH)
    qx, qy, qz = generate_particle_grid(N_MESH, BOX_SIZE)
    pos_x = (qx + psi_x.ravel()) % BOX_SIZE
    pos_y = (qy + psi_y.ravel()) % BOX_SIZE
    pos_z = (qz + psi_z.ravel()) % BOX_SIZE
    vel_factor = a_init * H_init * f_init
    vel_x = psi_x.ravel() * vel_factor
    vel_y = psi_y.ravel() * vel_factor
    vel_z = psi_z.ravel() * vel_factor
    np.save(os.path.join(data_dir, "pos_init.npy"), np.stack([pos_x, pos_y, pos_z], axis=1))
    np.save(os.path.join(data_dir, "vel_init.npy"), np.stack([vel_x, vel_y, vel_z], axis=1))
    np.save(os.path.join(data_dir, "cosmo_params.npy"), np.array([a_init, H_init, f_init]))
    print("Initial conditions saved to data/")