# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import camb
import os
import time
import urllib.request
from scipy.interpolate import interp1d

def setup_camb_cosmology():
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.11, ombh2=0.049 * 0.6711**2, omch2=(0.3175 - 0.049) * 0.6711**2, mnu=0.0, omk=0.0, tau=0.0952)
    pars.InitPower.set_params(ns=0.9624, As=2.0e-9)
    pars.set_matter_power(redshifts=[0.0, 0.5, 1.0, 2.0, 127.0], kmax=100.0)
    pars.NonLinear = camb.model.NonLinear_none
    return pars

def compute_sigma8_rescaling(pars, target_sigma8=0.834):
    results = camb.get_results(pars)
    sigma8_current = results.get_sigma8_0()
    rescale = (target_sigma8 / sigma8_current) ** 2
    return rescale, sigma8_current

def get_linear_pk_camb(target_sigma8=0.834, z_list=None, kmax=100.0):
    if z_list is None:
        z_list = [0.0, 0.5, 1.0, 2.0, 127.0]
    pars = setup_camb_cosmology()
    rescale, sigma8_init = compute_sigma8_rescaling(pars, target_sigma8)
    pars.InitPower.As *= rescale
    z_sorted = sorted(z_list, reverse=True)
    pars.set_matter_power(redshifts=z_sorted, kmax=kmax)
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    sigma8_final = results.get_sigma8_0()
    kh, z_out, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=kmax, npoints=2000)
    pk_dict = {}
    for iz, zval in enumerate(z_out):
        zkey = float(round(zval, 4))
        pk_dict[zkey] = pk[iz, :]
    growth_factor_z127 = np.sqrt(pk_dict[127.0][100] / pk_dict[0.0][100])
    return kh, pk_dict, results, sigma8_final, growth_factor_z127

def get_halofit_pk(target_sigma8=0.834, z_list=None, kmax=100.0):
    if z_list is None:
        z_list = [0.0, 0.5, 1.0, 2.0]
    pars = setup_camb_cosmology()
    rescale, _ = compute_sigma8_rescaling(pars, target_sigma8)
    pars.InitPower.As *= rescale
    z_sorted = sorted(z_list, reverse=True)
    pars.set_matter_power(redshifts=z_sorted, kmax=kmax)
    pars.NonLinear = camb.model.NonLinear_both
    results = camb.get_results(pars)
    k_nl, z_out, pk_nl = results.get_matter_power_spectrum(minkh=1e-4, maxkh=kmax, npoints=2000)
    pk_nl_dict = {}
    for iz, zval in enumerate(z_out):
        zkey = float(round(zval, 4))
        pk_nl_dict[zkey] = pk_nl[iz, :]
    return k_nl, pk_nl_dict

def generate_za_ics(N, L, pk_interp_func, z_init, a_init, H_init, seed=0):
    rng = np.random.default_rng(seed)
    dx = L / N
    cell_idx = np.arange(N, dtype=np.float32)
    gx, gy, gz = np.meshgrid(cell_idx, cell_idx, cell_idx, indexing='ij')
    q = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1) * dx
    kf = 2.0 * np.pi / L
    kx = np.fft.fftfreq(N, d=1.0 / N) * kf
    ky = np.fft.fftfreq(N, d=1.0 / N) * kf
    kz = np.fft.rfftfreq(N, d=1.0 / N) * kf
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K2[0, 0, 0] = 1.0
    Kmag = np.sqrt(K2)
    Kmag[0, 0, 0] = 1.0
    Pk_grid = np.zeros_like(Kmag)
    mask = Kmag > 0
    Pk_grid[mask] = pk_interp_func(Kmag[mask])
    V = L**3
    amplitude = (N**3 / np.sqrt(2.0 * V)) * np.sqrt(Pk_grid)
    n_rfft = N * N * (N // 2 + 1)
    noise_real = rng.standard_normal(n_rfft).reshape(N, N, N // 2 + 1)
    noise_imag = rng.standard_normal(n_rfft).reshape(N, N, N // 2 + 1)
    delta_k = amplitude * (noise_real + 1j * noise_imag)
    delta_k[0, 0, 0] = 0.0
    f_growth = 1.0
    vel_prefactor = a_init * H_init * f_growth
    pos = np.empty((N**3, 3), dtype=np.float32)
    vel = np.empty((N**3, 3), dtype=np.float32)
    for dim in range(3):
        if dim == 0: Kdim = KX
        elif dim == 1: Kdim = KY
        else: Kdim = KZ
        psi_k = -1j * Kdim / K2 * delta_k
        psi_k[0, 0, 0] = 0.0
        psi_real = np.fft.irfftn(psi_k, s=(N, N, N))
        pos[:, dim] = (q[:, dim] + psi_real.ravel()).astype(np.float32)
        vel[:, dim] = (vel_prefactor * psi_real.ravel()).astype(np.float32)
    pos = np.mod(pos, L)
    psi_x = np.fft.irfftn(-1j * KX / K2 * delta_k, s=(N, N, N))
    psi_y = np.fft.irfftn(-1j * KY / K2 * delta_k, s=(N, N, N))
    psi_z = np.fft.irfftn(-1j * KZ / K2 * delta_k, s=(N, N, N))
    psi_rms = float(np.sqrt(np.mean(psi_x**2 + psi_y**2 + psi_z**2)))
    return pos, vel, psi_rms

def compute_hubble(z, Om=0.3175, h=0.6711):
    H0 = 100.0 * h
    Ez = np.sqrt(Om * (1.0 + z)**3 + (1.0 - Om))
    return H0 * Ez

if __name__ == '__main__':
    N, L = 512, 1000.0
    z_init = 127.0
    a_init = 1.0 / (1.0 + z_init)
    H_init = compute_hubble(z_init)
    kh, pk_dict, _, _, growth_factor = get_linear_pk_camb()
    pk_interp = interp1d(kh, pk_dict[127.0], kind='linear', fill_value='extrapolate')
    pos, vel, psi_rms = generate_za_ics(N, L, pk_interp, z_init, a_init, H_init)
    k_nl, pk_nl_dict = get_halofit_pk()
    data_dir = 'data/'
    np.savez(os.path.join(data_dir, 'ics.npz'), pos=pos, vel=vel, psi_rms=psi_rms)
    np.savez(os.path.join(data_dir, 'pk_ref.npz'), kh=kh, pk_lin=pk_dict, k_nl=k_nl, pk_nl=pk_nl_dict)
    print('Simulation parameters: N=' + str(N) + ', L=' + str(L) + ', z_init=' + str(z_init))
    print('RMS displacement: ' + str(psi_rms) + ' Mpc/h')
    print('Growth factor at z=127: ' + str(growth_factor))