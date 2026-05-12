# filename: codebase/step_1.py
import sys
import os
sys.path.insert(0, os.path.abspath("codebase"))
sys.path.insert(0, "/home/node/data/compsep_data/")
import numpy as np
import camb
import os

def compute_camb_pk(omega_m, omega_b, h, ns, sigma8, z_init, z_out=0.0):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100.0 * h, ombh2=omega_b * h**2, omch2=(omega_m - omega_b) * h**2, omk=0.0, mnu=0.0, neutrino_hierarchy='degenerate', num_massive_neutrinos=0)
    pars.InitPower.set_params(ns=ns, As=2.0e-9)
    pars.set_matter_power(redshifts=[z_out, z_init], kmax=100.0)
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    sigma8_camb_raw = results.get_sigma8_0()
    rescale = (sigma8 / sigma8_camb_raw) ** 2
    pars.InitPower.set_params(ns=ns, As=2.0e-9 * rescale)
    results = camb.get_results(pars)
    sigma8_final = results.get_sigma8_0()
    k_arr, z_arr, pk_arr = results.get_matter_power_spectrum(minkh=1e-4, maxkh=100.0, npoints=2048)
    s8_all = results.get_sigma8()
    return k_arr, pk_arr[0], pk_arr[1], sigma8_final, s8_all

def build_k_grids(N, L):
    dk = 2.0 * np.pi / L
    kx = np.fft.fftfreq(N, d=1.0 / N) * dk
    ky = np.fft.fftfreq(N, d=1.0 / N) * dk
    kz = np.fft.rfftfreq(N, d=1.0 / N) * dk
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K2 = KX**2 + KY**2 + KZ**2
    K = np.sqrt(K2)
    return KX, KY, KZ, K2, K

def interpolate_pk(k_camb, pk_camb, K):
    log_k = np.log(k_camb)
    log_pk = np.log(pk_camb)
    K_flat = K.ravel()
    log_K_flat = np.where(K_flat > 0, np.log(np.maximum(K_flat, k_camb[0])), np.log(k_camb[0]))
    log_pk_interp = np.interp(log_K_flat, log_k, log_pk, left=log_pk[0], right=log_pk[-1])
    pk_grid = np.exp(log_pk_interp).reshape(K.shape)
    pk_grid[K == 0] = 0.0
    return pk_grid

def generate_gaussian_field(N, L, pk_grid, seed):
    V = L**3
    rng = np.random.default_rng(seed)
    amplitude = (N**3 / np.sqrt(2.0 * V)) * np.sqrt(pk_grid)
    real_part = rng.standard_normal(pk_grid.shape)
    imag_part = rng.standard_normal(pk_grid.shape)
    delta_k = amplitude * (real_part + 1j * imag_part)
    delta_k[0, 0, 0] = 0.0
    return delta_k

def compute_psi1(delta_k, KX, KY, KZ, K2, N):
    K2_safe = np.where(K2 > 0, K2, 1.0)
    phi1_k = -delta_k / K2_safe
    phi1_k[K2 == 0] = 0.0
    psi1x_k = -1j * KX * phi1_k
    psi1y_k = -1j * KY * phi1_k
    psi1z_k = -1j * KZ * phi1_k
    psi1_x = np.fft.irfftn(psi1x_k, s=(N, N, N))
    psi1_y = np.fft.irfftn(psi1y_k, s=(N, N, N))
    psi1_z = np.fft.irfftn(psi1z_k, s=(N, N, N))
    return psi1_x, psi1_y, psi1_z, phi1_k

def compute_psi2(phi1_k, KX, KY, KZ, K2, N):
    K2_safe = np.where(K2 > 0, K2, 1.0)
    phi1_xx_k = -KX**2 * phi1_k
    phi1_yy_k = -KY**2 * phi1_k
    phi1_zz_k = -KZ**2 * phi1_k
    phi1_xy_k = -KX * KY * phi1_k
    phi1_xz_k = -KX * KZ * phi1_k
    phi1_yz_k = -KY * KZ * phi1_k
    phi1_xx = np.fft.irfftn(phi1_xx_k, s=(N, N, N))
    phi1_yy = np.fft.irfftn(phi1_yy_k, s=(N, N, N))
    phi1_zz = np.fft.irfftn(phi1_zz_k, s=(N, N, N))
    phi1_xy = np.fft.irfftn(phi1_xy_k, s=(N, N, N))
    phi1_xz = np.fft.irfftn(phi1_xz_k, s=(N, N, N))
    phi1_yz = np.fft.irfftn(phi1_yz_k, s=(N, N, N))
    source2 = (phi1_xx * phi1_yy - phi1_xy**2 + phi1_xx * phi1_zz - phi1_xz**2 + phi1_yy * phi1_zz - phi1_yz**2)
    source2_k = np.fft.rfftn(source2)
    phi2_k = -source2_k / K2_safe
    phi2_k[K2 == 0] = 0.0
    psi2x_k = -1j * KX * phi2_k
    psi2y_k = -1j * KY * phi2_k
    psi2z_k = -1j * KZ * phi2_k
    psi2_x = np.fft.irfftn(psi2x_k, s=(N, N, N))
    psi2_y = np.fft.irfftn(psi2y_k, s=(N, N, N))
    psi2_z = np.fft.irfftn(psi2z_k, s=(N, N, N))
    return psi2_x, psi2_y, psi2_z

if __name__ == '__main__':
    N, L, z_init, seed = 512, 1000.0, 127.0, 0
    k_camb, pk_z0, pk_zinit, s8, s8_all = compute_camb_pk(0.3175, 0.049, 0.6711, 0.9624, 0.834, z_init)
    KX, KY, KZ, K2, K = build_k_grids(N, L)
    pk_grid = interpolate_pk(k_camb, pk_zinit, K)
    delta_k = generate_gaussian_field(N, L, pk_grid, seed)
    psi1_x, psi1_y, psi1_z, phi1_k = compute_psi1(delta_k, KX, KY, KZ, K2, N)
    psi2_x, psi2_y, psi2_z = compute_psi2(phi1_k, KX, KY, KZ, K2, N)
    a_init = 1.0 / (1.0 + z_init)
    D1 = 1.0
    D2 = -3.0 / 7.0 * D1**2
    f1 = 1.0
    f2 = 2.0 * f1
    H = 100.0 * np.sqrt(0.3175 * (1.0 + z_init)**3 + (1.0 - 0.3175))
    x = np.linspace(0, L, N, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    pos_x = (X + D1 * psi1_x + D2 * psi2_x) % L
    pos_y = (Y + D1 * psi1_y + D2 * psi2_y) % L
    pos_z = (Z + D1 * psi1_z + D2 * psi2_z) % L
    vel_x = a_init * H * f1 * psi1_x + 2.0 * a_init * H * f2 * psi2_x
    vel_y = a_init * H * f1 * psi1_y + 2.0 * a_init * H * f2 * psi2_y
    vel_z = a_init * H * f1 * psi1_z + 2.0 * a_init * H * f2 * psi2_z
    rms_disp = np.sqrt(np.mean(D1**2 * (psi1_x**2 + psi1_y**2 + psi1_z**2)))
    print('Sigma8 at z=0:', s8)
    print('RMS displacement:', rms_disp, 'Mpc/h')
    print('Inter-particle spacing:', L / N, 'Mpc/h')
    print('Particle count:', N**3)
    np.save('data/pos.npy', np.stack([pos_x, pos_y, pos_z], axis=-1))
    np.save('data/vel.npy', np.stack([vel_x, vel_y, vel_z], axis=-1))