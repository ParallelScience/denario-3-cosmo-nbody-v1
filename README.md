# denario-3-cosmo-nbody-v1

**Scientist:** denario-3
**Date:** 2026-05-11

# GPU-Accelerated Cosmological N-body Simulation with NVIDIA Warp

## Overview

This project implements a cosmological N-body (PM — Particle-Mesh) simulation using NVIDIA Warp on GPU, then validates it against reference Quijote power spectra. The goal is to quantify both the physical accuracy (P(k) comparison) and the computational speedup vs. equivalent CPU code.

## Simulation Parameters (matching Quijote fiducial)

| Parameter | Value |
|---|---|
| Box size | 1000 Mpc/h |
| N_particles | 512³ = 134,217,728 |
| Mesh resolution | 512³ (or 1024³) |
| Cosmology | Planck 2018 fiducial |
| Ωm | 0.3175 |
| Ωb | 0.049 |
| h | 0.6711 |
| ns | 0.9624 |
| σ8 | 0.834 |
| Starting redshift | z_init = 127 |
| Output redshifts | z = 0, 0.5, 1, 2 |
| Random seed | 0 (fixed, reproducible) |

## Initial Conditions

ICs are generated on-the-fly using the Zel'dovich Approximation (ZA) / first-order Lagrangian Perturbation Theory:
1. Compute linear matter power spectrum P(k, z_init) using `camb` with the Quijote fiducial cosmology
2. Generate a Gaussian random field in Fourier space using P(k) and a fixed random seed
3. Displace particles from a regular grid using the ZA displacement field
4. Assign velocities using the ZA velocity field (∇φ / (aH f))

Note: Quijote uses 2LPT (2nd-order LPT) for ICs. Our ZA ICs introduce small differences at high-k that are expected and will be characterized.

## Simulation Code: NVIDIA Warp (GPU)

- **Framework**: NVIDIA Warp 1.13.0 (`warp-lang`)
- **Device**: NVIDIA RTX PRO 6000 Blackwell (95 GiB VRAM, sm_120, CUDA 13.0)
- **Method**: Particle-Mesh (PM) — O(N log N) complexity
  - CIC (Cloud-in-Cell) mass assignment to 512³ mesh
  - FFT-based Poisson solver: ∇²φ = -4πGρ → φ(k) = -4πGρ(k)/k²
  - Gradient of potential → forces on particles
  - Leapfrog integration (kick-drift-kick scheme)
- **Time stepping**: ~50-100 steps from z=127 to z=0, step size controlled by Courant condition

## Reference Data: Quijote Power Spectra

The Quijote fiducial P(k) at z=0 is publicly available at:
`https://raw.githubusercontent.com/franciscovillaescusa/Quijote-simulations/master/summary_statistics/Pk/Pk_m_z=0_0.txt`

This provides the non-linear matter power spectrum P(k) averaged over multiple Quijote realizations. The comparison metric is:
- Ratio P_warp(k) / P_quijote(k) across k range 0.01–1 h/Mpc
- Target: agreement within 5% for k < 0.3 h/Mpc

## Speedup Benchmark

Compare wall-clock time for:
- Full PM simulation (512³): Warp GPU vs. equivalent numpy/scipy CPU implementation
- Per-step breakdown: mass assignment, FFT, force interpolation, integration
- Memory usage: GPU vs. CPU

## File Paths

All generated files are saved to the project data directory:
- `/home/node/work/projects/cosmo_nbody_v1/data/` — IC files, snapshot outputs, P(k) outputs
- No pre-existing input files; everything is generated in-code

## Software Dependencies (all available in /opt/denario-venv)

- `warp-lang==1.13.0` — GPU-accelerated kernels
- `camb==1.6.6` — linear matter power spectrum
- `numpy`, `scipy` — FFT, array operations
- `matplotlib` — plotting
