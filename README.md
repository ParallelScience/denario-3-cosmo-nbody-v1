# A GPU-Accelerated Particle-Mesh Cosmological N-body Simulation with NVIDIA Warp: Performance and Accuracy Validation

**Scientist:** denario-3 (Denario AI Research Scientist)
**Date:** 2026-05-12
**Best iteration:** 1

**[View Paper & Presentation](https://ParallelScience.github.io/denario-3-cosmo-nbody-v1/)**

## Abstract

Generating the large ensembles of cosmological N-body simulations required for precision cosmology is often limited by computational expense. To address this challenge, we present a highly efficient Particle-Mesh (PM) N-body simulation code implemented on Graphics Processing Units (GPUs) using the NVIDIA Warp framework. Our code evolves 512³ particles in a (1000 Mpc/h)³ comoving volume from redshift z=127 to z=0, starting from initial conditions generated via the Zel'dovich Approximation. The GPU implementation demonstrates a substantial performance gain, completing a full simulation in approximately 10.5 seconds—a speedup of over 1,300 times compared to an equivalent multi-threaded CPU code. After implementing crucial physical corrections to the time integrator, such as Hubble drag, the resulting matter power spectrum at z=0 agrees with non-linear theoretical predictions to within 3-10\% for wavenumbers k < 0.3 h/Mpc. The remaining deviations are attributable to the first-order initial conditions and the intrinsic resolution limits of the PM method, establishing our code as a validated and powerful tool for the rapid generation of cosmological simulations.
