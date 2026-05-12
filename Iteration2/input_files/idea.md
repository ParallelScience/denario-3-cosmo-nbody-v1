**Title: GPU-Accelerated Cosmological N-body Simulation with NVIDIA Warp: Accuracy and Speedup vs. Quijote**

This project implements a cosmological Particle-Mesh (PM) N-body simulation using NVIDIA Warp on GPU, benchmarks its performance against an equivalent CPU implementation, and validates its physical accuracy against the Quijote reference power spectra.

**Research Questions:**
1. How accurately does the Warp PM simulation reproduce the matter power spectrum P(k) of the Quijote fiducial cosmology?
2. What is the GPU speedup over an equivalent CPU (numpy/scipy) PM implementation?

**Approach:**

*Initial Conditions*: Generate ICs at z=127 using the Zel'dovich Approximation (ZA) with the Quijote fiducial cosmology (Ωm=0.3175, h=0.6711, ns=0.9624, σ8=0.834, box=1000 Mpc/h). The linear P(k) is computed with `camb`. Note: Quijote uses 2LPT; the ZA difference will be characterized but is not the focus.

*GPU Simulation*: Implement PM gravity in NVIDIA Warp (1.13.0) on RTX PRO 6000 (95 GiB VRAM):
- Cloud-in-Cell (CIC) mass assignment to 512³ mesh
- FFT-based Poisson solver
- Leapfrog (kick-drift-kick) integration from z=127 to z=0
- ~50 time steps with adaptive step size

*CPU Baseline*: Equivalent numpy/scipy PM implementation for timing comparison.

*Validation*: Compare the final P(k) at z=0 against the publicly available Quijote fiducial power spectrum. Target: P_warp(k)/P_quijote(k) within 5% for k < 0.3 h/Mpc.

*Deliverables*:
1. Wall-clock time: GPU vs CPU (per step and total)
2. P(k) ratio plot: Warp vs Quijote reference
3. Qualitative assessment of where PM + ZA deviates from the full Quijote result and why
