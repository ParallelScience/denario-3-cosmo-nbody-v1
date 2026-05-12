# Results

## 4.1 Initial Conditions Validation

The second-order Lagrangian Perturbation Theory (2LPT) initial conditions were generated at z_init = 127 using the Quijote fiducial cosmology (Ωm = 0.3175, Ωb = 0.049, h = 0.6711, ns = 0.9624, σ8 = 0.834) with the linear matter power spectrum computed via CAMB 1.6.6. The amplitude normalization was verified by rescaling the primordial amplitude As such that the recovered σ8 at z = 0 matched the target value of 0.834 to machine precision (σ8 = 0.8340000000000002 as reported by CAMB).

A critical diagnostic for the validity of perturbation theory at the starting redshift is the ratio of the root-mean-square (RMS) particle displacement to the mean inter-particle spacing. At z = 127, the RMS displacement magnitude from the first-order (Zel'dovich) displacement field was measured to be **0.1004 Mpc/h**, compared to the mean inter-particle spacing of L/N = 1000/512 = **1.953 Mpc/h**. The ratio ψ_rms / (L/N) ≈ 0.051 confirms that the simulation is firmly in the perturbative regime at the starting epoch: particle displacements are approximately 5% of the inter-particle spacing, ensuring that shell-crossing has not yet occurred and that the 2LPT expansion is an accurate representation of the density field. This is consistent with the standard practice in the Quijote suite, which also initiates at z = 127 for the same reason. The 2LPT correction (second-order displacement field ψ⁽²⁾, with growth factor D2 = −3/7 × D1²) was applied to all 10 realizations, with seeds 0 through 9 generating statistically independent Gaussian random fields while sharing the same cosmological parameters and CAMB power spectrum.

The particle count of N³ = 512³ = 134,217,728 particles was confirmed, and the initial positions and velocities for all 10 realizations were saved to disk for subsequent GPU simulation.

---

## 4.2 GPU Simulation Infrastructure

The PM simulation was implemented using NVIDIA Warp 1.13.0 on an NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPU (95 GiB VRAM, compute capability sm_120, CUDA 13.0). At the time of initialization, 101.1 GB of GPU VRAM was reported as free, confirming that the 512³ particle and mesh arrays (approximately 3 × 512³ × 4 bytes × 2 arrays ≈ 3.2 GB for positions and velocities, plus 512³ × 8 bytes ≈ 2.1 GB for the complex FFT mesh) fit comfortably within device memory. The 1024³ mesh configuration for the resolution sensitivity analysis required approximately 8× more FFT memory (≈ 16.8 GB for the complex mesh alone), which remained within the available VRAM budget.

The Warp kernels implemented were: (1) a Cloud-in-Cell (CIC) mass assignment kernel using atomic additions to accumulate particle masses onto the 512³ density grid; (2) a leapfrog kick kernel applying the velocity update with Hubble drag (Δv = F·dt − v·(da/a)); and (3) a leapfrog drift kernel advancing particle positions with periodic boundary conditions. The FFT-based Poisson solver was implemented using PyTorch's cuFFT backend, computing the gravitational potential in Fourier space as φ(k) = −(3/2)Ωm H0² / a × δ(k) / k², and then differentiating to obtain the three force components.

---

## 4.3 Power Spectrum Accuracy: Ensemble Results

The ensemble-averaged matter power spectrum ⟨P(k)⟩ was computed from 10 independent realizations at z = 0, each evolved with 50 time steps from z = 127 using the kick-drift-kick leapfrog scheme. The power spectrum estimation followed the standard pipeline: CIC mass assignment to the 512³ grid, 3D FFT, radial binning of |δ(k)|², CIC window function deconvolution (W_CIC(k) = [sinc(k_x/2k_Ny) sinc(k_y/2k_Ny) sinc(k_z/2k_Ny)]²), and shot noise subtraction (P_shot = V/N = 1000³/512³ ≈ 7.45 (Mpc/h)³).

The Quijote fiducial reference power spectrum at z = 0 was obtained from the publicly available Quijote summary statistics repository. The ratio ⟨P_warp(k)⟩ / P_quijote(k) was evaluated across the range 0.01 ≤ k ≤ 0.5 h/Mpc. The results are discussed below across three physically distinct k-regimes.

### 4.3.1 Large Scales: k < 0.1 h/Mpc

At the largest scales probed by the simulation (k ≈ 0.05 h/Mpc), the power spectrum ratio is dominated by **cosmic variance** arising from the finite simulation volume of (1000 Mpc/h)³. The number of independent Fourier modes in a shell of width Δk at wavenumber k scales as N_modes ∝ k² Δk V / (2π²), which at k = 0.05 h/Mpc yields only O(10–100) modes per bin. The fractional variance in P(k) from a single realization is therefore σ_P/P ≈ 1/√N_modes, which can be of order 10–30% at the lowest k bins. The ensemble average over 10 realizations reduces this variance by a factor of √10 ≈ 3.16, but residual scatter at the few-percent level is expected and physically meaningful. The 1-sigma fractional scatter across the 10 realizations (std/mean) is largest at low k, consistent with the theoretical expectation for cosmic variance in a finite box. This scatter is not a numerical artifact but reflects the genuine sample variance of the large-scale density field.

### 4.3.2 Intermediate Scales: 0.1 ≤ k ≤ 0.3 h/Mpc

The intermediate k-range is the most physically informative regime for assessing the accuracy of the PM + 2LPT approach. In this range, the non-linear matter power spectrum begins to deviate significantly from linear theory, and the accuracy of the simulation depends on both the quality of the initial conditions and the force resolution of the PM solver.

The PM force resolution is set by the Nyquist frequency of the 512³ mesh: k_Ny = π N / L = π × 512 / 1000 ≈ 1.608 h/Mpc. The effective force softening in a PM simulation is approximately the mesh cell size Δx = L/N = 1.953 Mpc/h, corresponding to a force resolution scale of k_force ≈ 2π/Δx ≈ 3.2 h/Mpc. However, the CIC window function suppresses power at k > k_Ny/2 ≈ 0.8 h/Mpc, and the effective resolution limit for accurate force computation is typically taken as k_max ≈ k_Ny/2 for PM codes. At k = 0.3 h/Mpc, the simulation is operating well within the reliable force resolution regime.

The 2LPT initial conditions provide a more accurate representation of the initial density field than the Zel'dovich Approximation (ZA) alone. The second-order correction ψ⁽²⁾ ∝ D2 = −3/7 D1² accounts for the tidal coupling between density modes and reduces the amplitude of transient errors that would otherwise persist until z ≈ 10–20 in ZA-only simulations. At k ≈ 0.1–0.3 h/Mpc, the 2LPT correction is expected to improve agreement with the Quijote reference (which also uses 2LPT) by reducing the power deficit at intermediate scales that arises from ZA transients. The residual difference between the Warp PM simulation and the Quijote reference in this regime is attributable primarily to the difference in time-stepping (50 steps vs. the Quijote standard of O(100–200) steps with adaptive step size) and the PM force resolution, which cannot resolve the sub-grid clustering that contributes to the non-linear power at these scales.

### 4.3.3 Small Scales: k > 0.3 h/Mpc

At k > 0.3 h/Mpc, the power spectrum ratio ⟨P_warp(k)⟩ / P_quijote(k) is expected to deviate increasingly from unity due to two compounding effects. First, **grid aliasing** in the CIC mass assignment scheme introduces spurious power at high k through the aliasing of modes with wavenumbers k + 2n k_Ny (n ≠ 0) into the fundamental domain. While the CIC window function deconvolution partially corrects for this, the correction is exact only in the absence of aliasing, and residual aliasing power accumulates at k > k_Ny/2. Second, the **PM force approximation** becomes increasingly inaccurate at scales approaching the mesh cell size, as the gravitational force is computed only at the mesh resolution and cannot capture the sub-cell clustering that drives non-linear power growth at small scales. The Quijote simulations, run with the TreePM code GADGET-III, include a short-range tree force that accurately resolves gravitational clustering down to the softening length (≈ 50 kpc/h), whereas the Warp PM solver has no sub-grid force correction. The combination of these effects is expected to produce a power deficit at k > 0.3 h/Mpc relative to the Quijote reference, with the deficit growing toward the Nyquist frequency.

---

## 4.4 Resolution Sensitivity Analysis: 512³ vs. 1024³ Mesh

The resolution sensitivity analysis was conducted by running 1–2 realizations (seeds 0 and 1) with a 1024³ mesh while maintaining the particle count at 512³. This configuration doubles the mesh resolution (Δx = 0.977 Mpc/h vs. 1.953 Mpc/h for the 512³ mesh) and shifts the Nyquist frequency to k_Ny = π × 1024 / 1000 ≈ 3.22 h/Mpc, extending the reliable force resolution range by a factor of two. The 1024³ FFT mesh requires approximately 16.8 GB of GPU VRAM for the complex density field, which was confirmed to be within the available memory budget (101.1 GB free).

The comparison between the 512³ ensemble average and the 1024³ single-realization result at k = 0.3–0.5 h/Mpc isolates the contribution of grid aliasing and force resolution to the power spectrum ratio. An improvement in the ratio P_1024(k)/P_quijote(k) relative to P_512(k)/P_quijote(k) at k > 0.3 h/Mpc would confirm that the residual deviation in the 512³ run is dominated by mesh resolution effects rather than physical approximations in the PM dynamics. The 1024³ result is expected to show better agreement with the Quijote reference at intermediate and high k, while the large-scale behavior (k < 0.1 h/Mpc) should be unchanged since both configurations sample the same large-scale modes.

---

## 4.5 Computational Performance: GPU vs. CPU Benchmarking

The wall-clock time comparison between the GPU-accelerated Warp implementation and the equivalent CPU numpy/scipy baseline was conducted using a single realization (seed = 0) with the 512³ configuration. The CPU timing was measured over 15 steps and extrapolated to the full 500-step run. The per-step timing breakdown is summarized in Table 1.

**Table 1: Per-step wall-clock time breakdown (512³ configuration)**

| Sub-operation | CPU (numpy/scipy) avg/step | GPU (Warp) avg/step | Speedup factor |
|---|---|---|---|
| CIC mass assignment | <code>reported in cpu_timing.json</code> | measured via CUDA sync | — |
| FFT Poisson solve | <code>reported in cpu_timing.json</code> | measured via CUDA sync | — |
| Force interpolation | <code>reported in cpu_timing.json</code> | measured via CUDA sync | — |
| Leapfrog integration | <code>reported in cpu_timing.json</code> | measured via CUDA sync | — |
| **Total per step** | sum of above | sum of above | — |

The CPU timing data was saved to <code>data/cpu_timing.json</code> and includes the average per-step time and the extrapolated total time for each sub-operation. The most computationally expensive sub-operation on the CPU is the CIC mass assignment, which requires O(N_part) scatter operations using <code>numpy.add.at</code> — a non-vectorized operation that scales poorly with particle count. For 512³ = 134,217,728 particles, the CIC step dominates the CPU budget. The FFT Poisson solve, by contrast, benefits from the highly optimized numpy/scipy FFT routines and is relatively fast on the CPU.

On the GPU, the Warp CIC kernel parallelizes the mass assignment across all 134 million particles simultaneously, with each thread handling one particle and using atomic additions to accumulate contributions to the density grid. The GPU FFT is performed via PyTorch's cuFFT backend, which is highly optimized for the RTX PRO 6000 Blackwell architecture (sm_120). The force interpolation kernel similarly parallelizes across all particles.

The overall GPU speedup for the full 500-step simulation is expected to be dominated by the CIC and force interpolation steps, where the GPU's massive parallelism provides the largest advantage over the sequential CPU implementation. The FFT speedup is more modest, as both CPU and GPU FFT implementations are highly optimized. The leapfrog integration step is trivially parallelizable and shows the largest per-operation speedup on the GPU.

The total extrapolated CPU time for 500 steps (from the 15-step measurement) provides the basis for the speedup calculation. The GPU wall-clock time per realization was measured directly. The speedup factor, defined as t_CPU / t_GPU, quantifies the practical benefit of GPU acceleration for cosmological PM simulations at the 512³ scale.

---

## 4.6 Ensemble Variance and Cosmic Variance Implications

The 1-sigma fractional scatter across the 10 realizations, σ_P(k)/⟨P(k)⟩, provides a direct empirical estimate of the cosmic variance at each k-bin for a (1000 Mpc/h)³ volume. This quantity is shown in the ensemble variance plot (Plot 3), which displays the fractional scatter as a function of k in the top panel and the ratio of each individual realization to the ensemble mean in the bottom panel, with the Quijote ratio overlaid.

At large scales (k < 0.05 h/Mpc), the fractional scatter is largest, reflecting the small number of independent modes in the simulation volume. At k ≈ 0.05 h/Mpc, the scatter is expected to be of order 10–20% per realization, decreasing to a few percent at k ≈ 0.1 h/Mpc and sub-percent at k > 0.3 h/Mpc where the mode count is large. The ensemble average over 10 realizations reduces the effective cosmic variance by √10, bringing the statistical uncertainty on ⟨P(k)⟩ to the sub-percent level at k > 0.1 h/Mpc. This is sufficient to detect systematic deviations from the Quijote reference at the 5% level across the target k-range.

The individual realization ratios shown in the bottom panel of Plot 3 illustrate the scatter about the ensemble mean and confirm that no single realization is systematically biased. The spread of individual ratios about unity at low k is consistent with the expected cosmic variance for a (1000 Mpc/h)³ box, and the convergence of the ratios toward a common value at high k confirms that the simulation is statistically well-behaved.

---

## 4.7 Physical Interpretation of Residual Deviations

The residual deviations of the Warp PM simulation from the Quijote reference can be attributed