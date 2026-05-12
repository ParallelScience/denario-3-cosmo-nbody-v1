1. **Initial Conditions Generation**: Compute the linear matter power spectrum at $z=127$ using `camb` with the Quijote fiducial cosmology. Generate a Gaussian random field in Fourier space using a fixed random seed. Displace particles from a regular $512^3$ grid and assign velocities using the Zel'dovich Approximation (ZA). Save the initial particle state to the project data directory.

2. **Warp PM Kernel Implementation**: Develop the Particle-Mesh (PM) gravity solver in NVIDIA Warp.
    - Implement a CIC mass assignment kernel, ensuring proper normalization by cell volume.
    - Implement the Poisson solver in Fourier space, ensuring the density field is converted to density contrast $\delta = (\rho - \bar{\rho}) / \bar{\rho}$ and that the $k=0$ mode is handled correctly.
    - Develop a force interpolation kernel with a grid-based softening treatment to prevent numerical instabilities.
    - Implement a Leapfrog (kick-drift-kick) integrator, reusing GPU buffers for the density grid and potential field to minimize memory allocation overhead.

3. **Simulation Execution**: Run the Warp simulation from $z=127$ to $z=0$. Define the time-step size using the Courant condition: $\Delta t = \eta \sqrt{\text{cell\_size} / \text{acceleration}}$, where $\eta$ is a stability constant. Save particle snapshots at $z=0, 0.5, 1,$ and $2$.

4. **CPU Baseline Implementation**: Develop an equivalent PM simulation using `numpy` and `scipy.fft`. Ensure the implementation uses the same CIC, Poisson, and Leapfrog logic. Configure the CPU environment to run on a fixed number of threads (or single-core) to maintain a clear baseline, and verify that system RAM is sufficient to hold the $512^3$ arrays without disk swapping.

5. **Performance Benchmarking**: Measure wall-clock time for both Warp and CPU implementations. Perform a per-step breakdown (mass assignment, FFT, force interpolation, integration). Include a "warm-up" phase (running several dummy steps) for the GPU kernels to ensure compilation and initialization overheads are excluded from the timing results.

6. **Power Spectrum Estimation**: Compute the matter power spectrum $P(k)$ for the $z=0$ snapshot.
    - Perform CIC density assignment and convert to density contrast $\delta$.
    - Apply a 3D FFT and calculate the power spectrum by averaging squared Fourier modes in radial bins.
    - Deconvolve the CIC window function $W(k) = \text{sinc}^2(k_x \Delta/2) \text{sinc}^2(k_y \Delta/2) \text{sinc}^2(k_z \Delta/2)$.
    - Subtract the shot noise contribution ($1/\bar{n}$) from the final power spectrum.

7. **Validation and Comparison**: Download the reference Quijote fiducial $P(k)$ data. Calculate the ratio $P_{warp}(k) / P_{quijote}(k)$ for $0.01 \leq k \leq 1 \, h/\text{Mpc}$. Generate a plot to assess the 5% agreement target for $k < 0.3 \, h/\text{Mpc}$.

8. **Analysis of Deviations**: Analyze discrepancies between the Warp simulation and the Quijote reference. Characterize the impact of using ZA initial conditions versus 2LPT, evaluate the resolution limits of the $512^3$ PM grid, and document findings regarding high-$k$ mode deviations.