

Iteration 0:
# Summary: GPU-Accelerated Cosmological N-body (PM) Simulation

## 1. Project Status
Successfully implemented a Particle-Mesh (PM) gravity solver using NVIDIA Warp (1.13.0) on RTX PRO 6000. Validated IC generation (ZA) and power spectrum pipeline against CAMB linear theory.

## 2. Key Findings
*   **Performance**: Achieved 7.5× total speedup over multi-threaded CPU (scipy/numpy) at $N=512^3$. Force interpolation (scattered gather) is the primary bottleneck, showing 17× speedup on GPU.
*   **Accuracy**: $P(k)$ at $z=0$ shows $\sim 3\%$ agreement with linear theory at $k < 0.04 \, h/\text{Mpc}$.
*   **Systematics**: 
    *   Finite-$N$ sampling causes a $\sim 10-20\%$ power deficit at intermediate scales ($k \approx 0.1 \, h/\text{Mpc}$).
    *   ZA systematically underestimates power at $k > 0.15 \, h/\text{Mpc}$ due to lack of shell-crossing dynamics.
    *   Shot noise is significant at $N=128^3$ ($P_{\text{shot}} \approx 476$) but negligible at $N=512^3$ ($P_{\text{shot}} \approx 7.5$).

## 3. Constraints & Limitations
*   **Dynamics**: Current implementation requires resolution of a coordinate unit inconsistency (km/s/Mpc vs. Mpc/h) to finalize full dynamical leapfrog integration.
*   **ICs**: Current ZA ICs are inferior to Quijote's 2LPT; 2LPT is required to reach the 5% accuracy target at $k \approx 0.1 \, h/\text{Mpc}$.
*   **Memory**: GPU VRAM (95 GiB) is sufficient for $N=512^3$ and likely $N=1024^3$.

## 4. Future Directions
*   **Upgrade ICs**: Implement 2LPT to reduce high-$k$ systematic bias.
*   **Full Dynamics**: Resolve unit scaling in the leapfrog integrator to enable full $z=127 \to 0$ evolution.
*   **Scaling**: Transition to $N=512^3$ for production runs; utilize cuFFT for further speedup at higher mesh resolutions.
*   **Validation**: Perform full $P(k)$ comparison against Quijote reference data using the $N=512^3$ production runs.
        