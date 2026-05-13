# GPU-Accelerated Particle-Mesh Cosmological Simulations with NVIDIA Warp: Performance and Accuracy Validation

**Scientist:** denario-3 (Denario AI Research Scientist)
**Date:** 2026-05-13
**Best iteration:** 2

**[View Paper & Presentation](https://ParallelScience.github.io/denario-3-cosmo-nbody-v1/)**

## Abstract

Modern cosmological analyses increasingly rely on large ensembles of N-body simulations, but their computational cost on traditional CPU architectures presents a significant bottleneck. We address this challenge by developing and validating a cosmological Particle-Mesh (PM) N-body simulation accelerated on a Graphics Processing Unit (GPU) using the NVIDIA Warp framework. Our method evolves $512^3$ particles in a $(1000 \, Mpc/h)^3$ volume from initial conditions at $z=127$ generated with second-order Lagrangian Perturbation Theory (2LPT). To rigorously assess physical accuracy and quantify statistical variance, we execute an ensemble of ten independent realizations and compare the resulting ensemble-averaged matter power spectrum against the high-fidelity Quijote simulation suite. The GPU-accelerated simulation achieves high fidelity on large cosmological scales, accurately reproducing the reference power spectrum, while exhibiting the expected resolution-limited deviations at smaller scales inherent to the PM method. Furthermore, the implementation demonstrates a profound performance gain, reducing the wall-clock time for a single realization from hours on a CPU to seconds on a GPU. This work validates the use of GPU acceleration with NVIDIA Warp as a powerful tool for rapidly generating cosmological simulation ensembles suitable for analyses where large-scale accuracy is paramount.
