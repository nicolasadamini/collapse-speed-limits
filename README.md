# No-Signaling Speed Limits on Continuous Wavefunction Collapse

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24484416.svg)](https://doi.org/10.5281/zenodo.24484416)

**Author:** Nicolas Adamini (Independent researcher)  
**Preprint (Zenodo):** [https://doi.org/10.5281/zenodo.24484416](https://doi.org/10.5281/zenodo.24484416)

This repository contains the manuscript and standalone reproducible simulation code for:

> **No-Signaling Speed Limits on Continuous Wavefunction Collapse: A Universal Log-Tail Bound and Mixed-State Finite-Time Escape Hatches**

## Contents

- [`paper/collapse_speed_limits.pdf`](paper/collapse_speed_limits.pdf) — Compiled 8-page manuscript (REVTeX 4.2, two-column format).
- [`paper/collapse_speed_limits.tex`](paper/collapse_speed_limits.tex) — Full LaTeX source.
- [`reproduce_one_outcome_tables.py`](reproduce_one_outcome_tables.py) — Standalone Python/NumPy script reproducing every numerical entry in Tables I–III, the extended-state two-observer consistency check, and the damped-Rabi non-commuting Hamiltonian integrator check.

## Summary

We establish a $d$-level speed limit governing the mathematical tails problem of continuous objective collapse models under pure-dephasing measurement dynamics ($[H, P_k] = 0$):

1. **Pure-State Continuous Zero-Tail No-Go & Universal Log-Tail Floor (Theorem 1):** For any Markovian continuous Itô diffusion on $\mathbb{C}P^{d-1}$ unraveling a Born-preserving GKSL master equation with finite integrated dephasing $\int_0^T \Gamma_{\max}(t)\,dt < \infty$, no initially populated branch can reach zero in finite time ($\mathbb{P}(p_j(t)>0\ \forall t<\infty)=1$), and every branch obeys $\mathbb{E}[\ln(p_j(t)/p_j(0))] \ge -4\int_0^t \Gamma_{\max}(s)\mathbb{E}[(1-p_j(s))^2]\,ds \ge -4(1-p_j(0))\int_0^t \Gamma_{\max}(s)\,ds$.
2. **Global $d=2$ Optimality of Homodyne CSL (Corollary 1):** Standard homodyne CSL saturates the pointwise volatility bound $\|\vec{v}_1\|^2 \le 8\Gamma(1-p)^2$ and globally minimizes $\mathbb{E}[\ln p_t]$ among all $d=2$ Markovian continuous pure-state unravelings.
3. **Four Sufficient No-Signaling Escape Hatches (Section III):** Relaxing continuity (Poisson jumps), finite integrated dephasing (Brody–Hughston bridges), Markovianity (a unique Erlang-2 randomized Brody–Hughston mixture reproducing $e^{\mathcal{L}t}$), or trajectory purity (a Markovian continuous mixed-state collapse SDE on $\mathcal{D}(\mathbb{C}^d \otimes \mathcal{H}_B)$ with exact closed-form mean collapse time).

## Reproducing All Numerical Tables

Requires Python 3.10+ and NumPy:

```bash
python3 reproduce_one_outcome_tables.py
```

Running this script (~45 seconds on a modern laptop) reproduces every numerical value reported in Tables I, II, and III, Section IV.B, and Appendix A using explicit NumPy seeds (`1001`–`6001`).

## Citation

```bibtex
@misc{adamini2026collapse,
  author       = {Adamini, Nicolas},
  title        = {No-Signaling Speed Limits on Continuous Wavefunction Collapse: A Universal Log-Tail Bound and Mixed-State Finite-Time Escape Hatches},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.24484416},
  url          = {https://doi.org/10.5281/zenodo.24484416}
}
```

## License

- Code (`reproduce_one_outcome_tables.py`): MIT License (see [`LICENSE`](LICENSE)).
- Manuscript (`paper/collapse_speed_limits.tex`, `paper/collapse_speed_limits.pdf`): Creative Commons Attribution 4.0 International (CC BY 4.0).
