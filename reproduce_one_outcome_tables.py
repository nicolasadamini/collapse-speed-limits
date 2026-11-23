#!/usr/bin/env python3
"""Standalone reproducible simulation script for the manuscript:
'No-Signaling Speed Limits on Continuous Wavefunction Collapse:
 A Universal Log-Tail Bound and Mixed-State Finite-Time Escape Hatches'

Dependencies: Python >= 3.10, NumPy.
Usage: python3 reproduce_one_outcome_tables.py

Reproduces every numerical entry in Tables I, II, III, the extended-state
two-observer consistency check (Section V.B), and the damped-Rabi non-commuting
Hamiltonian integrator test (Section V.B / Appendix A).
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np


def _shannon_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def _kimura_mean_fixation_time(probs: np.ndarray, sigma_p: float) -> float:
    total = 0.0
    for p in probs:
        q = 1.0 - float(p)
        if q > 0.0:
            total += -q * math.log(q)
    return (2.0 / (sigma_p ** 2)) * total


def _trace_distance(rho1: np.ndarray, rho2: np.ndarray) -> float:
    diff = rho1 - rho2
    eigs = np.linalg.eigvalsh(0.5 * (diff + diff.conj().T))
    return float(0.5 * np.sum(np.abs(eigs)))


def run_two_observer_check(
    theta: float = math.pi / 4.0,
    gamma_a: float = 1.0,
    gamma_b: float = 1.3,
    sigma_p: float = 1.5,
    t_final: float = 14.0,
    dt: float = 1e-3,
    n_traj: int = 1500,
    seed: int = 5001,
) -> dict[str, Any]:
    """Extended-state two-observer consistency check on |Phi+> (Section III.B.2 & V.B)."""
    c2 = math.cos(0.5 * theta)
    s2 = math.sin(0.5 * theta)
    psi0 = np.array([c2, -s2, s2, c2], dtype=np.float64) / math.sqrt(2.0)
    rho0 = np.outer(psi0, psi0)
    p_target = psi0 ** 2

    rng = np.random.default_rng(seed)
    n_steps = int(round(t_final / dt))
    check_step = int(round(1.0 / dt))
    sqrt_dt = math.sqrt(dt)
    sig_ra = math.sqrt(2.0 * gamma_a)
    sig_rb = math.sqrt(2.0 * gamma_b)

    mu_a = np.ones(n_traj)
    mu_b = np.ones(n_traj)
    P = np.tile(p_target, (n_traj, 1))
    min_eig_seen = 0.0
    rho_ab_check_sum = np.zeros((4, 4), dtype=np.float64)

    for step in range(1, n_steps + 1):
        a_p1 = mu_a > 0.0
        if np.any(a_p1):
            m = mu_a[a_p1]
            w = rng.standard_normal(len(m)) * sqrt_dt
            mn = m - gamma_a * m * dt + sig_ra * np.sqrt(np.maximum(m * (1.0 - m), 0.0)) * w
            mn += 0.25 * (sig_ra ** 2) * (1.0 - 2.0 * m) * (w ** 2 - dt)
            mn = np.where(mn > 1.0, 2.0 - mn, mn)
            mu_a[a_p1] = np.clip(mn, 0.0, 1.0)

        b_p1 = mu_b > 0.0
        if np.any(b_p1):
            m = mu_b[b_p1]
            w = rng.standard_normal(len(m)) * sqrt_dt
            mn = m - gamma_b * m * dt + sig_rb * np.sqrt(np.maximum(m * (1.0 - m), 0.0)) * w
            mn += 0.25 * (sig_rb ** 2) * (1.0 - 2.0 * m) * (w ** 2 - dt)
            mn = np.where(mn > 1.0, 2.0 - mn, mn)
            mu_b[b_p1] = np.clip(mn, 0.0, 1.0)

        p_a = P[:, 0] + P[:, 1]
        a_p2 = (mu_a <= 0.0) & (p_a > 1e-12) & (p_a < 1.0 - 1e-12)
        if np.any(a_p2):
            idx = np.flatnonzero(a_p2)
            pa = p_a[idx]
            w = rng.standard_normal(len(idx)) * sqrt_dt
            pa_new = pa + sigma_p * np.sqrt(np.maximum(pa * (1.0 - pa), 0.0)) * w
            pa_new += 0.25 * (sigma_p ** 2) * (1.0 - 2.0 * pa) * (w ** 2 - dt)
            pa_new = np.clip(pa_new, 0.0, 1.0)
            s0 = pa_new / pa
            s1 = (1.0 - pa_new) / (1.0 - pa)
            P[idx, 0] *= s0
            P[idx, 1] *= s0
            P[idx, 2] *= s1
            P[idx, 3] *= s1

        p_b = P[:, 0] + P[:, 2]
        b_p2 = (mu_b <= 0.0) & (p_b > 1e-12) & (p_b < 1.0 - 1e-12)
        if np.any(b_p2):
            idx = np.flatnonzero(b_p2)
            pb = p_b[idx]
            w = rng.standard_normal(len(idx)) * sqrt_dt
            pb_new = pb + sigma_p * np.sqrt(np.maximum(pb * (1.0 - pb), 0.0)) * w
            pb_new += 0.25 * (sigma_p ** 2) * (1.0 - 2.0 * pb) * (w ** 2 - dt)
            pb_new = np.clip(pb_new, 0.0, 1.0)
            s0 = pb_new / pb
            s1 = (1.0 - pb_new) / (1.0 - pb)
            P[idx, 0] *= s0
            P[idx, 2] *= s0
            P[idx, 1] *= s1
            P[idx, 3] *= s1

        if step % 250 == 0 or step == check_step:
            n_audit = n_traj if step == check_step else min(100, n_traj)
            for i in range(n_audit):
                ma, mb = mu_a[i], mu_b[i]
                pi = P[i]
                pa_i = pi[0] + pi[1]
                pb_i = pi[0] + pi[2]
                c01 = rho0[0, 1] * (pa_i / 0.5) * mb
                c23 = rho0[2, 3] * ((1.0 - pa_i) / 0.5) * mb
                c02 = rho0[0, 2] * (pb_i / 0.5) * ma
                c13 = rho0[1, 3] * ((1.0 - pb_i) / 0.5) * ma
                c03 = rho0[0, 3] * ma * mb
                c12 = rho0[1, 2] * ma * mb
                rho_i = np.array([
                    [pi[0], c01, c02, c03],
                    [c01, pi[1], c12, c13],
                    [c02, c12, pi[2], c23],
                    [c03, c13, c23, pi[3]],
                ], dtype=np.float64)
                if step == check_step:
                    rho_ab_check_sum += rho_i
                if i < 100:
                    ev = float(np.min(np.linalg.eigvalsh(rho_i)))
                    if ev < min_eig_seen:
                        min_eig_seen = ev

    rho_ab_avg = rho_ab_check_sum / n_traj
    ea, eb = math.exp(-gamma_a), math.exp(-gamma_b)
    H_target = np.array([
        [1.0, eb, ea, ea * eb],
        [eb, 1.0, ea * eb, ea],
        [ea, ea * eb, 1.0, eb],
        [ea * eb, ea, eb, 1.0],
    ])
    rho_ab_target = rho0 * H_target
    corr_samples = P[:, 0] - P[:, 1] - P[:, 2] + P[:, 3]
    return {
        "seed": seed,
        "min_eigenvalue_seen": min_eig_seen,
        "lindblad_max_entry_error": float(np.max(np.abs(rho_ab_avg - rho_ab_target))),
        "correlator_empirical": float(np.mean(corr_samples)),
        "correlator_se": float(np.std(corr_samples, ddof=1) / math.sqrt(n_traj)),
        "correlator_target": math.cos(theta),
    }


def run_damped_rabi_d2_check(
    omega: float = 1.0,
    gamma: float = 2.0,
    sigma_p: float = 2.0,
    t_final: float = 4.0,
    t_mid: float = 0.5,
    dt: float = 5e-4,
    n_traj: int = 2000,
    seed: int = 6001,
) -> dict[str, Any]:
    """Damped-Rabi test of Exit D2 under H = omega * sigma_x from |0> (Appendix A)."""
    n_steps = int(round(t_final / dt))
    mid_step = int(round(t_mid / dt))
    sqrt_dt = math.sqrt(dt)
    sigma_r = math.sqrt(2.0 * gamma)

    # Exact Lindblad reference via RK2
    p_det, x_det, y_det = 1.0, 0.0, 0.0
    sz_det_mid, sy_det_mid = 0.0, 0.0
    for step in range(1, n_steps + 1):
        dp1 = -2.0 * omega * y_det
        dx1 = -gamma * x_det
        dy1 = omega * (2.0 * p_det - 1.0) - gamma * y_det
        pm = p_det + 0.5 * dt * dp1
        xm = x_det + 0.5 * dt * dx1
        ym = y_det + 0.5 * dt * dy1
        p_det += dt * (-2.0 * omega * ym)
        x_det += dt * (-gamma * xm)
        y_det += dt * (omega * (2.0 * pm - 1.0) - gamma * ym)
        if step == mid_step:
            sz_det_mid = 2.0 * p_det - 1.0
            sy_det_mid = -2.0 * y_det

    rng = np.random.default_rng(seed)
    p = np.ones(n_traj, dtype=np.float64)
    x = np.zeros(n_traj, dtype=np.float64)
    y = np.zeros(n_traj, dtype=np.float64)
    min_eig = 0.0
    sz_emp_mid = sz_se_mid = sy_emp_mid = sy_se_mid = 0.0

    for step in range(1, n_steps + 1):
        # 1. Exact unitary rotation under H = omega * sigma_x
        theta = 2.0 * omega * dt
        c, s = math.cos(theta), math.sin(theta)
        sz = 2.0 * p - 1.0
        sz_new = sz * c - 2.0 * y * s
        y_new = y * c + 0.5 * sz * s
        p = np.clip(0.5 * (1.0 + sz_new), 0.0, 1.0)
        y = y_new

        # 2. Symmetric positivity-preserving martingale step for p
        r = np.hypot(x, y)
        pp = p * (1.0 - p)
        u = np.maximum(pp - r * r, 0.0)
        w1 = rng.standard_normal(n_traj) * sqrt_dt
        w2 = rng.standard_normal(n_traj) * sqrt_dt

        amp_p = sigma_p * (u / np.sqrt(np.maximum(pp, 1e-30)))
        disc = np.sqrt(np.maximum(0.25 - r * r, 0.0))
        p_lo, p_hi = 0.5 - disc, 0.5 + disc
        max_sym_p = np.maximum(0.0, np.minimum(p - p_lo, p_hi - p))
        p_new = p + np.clip(amp_p * w2, -max_sym_p, max_sym_p)

        # 3. Symmetric positivity-preserving radial step around r * exp(-gamma * dt)
        r_pos = r > 1e-15
        r_new = np.zeros_like(r)
        if np.any(r_pos):
            rr = r[r_pos]
            ww = w1[r_pos]
            r_max = np.sqrt(np.maximum(p_new[r_pos] * (1.0 - p_new[r_pos]), 0.0))
            r_det = rr * math.exp(-gamma * dt)
            amp_r = sigma_r * np.sqrt(np.maximum(r_det * np.maximum(r_max * r_max - r_det * r_det, 0.0), 0.0))
            max_sym_r = np.maximum(0.0, np.minimum(r_det, r_max - r_det))
            r_new[r_pos] = r_det + np.clip(amp_r * ww, -max_sym_r, max_sym_r)
            scale = r_new[r_pos] / rr
            x[r_pos] *= scale
            y[r_pos] *= scale
        x[~r_pos] = 0.0
        y[~r_pos] = 0.0
        p = p_new

        if step == mid_step:
            sz_arr = 2.0 * p - 1.0
            sy_arr = -2.0 * y
            sz_emp_mid = float(np.mean(sz_arr))
            sz_se_mid = float(np.std(sz_arr, ddof=1) / math.sqrt(n_traj))
            sy_emp_mid = float(np.mean(sy_arr))
            sy_se_mid = float(np.std(sy_arr, ddof=1) / math.sqrt(n_traj))

        u_check = p * (1.0 - p) - (x * x + y * y)
        ev = float(np.min(0.5 - np.sqrt(np.maximum(0.25 - u_check, 0.0))))
        if ev < min_eig:
            min_eig = ev

    return {
        "seed": seed,
        "sz_mid_empirical": sz_emp_mid,
        "sz_mid_se": sz_se_mid,
        "sz_mid_lindblad": sz_det_mid,
        "sz_mid_z": (sz_emp_mid - sz_det_mid) / sz_se_mid,
        "sy_mid_empirical": sy_emp_mid,
        "sy_mid_se": sy_se_mid,
        "sy_mid_lindblad": sy_det_mid,
        "sy_mid_z": (sy_emp_mid - sy_det_mid) / sy_se_mid,
        "mean_abs_sz_t4": float(np.mean(np.abs(2.0 * p - 1.0))),
        "min_eigenvalue_seen": min_eig,
    }


def generate_all_tables() -> dict[str, Any]:
    p0 = 0.3
    gamma = 1.0
    n_t1 = 4000
    z0 = math.sqrt(p0 * (1.0 - p0))
    p_hi = 0.5 + math.sqrt(3.0) / 4.0
    sr = sp = math.sqrt(2.0 * gamma)

    table1: list[dict[str, Any]] = []

    # 1. CSL (seed=1001)
    rng = np.random.default_rng(1001)
    dt = 1e-3
    sqrt_dt = math.sqrt(dt)
    pref = math.sqrt(8.0 * gamma)
    L = np.full(n_t1, math.log(p0 / (1.0 - p0)))
    p = np.full(n_t1, p0)
    for s in range(1, 6001):
        L += 4.0 * gamma * (2.0 * p - 1.0) * dt + pref * rng.standard_normal(n_t1) * sqrt_dt
        p = np.where(L >= 0.0, 1.0 / (1.0 + np.exp(-L)), np.exp(L) / (1.0 + np.exp(L)))
        if s == 1000:
            r1 = np.sqrt(np.maximum(p * (1.0 - p), 0.0))
            coh_m, coh_se = float(np.mean(r1)), float(np.std(r1, ddof=1) / math.sqrt(n_t1))
            logp = -np.logaddexp(0.0, -L)
            lp_m, lp_se = float(np.mean(logp)), float(np.std(logp, ddof=1) / math.sqrt(n_t1))
    L1 = np.zeros(n_t1)
    p1 = np.full(n_t1, 0.5)
    L2 = np.full(n_t1, math.log(p_hi / (1.0 - p_hi)))
    p2 = np.full(n_t1, p_hi)
    for _ in range(1000):
        L1 += 4.0 * gamma * (2.0 * p1 - 1.0) * dt + pref * rng.standard_normal(n_t1) * sqrt_dt
        p1 = 1.0 / (1.0 + np.exp(-np.clip(L1, -500, 500)))
        L2 += 4.0 * gamma * (2.0 * p2 - 1.0) * dt + pref * rng.standard_normal(n_t1) * sqrt_dt
        p2 = 1.0 / (1.0 + np.exp(-np.clip(L2, -500, 500)))
    dz = 0.5 * np.sqrt(np.maximum(p1 * (1.0 - p1), 0.0)) - np.sqrt(np.maximum(p2 * (1.0 - p2), 0.0))
    dz_m, dz_se = float(np.mean(dz)), float(np.std(dz, ddof=1) / math.sqrt(n_t1))
    table1.append({
        "model": "CSL", "seed": 1001,
        "born": float(np.mean(p)), "born_se": float(np.std(p, ddof=1) / math.sqrt(n_t1)),
        "coh": coh_m, "coh_se": coh_se, "dz": dz_m, "dz_se": dz_se, "dz_z": dz_m / dz_se,
        "tau": None, "tau_se": None, "logp": lp_m, "logp_se": lp_se,
    })

    # 2. Naive Pure WF (seed=1002)
    rng = np.random.default_rng(1002)
    sigma = math.sqrt(2.0 * gamma)
    p = np.full(n_t1, p0)
    ht = np.full(n_t1, np.nan)
    for s in range(1, 6001):
        act = (p > 0.0) & (p < 1.0)
        if np.any(act):
            ii = np.flatnonzero(act)
            pa = p[ii]
            w = rng.standard_normal(len(ii)) * sqrt_dt
            pn = np.clip(pa + sigma * np.sqrt(pa * (1.0 - pa)) * w + 0.25 * (sigma ** 2) * (1.0 - 2.0 * pa) * (w ** 2 - dt), 0.0, 1.0)
            p[ii] = pn
            ht[ii[(pn <= 0.0) | (pn >= 1.0)]] = s * dt
        if s == 1000:
            r1 = np.sqrt(np.maximum(p * (1.0 - p), 0.0))
            coh_m, coh_se = float(np.mean(r1)), float(np.std(r1, ddof=1) / math.sqrt(n_t1))
    p1 = np.full(n_t1, 0.5)
    p2 = np.full(n_t1, p_hi)
    for _ in range(1000):
        for arr in (p1, p2):
            act = (arr > 0.0) & (arr < 1.0)
            if np.any(act):
                pa = arr[act]
                w = rng.standard_normal(len(pa)) * sqrt_dt
                arr[act] = np.clip(pa + sigma * np.sqrt(pa * (1.0 - pa)) * w + 0.25 * (sigma ** 2) * (1.0 - 2.0 * pa) * (w ** 2 - dt), 0.0, 1.0)
    dz = 0.5 * np.sqrt(np.maximum(p1 * (1.0 - p1), 0.0)) - np.sqrt(np.maximum(p2 * (1.0 - p2), 0.0))
    dz_m, dz_se = float(np.mean(dz)), float(np.std(dz, ddof=1) / math.sqrt(n_t1))
    table1.append({
        "model": "WF_pure", "seed": 1002,
        "born": float(np.mean(p)), "born_se": float(np.std(p, ddof=1) / math.sqrt(n_t1)),
        "coh": coh_m, "coh_se": coh_se, "dz": dz_m, "dz_se": dz_se, "dz_z": dz_m / dz_se,
        "tau": float(np.nanmean(ht)), "tau_se": float(np.nanstd(ht, ddof=1) / math.sqrt(np.sum(~np.isnan(ht)))),
    })

    # 3. Exit A: Poisson (seed=1003)
    rng = np.random.default_rng(1003)
    jt = rng.exponential(1.0 / gamma, size=n_t1)
    out = (rng.random(n_t1) < p0).astype(float)
    r1 = np.where(jt > 1.0, z0, 0.0)
    pf = np.where(jt <= 6.0, out, p0)
    dz = np.where(rng.exponential(1.0 / gamma, n_t1) > 1.0, 0.25, 0.0) - np.where(rng.exponential(1.0 / gamma, n_t1) > 1.0, 0.25, 0.0)
    dz_m, dz_se = float(np.mean(dz)), float(np.std(dz, ddof=1) / math.sqrt(n_t1))
    table1.append({
        "model": "ExitA_Poisson", "seed": 1003,
        "born": float(np.mean(pf)), "born_se": float(np.std(pf, ddof=1) / math.sqrt(n_t1)),
        "coh": float(np.mean(r1)), "coh_se": float(np.std(r1, ddof=1) / math.sqrt(n_t1)),
        "dz": dz_m, "dz_se": dz_se, "dz_z": dz_m / dz_se,
        "tau": float(np.mean(jt)), "tau_se": float(np.std(jt, ddof=1) / math.sqrt(n_t1)),
    })

    # 4. Exit C: Erlang-2 BH (seed=1004)
    rng = np.random.default_rng(1004)
    T = rng.gamma(2.0, 1.0 / gamma, size=n_t1)

    def samp_csl(pi: float, taus: np.ndarray) -> np.ndarray:
        Z = (rng.random(len(taus)) < pi).astype(float)
        Lv = math.log(pi / (1.0 - pi)) + 4.0 * (2.0 * Z - 1.0) * taus + np.sqrt(8.0 * taus) * rng.standard_normal(len(taus))
        return np.where(Lv >= 0.0, 1.0 / (1.0 + np.exp(-Lv)), np.exp(Lv) / (1.0 + np.exp(Lv)))

    a1 = T > 1.0
    pc = np.zeros(n_t1)
    pc[~a1] = (rng.random(int(np.sum(~a1))) < p0).astype(float)
    pc[a1] = samp_csl(p0, -np.log(1.0 - 1.0 / T[a1]))
    r1 = np.sqrt(np.maximum(pc * (1.0 - pc), 0.0))
    af = T > 10.0
    pf = np.zeros(n_t1)
    pf[~af] = (rng.random(int(np.sum(~af))) < p0).astype(float)
    if np.any(af):
        pf[af] = samp_csl(p0, -np.log(1.0 - 10.0 / T[af]))
    T1 = rng.gamma(2.0, 1.0 / gamma, n_t1)
    T2 = rng.gamma(2.0, 1.0 / gamma, n_t1)
    z1 = np.zeros(n_t1)
    z2 = np.zeros(n_t1)
    b1, b2 = T1 > 1.0, T2 > 1.0
    p1c = samp_csl(0.5, -np.log(1.0 - 1.0 / T1[b1]))
    p2c = samp_csl(p_hi, -np.log(1.0 - 1.0 / T2[b2]))
    z1[b1] = 0.5 * np.sqrt(np.maximum(p1c * (1.0 - p1c), 0.0))
    z2[b2] = np.sqrt(np.maximum(p2c * (1.0 - p2c), 0.0))
    dz = z1 - z2
    dz_m, dz_se = float(np.mean(dz)), float(np.std(dz, ddof=1) / math.sqrt(n_t1))
    table1.append({
        "model": "ExitC_Erlang2", "seed": 1004,
        "born": float(np.mean(pf)), "born_se": float(np.std(pf, ddof=1) / math.sqrt(n_t1)),
        "coh": float(np.mean(r1)), "coh_se": float(np.std(r1, ddof=1) / math.sqrt(n_t1)),
        "dz": dz_m, "dz_se": dz_se, "dz_z": dz_m / dz_se,
        "tau": float(np.mean(T)), "tau_se": float(np.std(T, ddof=1) / math.sqrt(n_t1)),
    })

    # 5. Exit D1: Stage-Gated (seed=1005)
    rng = np.random.default_rng(1005)
    dt5 = 5e-4
    sqrt_dt5 = math.sqrt(dt5)
    mu = np.ones(n_t1)
    p = np.full(n_t1, p0)
    tau2 = np.full(n_t1, np.nan)
    for s in range(1, 20001):
        m1 = mu > 0.0
        if np.any(m1):
            m = mu[m1]
            w = rng.standard_normal(len(m)) * sqrt_dt5
            mn = m - gamma * m * dt5 + sr * np.sqrt(np.maximum(m * (1.0 - m), 0.0)) * w + 0.25 * (sr ** 2) * (1.0 - 2.0 * m) * (w ** 2 - dt5)
            mn = np.where(mn > 1.0, 2.0 - mn, mn)
            mu[m1] = np.clip(mn, 0.0, 1.0)
        m2 = (mu <= 0.0) & (p > 0.0) & (p < 1.0)
        if np.any(m2):
            ii = np.flatnonzero(m2)
            pa = p[ii]
            w = rng.standard_normal(len(ii)) * sqrt_dt5
            pn = np.clip(pa + sp * np.sqrt(np.maximum(pa * (1.0 - pa), 0.0)) * w + 0.25 * (sp ** 2) * (1.0 - 2.0 * pa) * (w ** 2 - dt5), 0.0, 1.0)
            p[ii] = pn
            tau2[ii[(pn <= 0.0) | (pn >= 1.0)]] = s * dt5
        if s == 2000:
            r1 = z0 * mu
            coh_m, coh_se = float(np.mean(r1)), float(np.std(r1, ddof=1) / math.sqrt(n_t1))
    mu1, mu2 = np.ones(n_t1), np.ones(n_t1)
    for _ in range(2000):
        for arr in (mu1, mu2):
            act = arr > 0.0
            if np.any(act):
                m = arr[act]
                w = rng.standard_normal(len(m)) * sqrt_dt5
                mn = m - gamma * m * dt5 + sr * np.sqrt(np.maximum(m * (1.0 - m), 0.0)) * w + 0.25 * (sr ** 2) * (1.0 - 2.0 * m) * (w ** 2 - dt5)
                mn = np.where(mn > 1.0, 2.0 - mn, mn)
                arr[act] = np.clip(mn, 0.0, 1.0)
    dz = 0.25 * (mu1 - mu2)
    dz_m, dz_se = float(np.mean(dz)), float(np.std(dz, ddof=1) / math.sqrt(n_t1))
    table1.append({
        "model": "ExitD1_StageGated", "seed": 1005,
        "born": float(np.mean(p)), "born_se": float(np.std(p, ddof=1) / math.sqrt(n_t1)),
        "coh": coh_m, "coh_se": coh_se, "dz": dz_m, "dz_se": dz_se, "dz_z": dz_m / dz_se,
        "tau": float(np.nanmean(tau2)), "tau_se": float(np.nanstd(tau2, ddof=1) / math.sqrt(np.sum(~np.isnan(tau2)))),
    })

    # 6. Exit D2: Smooth-Coupled (seed=1006)
    rng = np.random.default_rng(1006)
    p = np.full(n_t1, p0)
    r = np.full(n_t1, z0)
    tau_d2 = np.full(n_t1, np.nan)
    for s in range(1, 20001):
        act = (p > 0.0) & (p < 1.0)
        if np.any(act):
            ii = np.flatnonzero(act)
            pa, ra = p[ii], r[ii]
            pp = pa * (1.0 - pa)
            u = np.maximum(pp - ra * ra, 0.0)
            w1 = rng.standard_normal(len(ii)) * sqrt_dt5
            w2 = rng.standard_normal(len(ii)) * sqrt_dt5
            pa_n = np.clip(pa + sp * (u / np.sqrt(np.maximum(pp, 1e-30))) * w2, 0.0, 1.0)
            rp = ra > 0.0
            ra_n = np.zeros_like(ra)
            if np.any(rp):
                rs = ra[rp] - gamma * ra[rp] * dt5 + sr * np.sqrt(np.maximum(ra[rp] * u[rp], 0.0)) * w1[rp] + 0.25 * (sr ** 2) * (pp[rp] - 3.0 * ra[rp] ** 2) * (w1[rp] ** 2 - dt5)
                rm = np.sqrt(np.maximum(pa_n[rp] * (1.0 - pa_n[rp]), 0.0))
                rs = np.where(rs > rm, np.maximum(0.0, 2.0 * rm - rs), rs)
                ra_n[rp] = np.clip(rs, 0.0, rm)
            p[ii], r[ii] = pa_n, ra_n
            tau_d2[ii[(pa_n <= 0.0) | (pa_n >= 1.0)]] = s * dt5
        if s == 2000:
            coh_m, coh_se = float(np.mean(r)), float(np.std(r, ddof=1) / math.sqrt(n_t1))
    z_out = []
    for p_init, scale in ((0.5, 0.5), (p_hi, 1.0)):
        pa = np.full(n_t1, p_init)
        ra = np.full(n_t1, math.sqrt(p_init * (1.0 - p_init)))
        for _ in range(2000):
            act = (pa > 0.0) & (pa < 1.0)
            if np.any(act):
                ii = np.flatnonzero(act)
                pc, rc = pa[ii], ra[ii]
                pp = pc * (1.0 - pc)
                u = np.maximum(pp - rc * rc, 0.0)
                w1 = rng.standard_normal(len(ii)) * sqrt_dt5
                w2 = rng.standard_normal(len(ii)) * sqrt_dt5
                pn = np.clip(pc + sp * (u / np.sqrt(np.maximum(pp, 1e-30))) * w2, 0.0, 1.0)
                rp = rc > 0.0
                rn = np.zeros_like(rc)
                if np.any(rp):
                    rs = rc[rp] - gamma * rc[rp] * dt5 + sr * np.sqrt(np.maximum(rc[rp] * u[rp], 0.0)) * w1[rp] + 0.25 * (sr ** 2) * (pp[rp] - 3.0 * rc[rp] ** 2) * (w1[rp] ** 2 - dt5)
                    rm = np.sqrt(np.maximum(pn[rp] * (1.0 - pn[rp]), 0.0))
                    rs = np.where(rs > rm, np.maximum(0.0, 2.0 * rm - rs), rs)
                    rn[rp] = np.clip(rs, 0.0, rm)
                pa[ii], ra[ii] = pn, rn
        z_out.append(scale * ra)
    dz = z_out[0] - z_out[1]
    dz_m, dz_se = float(np.mean(dz)), float(np.std(dz, ddof=1) / math.sqrt(n_t1))
    table1.append({
        "model": "ExitD2_Smooth", "seed": 1006,
        "born": float(np.mean(p)), "born_se": float(np.std(p, ddof=1) / math.sqrt(n_t1)),
        "coh": coh_m, "coh_se": coh_se, "dz": dz_m, "dz_se": dz_se, "dz_z": dz_m / dz_se,
        "tau": float(np.nanmean(tau_d2)), "tau_se": float(np.nanstd(tau_d2, ddof=1) / math.sqrt(np.sum(~np.isnan(tau_d2)))),
    })

    # Table II: Multi-level d=3, 4 (seeds 2001, 2002)
    table2 = []
    for probs0, sd in (([0.5, 0.3, 0.2], 2001), ([0.4, 0.3, 0.2, 0.1], 2002)):
        p0_arr = np.array(probs0, dtype=float)
        p0_arr /= p0_arr.sum()
        d = len(p0_arr)
        n_m = 2000
        dt_m = 1e-3
        sq_m = math.sqrt(dt_m)
        rng = np.random.default_rng(sd)
        mu = np.ones(n_m)
        P = np.tile(p0_arr, (n_m, 1))
        tau1 = np.full(n_m, np.nan)
        tau2_m = np.full(n_m, np.nan)
        pairs = [(j, k) for j in range(d) for k in range(j + 1, d)]
        for step in range(1, 14001):
            m1 = mu > 0.0
            if np.any(m1):
                ii = np.flatnonzero(m1)
                m = mu[ii]
                w = rng.standard_normal(len(m)) * sq_m
                mn = m - gamma * m * dt_m + sr * np.sqrt(np.maximum(m * (1.0 - m), 0.0)) * w + 0.25 * (sr ** 2) * (1.0 - 2.0 * m) * (w ** 2 - dt_m)
                mn = np.where(mn > 1.0, 2.0 - mn, mn)
                mn = np.clip(mn, 0.0, 1.0)
                mu[ii] = mn
                tau1[ii[mn <= 0.0]] = step * dt_m
            a2 = (mu <= 0.0) & (np.max(P, axis=1) < 1.0 - 1e-12)
            if np.any(a2):
                ii = np.flatnonzero(a2)
                Pa = P[ii].copy()
                dP = np.zeros_like(Pa)
                for j, k in pairs:
                    both = (Pa[:, j] > 0.0) & (Pa[:, k] > 0.0)
                    if np.any(both):
                        w = rng.standard_normal(int(np.sum(both))) * sq_m
                        flux = np.clip(sp * np.sqrt(Pa[both, j] * Pa[both, k]) * w, -Pa[both, j], Pa[both, k])
                        dP[both, j] += flux
                        dP[both, k] -= flux
                Pa = np.clip(Pa + dP, 0.0, 1.0)
                Pa /= np.maximum(Pa.sum(axis=1, keepdims=True), 1e-30)
                P[ii] = Pa
                jf = np.max(Pa, axis=1) >= 1.0 - 1e-12
                if np.any(jf):
                    win = np.argmax(Pa[jf], axis=1)
                    fi = ii[jf]
                    P[fi] = 0.0
                    P[fi, win] = 1.0
                    tau2_m[fi] = step * dt_m
        t1m, t1se = float(np.nanmean(tau1)), float(np.nanstd(tau1, ddof=1) / math.sqrt(np.sum(~np.isnan(tau1))))
        t2m, t2se = float(np.nanmean(tau2_m)), float(np.nanstd(tau2_m, ddof=1) / math.sqrt(np.sum(~np.isnan(tau2_m))))
        theo1 = (math.pi ** 2) / (6.0 * gamma)
        theo2 = theo1 + _kimura_mean_fixation_time(p0_arr, sp)
        table2.append({
            "d": d, "seed": sd,
            "max_born_err": float(np.max(np.abs(P.mean(axis=0) - p0_arr))),
            "max_born_se": float(np.max(np.std(P, axis=0, ddof=1) / math.sqrt(n_m))),
            "tau1": t1m, "tau1_se": t1se, "theo1": theo1,
            "tau_tot": t2m, "tau_tot_se": t2se, "theo_tot": theo2,
        })

    # Table III: dt convergence (seeds 3000..3003, 4000..4003)
    table3 = []
    theo_d1 = (math.pi ** 2) / (6.0 * gamma) + (2.0 / (sp ** 2)) * _shannon_entropy(p0)
    for k_idx, dt_c in enumerate((2e-3, 5e-4, 1.25e-4, 6.25e-5)):
        n_c = 2000
        sq_c = math.sqrt(dt_c)
        n_st = int(round(14.0 / dt_c))
        rng = np.random.default_rng(3000 + k_idx)
        mu = np.ones(n_c)
        p = np.full(n_c, p0)
        td1 = np.full(n_c, np.nan)
        for s in range(1, n_st + 1):
            m1 = mu > 0.0
            if np.any(m1):
                m = mu[m1]
                w = rng.standard_normal(len(m)) * sq_c
                mn = m - gamma * m * dt_c + sr * np.sqrt(np.maximum(m * (1.0 - m), 0.0)) * w + 0.25 * (sr ** 2) * (1.0 - 2.0 * m) * (w ** 2 - dt_c)
                mn = np.where(mn > 1.0, 2.0 - mn, mn)
                mu[m1] = np.clip(mn, 0.0, 1.0)
            m2 = (mu <= 0.0) & (p > 0.0) & (p < 1.0)
            if np.any(m2):
                ii = np.flatnonzero(m2)
                pa = p[ii]
                w = rng.standard_normal(len(ii)) * sq_c
                pn = np.clip(pa + sp * np.sqrt(np.maximum(pa * (1.0 - pa), 0.0)) * w + 0.25 * (sp ** 2) * (1.0 - 2.0 * pa) * (w ** 2 - dt_c), 0.0, 1.0)
                p[ii] = pn
                td1[ii[(pn <= 0.0) | (pn >= 1.0)]] = s * dt_c
        d1m, d1se = float(np.nanmean(td1)), float(np.nanstd(td1, ddof=1) / math.sqrt(np.sum(~np.isnan(td1))))

        rng = np.random.default_rng(4000 + k_idx)
        p = np.full(n_c, p0)
        r = np.full(n_c, z0)
        td2 = np.full(n_c, np.nan)
        for s in range(1, n_st + 1):
            act = (p > 0.0) & (p < 1.0)
            if np.any(act):
                ii = np.flatnonzero(act)
                pa, ra = p[ii], r[ii]
                pp = pa * (1.0 - pa)
                rmax = np.sqrt(np.maximum(pp, 1e-30))
                mv = np.clip(ra / rmax, 0.0, 1.0)
                w1 = rng.standard_normal(len(ii)) * sq_c
                w2 = rng.standard_normal(len(ii)) * sq_c
                gp = sp * (1.0 - mv * mv) * rmax
                # Exact Milstein for b(p) = sigma_p*(1 - r^2/q)*sqrt(q): 0.5*b*b' = 0.25*sigma_p^2*(1 - mu^4)*(1 - 2p)
                pa_n = np.clip(pa + gp * w2 + 0.25 * (sp ** 2) * (1.0 - mv ** 4) * (1.0 - 2.0 * pa) * (w2 ** 2 - dt_c), 0.0, 1.0)
                rmax_n = np.sqrt(np.maximum(pa_n * (1.0 - pa_n), 0.0))
                rp = ra > 0.0
                ra_n = np.zeros_like(ra)
                if np.any(rp):
                    rr, rm, ww = ra[rp], rmax[rp], w1[rp]
                    rs = rr - gamma * rr * dt_c + sr * np.sqrt(np.maximum(rr * (rm - rr), 0.0)) * ww + 0.25 * (sr ** 2) * (rm - 2.0 * rr) * (ww ** 2 - dt_c)
                    rm_new = rmax_n[rp]
                    rs = np.where(rs > rm_new, np.maximum(0.0, 2.0 * rm_new - rs), rs)
                    ra_n[rp] = np.clip(rs, 0.0, rm_new)
                p[ii], r[ii] = pa_n, ra_n
                td2[ii[(pa_n <= 0.0) | (pa_n >= 1.0)]] = s * dt_c
        d2m, d2se = float(np.nanmean(td2)), float(np.nanstd(td2, ddof=1) / math.sqrt(np.sum(~np.isnan(td2))))
        table3.append({
            "dt": dt_c, "seed_d1": 3000 + k_idx, "seed_d2": 4000 + k_idx,
            "d1_tau": d1m, "d1_se": d1se, "d1_z": (d1m - theo_d1) / d1se,
            "d2_tau": d2m, "d2_se": d2se,
        })

    return {
        "table1": table1,
        "table2": table2,
        "table3": table3,
        "two_observer": run_two_observer_check(seed=5001),
        "damped_rabi_d2": run_damped_rabi_d2_check(seed=6001),
    }


if __name__ == "__main__":
    print(json.dumps(generate_all_tables(), indent=2))
