#!/usr/bin/env python3
"""
Compute the wall-liquid work of adsorption W(l) from continuation isotherms and
fit TorchScript regressors for the contact-angle / cap-prediction scripts.

This script fuses (and fixes) the two root notebooks:
  - 00_isotherm_wl.ipynb   (step 2: isotherm post-processing -> isotherm_work.pkl)
  - 00_interpolant.ipynb   (step 3: W(l), mu0(l) regressors -> postproc/*.pt)

Pipeline position:
  adsoprtion_continuation.py -> isotherm_profiles_*.pkl
      -> [THIS SCRIPT]
          step 2: per-frame work of adsorption W = trapz(omegaX - omega_bulk),
                  film thickness l from the adsorbed excess, gamma_wl / gamma_wg
                  -> isotherm_work.pkl
          step 3: piecewise-linear regressors W(l) and mu0(l), TorchScript-saved
                  -> data/dataset/postproc/work_regressor_Xb/work_regressor_{T:.2f}.pt
                  -> data/dataset/postproc/mu0_regressor/mu0_regressor_{T:.2f}.pt
      -> 00_compute_contact_angle.py / 00_cap_prediction.py

Prerequisites (assumed already computed):
  - the isotherm profiles pickle (ISOTHERM_PKL below)
  - data/dataset/postproc/phase_curve_nn.pkl  (rho_v, rho_l, gamma_lg per T)

Fixes with respect to the notebooks:
  - no CDFT model / ML weights needed: the bulk grand-potential baseline is
    estimated from the vapor-side plateau of omegaX (this removes the undefined
    `omegab`, the update_model(x_ passed as rho_) bug, and the dead device logic)
  - clean temperature loop (the notebook used the debug variable T_i instead of
    the loop variable t when rebuilding the model)
  - grid- and frame-dependent indices are derived from the data instead of being
    hard-coded (-200 column, frame [100], [:300], WOmX[-100]) and are clamped,
    so the script also works on meshes other than L=30 / Nx=400
  - duplicate l knots are removed before spline fitting (LinearSpline raises on
    non strictly-increasing knots; continuation curves can revisit l)
  - the fit window is selected with a boolean mask (the notebook used
    argwhere(...)[0] on *unsorted* l for the mu0 fit)
  - Winf is evaluated at l = validity instead of extrapolating to 1.1*validity
  - explicit configuration block; `shift` / `validity` no longer leak between
    notebook cells; all output directories are created automatically

Diagnostic plots (under plot_adsorption_work_wl_{WALL}/):
  - work_integral/T*/frame_*_l*.png -- omega-omega_b integral window every
                                      WORK_PLOT_EVERY_N frames (+ first/last/wg),
                                      with rho(x) and the l-tracked LG interface
  - summary/                       -- W vs l, gammas, theta_Y, regressor fits

The saved TorchScript modules keep the exact attribute schema expected by
src/libs/omega_minimization/utils.py:
  work regressor: forward(l), dy_dx(l), .min {l_min, W_min}, .max {l_max, W_max},
                  .wg {l_wg, W_wg} (0-d tensors), .Winf, .validity
  mu0 regressor : forward(l), dy_dx(l), .validity
"""
import os
import sys
import pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
WALL = "wc"
N_VALUE = 8000                      # particle loading branch to process

datadir = os.path.normpath(os.path.join(script_dir, "..", "data"))
outdir = os.path.normpath(os.path.join(script_dir, "..", "output"))
pkldir = os.path.join(datadir, "dataset", "pkl", f"profiles_wl_{WALL}")

# Input: output of adsoprtion_continuation.py. Switch to the L15 pickle
# (isotherm_profiles_wl_wc_L15.pkl) once that run is complete.
ISOTHERM_PKL = os.path.join(pkldir, "isotherm_profiles_wl_wc_L30dx01.pkl")
PHASE_CURVE_PKL = os.path.join(datadir, "dataset", "pkl", "bulk_tmd", "pc_md_operative_with_gamma.pkl")

# Outputs
WORK_PKL = os.path.join(pkldir, f"isotherm_work_wl_{WALL}.pkl")
W_REGRESSOR_DIR = os.path.join(pkldir, f"work_regressor_wl_{WALL}")
MU0_REGRESSOR_DIR = os.path.join(pkldir, f"mu0_regressor_wl_{WALL}")
PLOT_DIR = os.path.join(outdir, f"plot_adsorption_work_wl_{WALL}")
PLOT_SUMMARY_DIR = os.path.join(PLOT_DIR, "summary")
PLOT_WORK_DIR = os.path.join(PLOT_DIR, "work_integral")

# Step 2: work integral / film thickness
TAIL_FRAC = 0.015        # fraction of the grid treated as vapor-side plateau
TAIL_NMIN = 100          # minimum number of plateau points
FILM_X_OFFSET = 1.375    # film support starts this far from the left wall
VAPOR_SAMPLE_FRAC = 0.70 # rho sampling column for the bulk vapor density
LIQUID_SAMPLE_FRAC = 0.30
RHOB_VAPOR_FRAME = 100   # frame used for the vapor density (clamped to n_frames-1)
WG_SAMPLE_FRAC = 0.50    # vapor-side column used to detect the wall-gas frame
WG_MAX_FRAMES = 300      # only scan the early (low-loading) part of the curve

# Step 3: regressor fit window
VALIDITY = 25.0          # regressors are trusted for l <= VALIDITY
L_FIT_MIN = 0.0        # discard frames with l below this

TORCH_DTYPE = torch.double

# Diagnostic plots (step 2)
WORK_INTEGRAL_TOL = 1e-5   # fixed tol for W integral plots (matches the notebook)
WORK_XLIM_PAD = 5.0        # extra x-range past the integration upper bound
WORK_PLOT_EVERY_N = 50     # work_integral: plot every N continuation frames (1 = all)


# ------------------------------------------------------------------------------
# Step 2 helpers: work of adsorption and film thickness
# ------------------------------------------------------------------------------
def area_under_interface(x, omega, tail_frac=TAIL_FRAC, nmin=TAIL_NMIN, tol=None):
    """
    Integral of omega(x) minus its bulk (vapor-side plateau) baseline over the
    wall + interface region. This is the work of adsorption per unit area.

    Returns:
        area     : float, trapz of (omega - baseline) over [0, i1]
        baseline : float, vapor-side plateau value subtracted
        bounds   : (i0, i1) integration window indices
        tol_used : float, plateau-deviation threshold
    """
    x = np.asarray(x, dtype=np.float64).squeeze()
    w = np.asarray(omega, dtype=np.float64).squeeze()
    assert x.ndim == 1 and w.ndim == 1 and x.size == w.size

    n = x.size
    k = min(max(nmin, int(tail_frac * n)), n - 2)
    tail = w[-k:]
    baseline = float(np.median(tail))
    w_corr = w - baseline

    if tol is None:
        tol = 1.1 * float(np.std(tail - baseline)) + 1e-12

    # First tail point that still deviates from the plateau: integrate up to
    # there so that plateau noise does not accumulate into the work integral.
    idx = np.where(np.abs(w_corr[-k:]) > tol)[0]
    i0 = 0
    i1 = (n - k + int(idx[0])) if idx.size > 0 else (n - k)

    area = float(np.trapezoid(w_corr[i0:i1 + 1], x[i0:i1 + 1]))
    return area, baseline, (i0, i1), float(tol)


def stack_isotherm_frames(x, frames):
    """
    Stack the continuation frames of one (N, T) branch into tensors and compute
    the per-frame work of adsorption.

    frames: list of dicts with keys rho [B,1,Nx], omegaX [B,1,Nx], chem_pot [B,1]
            (frame 0 of the continuation curve is the seed and lacks omegaX /
            chem_pot; callers should pass row["isotherm"][1:]).
    """
    nx = x.shape[-1]
    i_vap = min(int(VAPOR_SAMPLE_FRAC * nx), nx - 1)
    i_liq = min(int(LIQUID_SAMPLE_FRAC * nx), nx - 1)

    rho_list, omega_list, chem_list, work_list = [], [], [], []
    rho_bv_list, rho_bl_list = [], []
    baseline_list, i0_list, i1_list, tol_list = [], [], [], []
    for frame in frames:
        if "omegaX" not in frame or "chem_pot" not in frame:
            continue
        rho = frame["rho"].detach().to("cpu", TORCH_DTYPE)[0]        # [1, Nx]
        omega = frame["omegaX"].detach().to("cpu", TORCH_DTYPE)[0]   # [1, Nx]
        chem = frame["chem_pot"].detach().to("cpu", TORCH_DTYPE).reshape(-1)[:1]  # [1]

        area, baseline, (i0, i1), tol_used = area_under_interface(
            x.numpy(), omega.numpy(), tol=WORK_INTEGRAL_TOL,
        )

        rho_list.append(rho)
        omega_list.append(omega)
        chem_list.append(chem)
        work_list.append(area)
        rho_bv_list.append(rho[0, i_vap])
        rho_bl_list.append(rho[0, i_liq])
        baseline_list.append(baseline)
        i0_list.append(i0)
        i1_list.append(i1)
        tol_list.append(tol_used)

    if not rho_list:
        raise ValueError("No usable continuation frames (missing omegaX/chem_pot).")

    return {
        "rho": torch.stack(rho_list, dim=0),          # [F, 1, Nx]
        "omega": torch.stack(omega_list, dim=0),      # [F, 1, Nx]
        "chem_pot": torch.stack(chem_list, dim=0),    # [F, 1]
        "WOmX": torch.tensor(work_list, dtype=TORCH_DTYPE),   # [F]
        "rhoBv": torch.stack(rho_bv_list, dim=0),     # [F]
        "rhoBl": torch.stack(rho_bl_list, dim=0),     # [F]
        "omega_baseline": np.asarray(baseline_list, dtype=np.float64),
        "work_i0": np.asarray(i0_list, dtype=np.int64),
        "work_i1": np.asarray(i1_list, dtype=np.int64),
        "work_tol": np.asarray(tol_list, dtype=np.float64),
    }


def film_thickness(x, rho, rho_bv, rho_bl):
    """
    Adsorbed-excess film thickness l for each frame.

    rhoX masks out the wall region (x < x_min + FILM_X_OFFSET) and subtracts the
    bulk vapor density; l = (integral of rhoX) / (rho_l - rho_v).
    """
    dx = x[1] - x[0]
    x0 = x.min() + FILM_X_OFFSET
    mask = torch.sigmoid((x - x0) * 100.0)
    rhoX = (rho - rho_bv) * mask                       # [F, 1, Nx]
    NX = rhoX.sum(dim=-1) * dx                         # [F, 1]
    l = NX / (rho_bl - rho_bv)                         # [F, 1]
    return rhoX, NX, l


def find_wg_frame(rhoX):
    """
    Heuristic from 00_isotherm_wl.ipynb: scan the early frames at a vapor-side
    column and return the first frame index where the (squared) excess density
    stops decreasing -- taken as the wall-gas state used for gamma_wg.

    Falls back to the argmin of the scanned signal if no such frame exists.
    """
    n_frames, _, nx = rhoX.shape
    col = min(int(WG_SAMPLE_FRAC * nx), nx - 1)
    n_scan = min(WG_MAX_FRAMES, n_frames)
    reduced = rhoX[:n_scan, 0, col] ** 2
    for i in range(3, n_scan):
        if reduced[i] >= reduced[i - 1]:
            return i
    return int(torch.argmin(reduced).item())


def work_plot_frame_indices(n_frames, every_n, l_wg_idx=None):
    """
    Frame indices for work_integral diagnostics.

    Plots every `every_n`-th frame, plus the first, last, and wall-gas frames
    so key states are always visible. Set every_n=1 to plot all frames.
    """
    every_n = max(1, int(every_n))
    indices = set(range(0, n_frames, every_n))
    indices.add(0)
    indices.add(n_frames - 1)
    if l_wg_idx is not None:
        indices.add(int(l_wg_idx))
    return sorted(indices)


def plot_work_integral_frame(
    x, omega, rho, baseline, i0, i1, tol_used, work_val, l_val,
    rho_v, rho_l, t_val, frame_idx, out_path,
):
    """
    Per-frame work integral diagnostic (notebook / Fig. 2 style):
    omega(x)-omega_b, rho(x), shaded trapz region for W, and a horizontal
    segment at 0.5*(rho_l-rho_v) tracking the film thickness l from the
    film-support edge x0 = x_min + FILM_X_OFFSET.
    """
    x_np = x.detach().cpu().numpy().squeeze()
    omega_np = omega.detach().cpu().numpy().squeeze()
    rho_np = rho.detach().cpu().numpy().squeeze()
    w_corr = omega_np - baseline
    x_ref = float(x_np.min() + FILM_X_OFFSET)
    y_lg = 0.5 * (rho_l - rho_v)

    plt.plot(x_np, w_corr, label=r"$\omega(x) - \omega_b$")
    plt.plot(x_np, rho_np, label=r"$\rho(x)$")
    if i1 >= i0:
        plt.fill_between(
            x_np[i0:i1 + 1], w_corr[i0:i1 + 1], 0.0, alpha=0.3,
            label=r"$W = \gamma_{wl} + \gamma_{lg} + \Delta\gamma$",
        )
    plt.hlines(
        y_lg, x_ref, x_ref + l_val,
        colors="green", linestyles="-", linewidth=1.5,
        label=rf"$l$-interface ($l={l_val:.2f}$)",
    )
    plt.axhline(0.0, ls="--", lw=1, color="0.4")
    plt.axhline(+tol_used, ls=":", lw=1, color="0.5")
    plt.axhline(-tol_used, ls=":", lw=1, color="0.5")

    x_hi = max(x_np[i1] + WORK_XLIM_PAD, x_ref + l_val + WORK_XLIM_PAD)
    plt.xlim(x_np.min(), x_hi)

    plt.xlabel("x")
    plt.title(
        f"T={t_val:.2f}, frame={frame_idx}, "
        f"$l={l_val:.3f}$, $W={work_val:.4f}$, "
        f"$\\omega_b={baseline:.3e}$"
    )
    plt.legend()
    plt.grid(False)
    plt.savefig(out_path, dpi=150)
    plt.close()


# ------------------------------------------------------------------------------
# Step 3 helpers: TorchScript spline regressors
# ------------------------------------------------------------------------------
@torch.jit.script
def _select1d(v: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # TorchScript-friendly 1D gather with arbitrary-shaped indices
    flat = torch.index_select(v, 0, idx.reshape(-1))
    return flat.reshape(idx.shape)


class LinearSpline(nn.Module):
    """
    Piecewise-linear spline y(x) through sorted 1D knots (x_i, y_i).
    Fit once from 1D tensors x[N], y[N]; evaluate on any shape of xq.
    End-interval slopes are used for extrapolation.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        super().__init__()
        x = x.detach().reshape(-1).to(torch.double)
        y = y.detach().reshape(-1).to(torch.double)

        idx = torch.argsort(x)
        x = x.index_select(0, idx)
        y = y.index_select(0, idx)
        if x.numel() < 2:
            raise ValueError("Need at least two points.")
        if torch.any(torch.diff(x) <= 0):
            raise ValueError("x must be strictly increasing.")

        h = torch.diff(x)
        s = torch.diff(y) / h

        self.register_buffer("xk", x)
        self.register_buffer("yk", y)
        self.register_buffer("sk", s)

    def _interval_indices(self, xq: torch.Tensor) -> torch.Tensor:
        i = torch.searchsorted(self.xk, xq, right=True) - 1
        i = torch.clamp(i, 0, self.xk.numel() - 2)
        return i

    def forward(self, xq: torch.Tensor) -> torch.Tensor:
        xq = xq.to(self.xk.dtype)
        i = self._interval_indices(xq)
        xi = _select1d(self.xk, i)
        yi = _select1d(self.yk, i)
        si = _select1d(self.sk, i)
        return yi + si * (xq - xi)

    @torch.jit.export
    def dy_dx(self, xq: torch.Tensor) -> torch.Tensor:
        xq = xq.to(self.xk.dtype)
        i = self._interval_indices(xq)
        return _select1d(self.sk, i)


class ModelWithDerivatives(nn.Module):
    """Thin wrapper exposing dy_dx via autograd; metadata attributes are
    attached after construction and captured by torch.jit.script."""

    def __init__(self, traced):
        super().__init__()
        self.fwd = traced

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.fwd(inp)

    @torch.jit.export
    def dy_dx(self, inp: torch.Tensor) -> torch.Tensor:
        x = inp.detach().clone().requires_grad_(True)
        y = self.fwd(x)
        grads = torch.autograd.grad([y.sum()], [x], create_graph=False, allow_unused=True)
        return torch.jit._unwrap_optional(grads[0])


def _fit_window(l, y):
    """Sort by l, restrict to (L_FIT_MIN, VALIDITY], drop duplicate knots."""
    l = l.reshape(-1).to(torch.double)
    y = y.reshape(-1).to(torch.double)

    order = torch.argsort(l)
    l, y = l[order], y[order]

    keep = (l > L_FIT_MIN) & (l <= VALIDITY)
    l, y = l[keep], y[keep]
    if l.numel() < 2:
        raise ValueError("Not enough points in the fit window.")

    # LinearSpline requires strictly increasing knots; continuation curves can
    # revisit the same l, so drop non-advancing knots.
    keep = torch.ones_like(l, dtype=torch.bool)
    keep[1:] = torch.diff(l) > 1e-9
    return l[keep][..., None], y[keep][..., None]


def fit_work_regressor(t_val, l, work, l_wg_idx, plot_path):
    x, y = _fit_window(l, work)

    spline = LinearSpline(x, y).eval()
    traced = torch.jit.trace(spline, torch.randn(1, 1, dtype=torch.double))
    wrapped = ModelWithDerivatives(traced).eval()

    # Landmarks on a dense grid inside the validity window
    grid = torch.linspace(0.0, VALIDITY, 1000, dtype=torch.double)[..., None]
    with torch.no_grad():
        w_grid = wrapped(grid)
    j = int(torch.argmax(w_grid))
    i = int(torch.argmin(w_grid[: j + 1]))
    if j == 0:
        # W(l) is monotone decreasing on [0, VALIDITY]: no thin-film barrier
        # (expected close to the critical temperature). The min/max landmarks
        # then degenerate to l=0; downstream code uses l_min only as the lower
        # integration bound, so this is safe but worth flagging.
        print(f"[W ] T={t_val:.2f}: WARNING -- no interior min/max of W(l) on "
              f"[0, {VALIDITY}]; landmarks degenerate to l_min=l_max=0.")

    wrapped.min = {"l_min": float(grid[i]), "W_min": float(w_grid[i])}
    wrapped.max = {"l_max": float(grid[j]), "W_max": float(w_grid[j])}
    # 0-d tensors: downstream code calls .item() on these
    wrapped.wg = {
        "l_wg": l.reshape(-1)[l_wg_idx].detach().to(torch.double),
        "W_wg": work.reshape(-1)[l_wg_idx].detach().to(torch.double),
    }
    wrapped.Winf = float(w_grid[-1])
    wrapped.validity = VALIDITY

    scripted = torch.jit.script(wrapped)
    out_path = os.path.join(W_REGRESSOR_DIR, f"work_regressor_{t_val:.2f}.pt")
    scripted.save(out_path)

    # Smoke test the saved artifact API used by omega_minimization/utils.py
    reloaded = torch.jit.load(out_path)
    reloaded(torch.tensor([[0.5]], dtype=torch.double))
    reloaded.dy_dx(torch.tensor([[0.5]], dtype=torch.double))
    _ = reloaded.min["W_min"], reloaded.wg["W_wg"].item(), reloaded.Winf

    plt.scatter(x.numpy(), y.numpy(), s=8, alpha=0.4, label="data")
    plt.plot(grid.numpy(), w_grid.numpy(), "r-", label="W(l) spline")
    plt.scatter(
        [wrapped.min["l_min"], wrapped.max["l_max"]],
        [wrapped.min["W_min"], wrapped.max["W_max"]],
        marker="X", color="k", zorder=3, label="min/max",
    )
    plt.xlabel("l"); plt.ylabel("W")
    plt.title(f"Work regressor, T={t_val:.2f}")
    plt.legend(); plt.grid()
    plt.savefig(plot_path); plt.close()

    print(f"[W ] T={t_val:.2f}: l_min={wrapped.min['l_min']:.3f}, "
          f"W_min={wrapped.min['W_min']:.4f}, l_max={wrapped.max['l_max']:.3f}, "
          f"Winf={wrapped.Winf:.4f} -> {out_path}")
    return wrapped


def fit_mu0_regressor(t_val, l, chem_pot, plot_path):
    x, y = _fit_window(l, chem_pot)

    spline = LinearSpline(x, y).eval()
    traced = torch.jit.trace(spline, torch.randn(1, 1, dtype=torch.double))
    wrapped = ModelWithDerivatives(traced).eval()
    wrapped.validity = VALIDITY

    scripted = torch.jit.script(wrapped)
    out_path = os.path.join(MU0_REGRESSOR_DIR, f"mu0_regressor_{t_val:.2f}.pt")
    scripted.save(out_path)

    reloaded = torch.jit.load(out_path)
    reloaded(torch.tensor([[0.5]], dtype=torch.double))
    reloaded.dy_dx(torch.tensor([[0.5]], dtype=torch.double))

    grid = torch.linspace(0.0, VALIDITY, 1000, dtype=torch.double)[..., None]
    with torch.no_grad():
        mu_grid = wrapped(grid)
    plt.scatter(x.numpy(), y.numpy(), s=8, alpha=0.4, label="data")
    plt.plot(grid.numpy(), mu_grid.numpy(), "r-", label="mu0(l) spline")
    plt.xlabel("l"); plt.ylabel("mu0")
    plt.title(f"mu0 regressor, T={t_val:.2f}")
    plt.legend(); plt.grid()
    plt.savefig(plot_path); plt.close()

    print(f"[mu0] T={t_val:.2f} -> {out_path}")
    return wrapped


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    for d in (
        W_REGRESSOR_DIR, MU0_REGRESSOR_DIR, PLOT_DIR,
        PLOT_SUMMARY_DIR, PLOT_WORK_DIR,
    ):
        os.makedirs(d, exist_ok=True)

    with open(ISOTHERM_PKL, "rb") as f:
        data_nn = pickle.load(f)
    print(f"Loaded {len(data_nn)} isotherm branches from {ISOTHERM_PKL}")

    with open(PHASE_CURVE_PKL, "rb") as f:
        pc_nn = pickle.load(f)
    print(f"Loaded phase curve with {len(pc_nn)} temperatures from {PHASE_CURVE_PKL}")

    t_list = sorted(
        t for (n, t) in data_nn.index
        if n == N_VALUE and float(t) in set(float(ti) for ti in pc_nn.index)
    )
    if not t_list:
        raise ValueError(
            f"No temperatures found for N={N_VALUE} that are also in the phase curve."
        )
    print(f"Processing N={N_VALUE}, T={t_list}")

    # --------------------------------------------------------------------------
    # Step 2: work of adsorption and film thickness per temperature
    # --------------------------------------------------------------------------
    output_rows = []
    for t_val in t_list:
        row = data_nn.loc[(N_VALUE, t_val)]
        x = torch.as_tensor(np.asarray(row["x"]).squeeze(), dtype=TORCH_DTYPE)
        dx = x[1] - x[0]

        stacked = stack_isotherm_frames(x, row["isotherm"][1:])
        n_frames = stacked["rho"].shape[0]

        # Bulk densities measured on the profiles: vapor from an early
        # (low-loading) frame, liquid from the last (high-loading) frame.
        rho_bv = stacked["rhoBv"][min(RHOB_VAPOR_FRAME, n_frames - 1)]
        rho_bl = stacked["rhoBl"][-1]

        rhoX, NX, l = film_thickness(x, stacked["rho"], rho_bv, rho_bl)
        l_wg_idx = find_wg_frame(rhoX)
        n_tot = stacked["rho"].sum(dim=-1) * dx  # [F, 1]

        print(f"T={t_val:.2f}: {n_frames} frames, rho_v={rho_bv:.4f}, "
              f"rho_l={rho_bl:.4f}, l_wg_idx={l_wg_idx} (l={l[l_wg_idx, 0]:.3f})")

        output_rows.append({
            "T": float(t_val),
            "x": x,
            "l": l,
            "l_wg_idx": l_wg_idx,
            "omega": stacked["omega"],
            "rho": stacked["rho"],
            "WOm": stacked["omega"].sum(dim=-1) * dx,
            "WOmX": stacked["WOmX"],
            "NX": NX,
            "N": n_tot,
            "rhoX": rhoX,
            "rho_v": float(np.abs(pc_nn.loc[t_val, "rho_v"])),
            "rho_l": float(np.abs(pc_nn.loc[t_val, "rho_l"])),
            "chem_pot": stacked["chem_pot"],
            "omega_baseline": stacked["omega_baseline"],
            "work_i0": stacked["work_i0"],
            "work_i1": stacked["work_i1"],
            "work_tol": stacked["work_tol"],
        })

        rho_v_t = float(np.abs(pc_nn.loc[t_val, "rho_v"]))
        rho_l_t = float(np.abs(pc_nn.loc[t_val, "rho_l"]))

        # --- diagnostic: work integral shaded area (Fig. 2), every N frames ---
        work_t_dir = os.path.join(PLOT_WORK_DIR, f"T{t_val:.2f}")
        os.makedirs(work_t_dir, exist_ok=True)
        l_flat = l.reshape(-1).detach().cpu().numpy()
        plot_frames = work_plot_frame_indices(n_frames, WORK_PLOT_EVERY_N, l_wg_idx)
        print(f"  work_integral: plotting {len(plot_frames)}/{n_frames} frames "
              f"(every {WORK_PLOT_EVERY_N})")
        for i_frame in plot_frames:
            plot_work_integral_frame(
                x,
                stacked["omega"][i_frame, 0, :],
                stacked["rho"][i_frame, 0, :],
                stacked["omega_baseline"][i_frame],
                int(stacked["work_i0"][i_frame]),
                int(stacked["work_i1"][i_frame]),
                float(stacked["work_tol"][i_frame]),
                float(stacked["WOmX"][i_frame]),
                float(l_flat[i_frame]),
                rho_v_t,
                rho_l_t,
                t_val,
                i_frame,
                out_path=os.path.join(
                    work_t_dir,
                    f"frame_{i_frame:04d}_l{l_flat[i_frame]:.3f}.png",
                ),
            )

        # --- summary: W vs l ---
        plt.plot(l.reshape(-1).numpy(), stacked["WOmX"].numpy(), lw=1)
        plt.scatter(l[l_wg_idx, 0], stacked["WOmX"][l_wg_idx], marker="x",
                    color="k", zorder=3, label="wall-gas frame")
        plt.xlabel("l"); plt.ylabel("W")
        plt.title(f"W vs l at T={t_val:.2f}")
        plt.legend(); plt.grid()
        plt.savefig(os.path.join(PLOT_SUMMARY_DIR, f"W_vs_l_T{t_val:.2f}.png"))
        plt.close()

    output_df = pd.DataFrame(output_rows).set_index("T")
    output_df.to_pickle(WORK_PKL)
    print(f"Saved isotherm work data to {WORK_PKL}")

    # --------------------------------------------------------------------------
    # Surface tensions diagnostic (final values come from the regressors via
    # upload_gammas_in_pc_ in the contact-angle step)
    # --------------------------------------------------------------------------
    gamma_lg, gamma_wl, gamma_wg = [], [], []
    for t_val in t_list:
        row = output_df.loc[t_val]
        w = row["WOmX"].reshape(-1)
        l_flat = row["l"].reshape(-1)

        # W_inf: median over the thick-film tail (robust replacement for the
        # notebook's hard-coded WOmX[-100])
        tail_mask = l_flat > 0.8 * VALIDITY
        w_inf = float(torch.median(w[tail_mask])) if tail_mask.any() \
            else float(torch.median(w[-max(1, w.numel() // 10):]))

        g_lg = float(pc_nn.loc[t_val, "gamma_lg"])
        gamma_lg.append(g_lg)
        gamma_wl.append(w_inf - g_lg)
        gamma_wg.append(float(w[row["l_wg_idx"]]))

    plt.plot(t_list, gamma_lg, "o-", label=r"$\gamma_{lg}$")
    plt.plot(t_list, gamma_wl, "o-", label=r"$\gamma_{wl}$")
    plt.plot(t_list, gamma_wg, "o-", label=r"$\gamma_{wg}$")
    plt.xlabel("T"); plt.legend(); plt.grid()
    plt.savefig(os.path.join(PLOT_SUMMARY_DIR, "gammas_vs_T.png")); plt.close()

    cos_theta = np.clip(
        (np.array(gamma_wg) - np.array(gamma_wl)) / np.array(gamma_lg), -1.0, 1.0
    )
    plt.plot(t_list, np.degrees(np.arccos(cos_theta)), "o-")
    plt.xlabel("T"); plt.ylabel(r"$\theta_Y$ (deg)")
    plt.title("Young contact angle (diagnostic)"); plt.grid()
    plt.savefig(os.path.join(PLOT_SUMMARY_DIR, "thetaY_vs_T.png")); plt.close()

    # --------------------------------------------------------------------------
    # Step 3: fit and save the W(l) and mu0(l) regressors
    # --------------------------------------------------------------------------
    for t_val in t_list:
        row = output_df.loc[t_val]
        fit_work_regressor(
            t_val,
            row["l"], row["WOmX"], row["l_wg_idx"],
            plot_path=os.path.join(PLOT_SUMMARY_DIR, f"W_fit_T{t_val:.2f}.png"),
        )
        fit_mu0_regressor(
            t_val,
            row["l"], row["chem_pot"],
            plot_path=os.path.join(PLOT_SUMMARY_DIR, f"mu0_fit_T{t_val:.2f}.png"),
        )

    print("Done.")


if __name__ == "__main__":
    main()
