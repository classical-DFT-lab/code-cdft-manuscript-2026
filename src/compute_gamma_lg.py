#!/usr/bin/env python3
"""
Compute liquid-gas surface tension gamma_lg from isotherm trajectories.

This script ports the workflow from src/isotherm_lg.ipynb into a standalone script.
It estimates 2*gamma_lg as the area under omega(x)-baseline over the interface and
stores gamma_lg = area/2 for each temperature. It also writes gamma_lg_cdf_vs_md_N*.png
comparing cDFT (T from index 2 onward) to MD reference data from isotherm_lg.ipynb.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from libs.io_utils import load_pickle
from libs.plot_utils import ensure_plot_dir
from libs.utils import resolve_training_device, tensors_to_cpu_for_storage
from libs.cdft_1d.augmented_lda import CDFT_MODEL as CDFT
from libs.cdft_1d.external_potentials import LJ93
from libs.solve_1d.newton import newton
from libs.solve_1d.continuation_gpu import continuation
from libs.ml.surrogates import setDNN, setWDA, setDNNRep, load_ml_state_dicts
from libs.ml.loss import LossL1

# ------------------------------------------------------------------------------
# Fixed configuration (no CLI args)
# ------------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ISOTHERM_PKL = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "data", "dataset", "pkl", "profiles_lg", "isotherm_profiles_lg_lg.pkl")
)
PHASE_CURVE_PKL = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "data", "dataset", "pkl", "bulk_tmd", "pc_nn_operative.pkl")
)
N_VALUE = 8000
TEMPERATURES = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,0.95, 1.0]
FRAME_MODE = "max"  # "max" or "last"
TAIL_FRAC = 0.15
NMIN = 10
PLOT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "output", "plot_lg_isotherm", "gamma_lg"))
OUT_PKL = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "dataset", "pkl", "bulk_tmd", "pc_md_operative_with_gamma.pkl"))
POPULATED_ISOTHERM_PKL = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "output", "gamma_lg", "isotherm_profiles_lg_populated.pkl")
)

# Solver/model settings for isotherm population
DEVICE_KIND = "auto"
TORCH_DTYPE = torch.float32
RESTART_ML_MODEL = 1
USE_MODEL = 1
USE_DBH_DIAMETER = 0
TRAIN_DNN = 0
TRAIN_WDA = 0
SAVE_MODEL = 0
JACOBIAN = "STABLE"
FAST = False
WALL = "lg"

newton_max_steps = 1600
newton_tol = 1e-7
newton_alpha = 0.9
newton_verbose = 0
cont_ds = 0.05
cont_continuation_steps = 600
cont_max_corrector_steps = 500
cont_alpha = 0.9

R = 1.0
sigma_attr = R
eps_attr = 1.0
cutoff_attr = 2.5 * R
if WALL == "lg":
    Ew, sigmaw = 0.0, 1.0
elif WALL == "wc":
    Ew, sigmaw = 1.0, 2.0
elif WALL == "wn2":
    Ew, sigmaw = 1.2, 1.2
    
cutoff_wall = 10.0 * R
BC_R = "NONE"
ensemble = "NVT"
dft_type = "LDA"
str_param = "N"
Lambda = 1.0

# MD reference surface tension (same table as isotherm_lg.ipynb)
GAMMA_T_MD = [
    (0.65, 0.670),
    (0.7, 0.584),
    (0.75, 0.486),
    (0.8, 0.400),
    (0.85, 0.312),
    (0.9, 0.229),
    (0.95, 0.152),
    (1.0, 0.075),
    (1.05, 0.025),
]


def plot_gamma_lg_cdf_vs_md(gamma_series: list[tuple[float, float]], plot_dir: str, n_key: int) -> None:
    """
    Compare cDFT gamma_lg vs MD reference (isotherm_lg.ipynb).
    Uses T_list[2:], gamma_lg[2:] for cDFT when enough points exist.
    """
    if not gamma_series:
        return
    gamma_series = sorted(gamma_series, key=lambda p: p[0])
    t_list = [float(t) for t, _ in gamma_series]
    gamma_lg = [float(g) for _, g in gamma_series]
    t_cdf = t_list[2:] if len(t_list) >= 3 else t_list
    g_cdf = gamma_lg[2:] if len(gamma_lg) >= 3 else gamma_lg

    t_md = [GAMMA_T_MD[i][0] for i in range(len(GAMMA_T_MD) - 1)]
    gamma_md = [GAMMA_T_MD[i][1] for i in range(len(GAMMA_T_MD) - 1)]

    plt.figure(figsize=(6, 4))
    plt.plot(
        t_cdf,
        g_cdf,
        marker="o",
        color="red",
        label=r"$\gamma_{lg}$ • cDFT (this work)",
    )
    plt.plot(
        t_md,
        gamma_md,
        marker="x",
        linestyle="--",
        color="black",
        label=r"$\gamma_{lg}$ • MD",
    )
    plt.legend()
    plt.xlabel(r"$T$")
    plt.ylabel(r"$\gamma_{lg}$")
    plt.grid(True)
    out = os.path.join(plot_dir, f"gamma_lg_cdf_vs_md_N{n_key}.png")
    ensure_plot_dir(out)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot of gamma_lg vs MD reference to: {os.path.abspath(out)}")



def area_under_interface(
    x: np.ndarray,
    omega: np.ndarray,
    tail_frac: float = 0.15,
    nmin: int = 20,
    tol: float | None = None,
    absolute: bool = False,
):
    """
    Estimate interface area under (omega - baseline) using notebook logic.

    Returns:
        area: integral value over detected interface support.
        baseline: average of median left/right tail plateaus.
        bounds: tuple (i0, i1) integration bounds (inclusive).
        tol_used: threshold used to detect support.
    """
    x = np.asarray(x).squeeze()
    w = np.asarray(omega).squeeze()
    if not (x.ndim == 1 and w.ndim == 1 and x.size == w.size):
        raise ValueError(f"x and omega must be 1D with same size, got {x.shape=} and {w.shape=}")

    npts = x.size
    k_tail = max(nmin, int(tail_frac * npts))
    left = w[:k_tail]
    right = w[-k_tail:]

    base_l = float(np.median(left))
    base_r = float(np.median(right))
    baseline = 0.5 * (base_l + base_r)
    w_corr = w - baseline

    if tol is None:
        noise = np.concatenate([left - base_l, right - base_r])
        tol = float(1.1 * np.std(noise) + 1e-12)

    idx = np.where(np.abs(w_corr[k_tail:]) > tol)[0]
    if idx.size == 0:
        return 0.0, baseline, (None, None), tol

    # Keep the same support-selection heuristic used in the notebook.
    i0 = int(idx[0])
    i1 = int(npts - idx[0])
    y = np.abs(w_corr[i0 : i1 + 1]) if absolute else w_corr[i0 : i1 + 1]
    area = float(np.trapezoid(y, x[i0 : i1 + 1]))
    return area, baseline, (i0, i1), tol


def _to_tensor_1d(x, device, dtype):
    return torch.as_tensor(np.asarray(x).squeeze(), device=device, dtype=dtype)


def build_cdft_context(x_t: torch.Tensor, rho_init_t: torch.Tensor, t_val: float, device, ml_state_dicts):
    dx = x_t[1] - x_t[0]
    bs = 1
    l_box = 0.5 * (x_t.max().item() - x_t.min().item())
    x_wall = x_t.min() - 0.001 * sigmaw
    vext = (
        (LJ93(x_t, x_wall, Ew, sigmaw) - LJ93(cutoff_wall * torch.ones_like(x_t), x_wall, Ew, sigmaw))
        * torch.sigmoid(-(x_t - x_wall - cutoff_wall) / 0.001)
        * 0.0
    )[None, ...]

    n_start = rho_init_t.sum() * dx
    eq_params = {
        "R": torch.tensor(R, dtype=TORCH_DTYPE, device=device),
        "mu": n_start.view(1, 1),
        "beta": torch.tensor(1.0 / t_val, dtype=TORCH_DTYPE, device=device).view(1, 1),
        "Lambda": torch.tensor(Lambda, dtype=TORCH_DTYPE, device=device),
        "dft_type": dft_type,
        "ensemble": ensemble,
        "str_param": str_param,
        "sigma_attr": torch.tensor(sigma_attr, dtype=TORCH_DTYPE, device=device),
        "eps_attr": torch.tensor(eps_attr, dtype=TORCH_DTYPE, device=device),
        "cutoff_attr": torch.tensor(cutoff_attr, dtype=TORCH_DTYPE, device=device),
        "Ew": torch.tensor(Ew, dtype=TORCH_DTYPE, device=device),
        "sigmaw": torch.tensor(sigmaw, dtype=TORCH_DTYPE, device=device),
        "Vext": vext,
        "BC_R": BC_R,
        "fast": FAST,
        "safe": 1.0,
        "target_density": n_start / (2.0 * l_box),
    }
    mesh = {
        "BS": bs,
        "L": l_box,
        "Nx": x_t.numel(),
        "x_bc": [x_t.min().item(), x_t.max().item()],
        "x": x_t,
        "x_wall": torch.tensor(x_wall, dtype=TORCH_DTYPE, device=device),
        "dx": dx,
    }
    sol = {
        "rho_guess": rho_init_t.view(1, 1, -1),
        "device": device,
        "dtype": TORCH_DTYPE,
        "outdir": os.path.normpath(os.path.join(SCRIPT_DIR, "..", "output")) + os.sep,
        "datadir": os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data")) + os.sep,
        "pkldir": os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data", "dataset", "pkl", "profiles_lg")) + os.sep,
        "JACOBIAN": JACOBIAN,
        "RESTART_ML_MODEL": RESTART_ML_MODEL,
        "SAVE_MODEL": SAVE_MODEL,
        "USE_MODEL": USE_MODEL,
        "USE_DBH_DIAMETER": USE_DBH_DIAMETER,
        "TRAIN_DNN": TRAIN_DNN,
        "TRAIN_WDA": TRAIN_WDA,
        "LOSS": LossL1,
        "rho_vl": torch.tensor([0.0, 0.0], dtype=TORCH_DTYPE, device=device),
    }
    model = CDFT(eq_params, mesh, sol)
    setDNN(model, LR=0.0, state_dicts=ml_state_dicts)
    setDNNRep(model, LR=0.0, state_dict=ml_state_dicts["dnn_rep_fn"] if ml_state_dicts else None)
    setWDA(model, LR=0.0, modes=150, state_dict=ml_state_dicts["wda_fn"] if ml_state_dicts else None)
    if hasattr(model, "dnn_fn") and model.dnn_fn is not None:
        model.dnn_fn.eval()
        for p in model.dnn_fn.parameters():
            p.requires_grad = False
    if hasattr(model, "dnn_rep_fn") and model.dnn_rep_fn is not None:
        model.dnn_rep_fn.eval()
        for p in model.dnn_rep_fn.parameters():
            p.requires_grad = False
    if hasattr(model, "wda_fn") and model.wda_fn is not None:
        model.wda_fn.eval()
        for p in model.wda_fn.parameters():
            p.requires_grad = False
    return model


def _normalize_isotherm_frames(frames):
    normalized = []
    for frame in frames[1:]:
        omega_val = frame.get("omega", frame.get("omegaX"))
        if omega_val is None:
            continue
        normalized.append(
            {
                "rho": np.asarray(frame["rho"]).squeeze(),
                "mu": frame.get("mu"),
                "omega": np.asarray(omega_val).squeeze(),
                "chem_pot": frame.get("chem_pot"),
            }
        )
    return normalized


def populate_missing_isotherm(isotherm_df, n_key, t_list, plot_dir, device, ml_state_dicts):
    updated = False
    for t_i in t_list:
        row = isotherm_df.loc[(n_key, t_i)]
        frames = row.get("isotherm", None)
        if frames is not None and len(frames) > 0:
            continue

        print(f"[POPULATE] Building isotherm for N={n_key}, T={t_i} ...")
        x_t = _to_tensor_1d(row["x"], device=device, dtype=TORCH_DTYPE)
        rho_init_t = _to_tensor_1d(row["rho"], device=device, dtype=TORCH_DTYPE)
        model = build_cdft_context(x_t=x_t, rho_init_t=rho_init_t, t_val=float(t_i), device=device, ml_state_dicts=ml_state_dicts)

        out_fwd = newton(
            model=model,
            U_guess=model.sol["rho_guess"],
            max_steps=newton_max_steps,
            tol=newton_tol,
            alpha=newton_alpha,
            verbose=newton_verbose,
            detach_tensors=True,
        )
        sol_0 = {
            "mu": (out_fwd["U"].sum() * model.mesh["dx"]).view(1, 1),
            "rho": out_fwd["U"],
            "rhocoex_vl": torch.tensor([0.0, 0.0], dtype=TORCH_DTYPE, device=device),
        }
        isotherm_curve = continuation(
            sol_0,
            model,
            ds=cont_ds,
            continuation_steps=cont_continuation_steps,
            max_corrector_steps=cont_max_corrector_steps,
            alpha=cont_alpha,
            plotdir=plot_dir,
            detach_tensors=True,
        )
        normalized_frames = _normalize_isotherm_frames(isotherm_curve)
        isotherm_df.at[(n_key, t_i), "isotherm"] = normalized_frames
        updated = True
        print(f"[POPULATE] Added {len(normalized_frames)} frames for T={t_i}.")
    return updated


def main():
    device = resolve_training_device(DEVICE_KIND)
    ml_state_dicts = (
        load_ml_state_dicts(os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data")) + os.sep, device)
        if RESTART_ML_MODEL
        else None
    )

    isotherm_df = load_pickle(
        ISOTHERM_PKL,
        description="LG isotherm trajectories",
        hint="Run lg continuation workflow first to generate profile isotherms.",
    )
    if not isinstance(isotherm_df, pd.DataFrame):
        raise TypeError("isotherm-pkl must contain a pandas DataFrame.")
    if "isotherm" not in isotherm_df.columns:
        isotherm_df["isotherm"] = None

    n_key = N_VALUE
    t_list = TEMPERATURES
    plot_dir = ensure_plot_dir(PLOT_DIR)

    updated = populate_missing_isotherm(
        isotherm_df=isotherm_df,
        n_key=n_key,
        t_list=t_list,
        plot_dir=plot_dir,
        device=device,
        ml_state_dicts=ml_state_dicts,
    )
    if updated:
        populated_path = ensure_plot_dir(POPULATED_ISOTHERM_PKL)
        tensors_to_cpu_for_storage(isotherm_df).to_pickle(populated_path)
        print(f"Saved populated isotherm dataframe to: {os.path.abspath(populated_path)}")
        # Also update the source isotherm pickle in-place so future runs reuse populated frames.
        source_path = ensure_plot_dir(ISOTHERM_PKL)
        tensors_to_cpu_for_storage(isotherm_df).to_pickle(source_path)
        print(f"Updated source isotherm dataframe with populated frames: {os.path.abspath(source_path)}")

    pc_df = load_pickle(
        PHASE_CURVE_PKL,
        description="Phase curve dataframe",
        hint="Expected ../data/dataset/postproc/phase_curve_nn.pkl",
    )
    if "gamma_lg" not in pc_df.columns:
        pc_df["gamma_lg"] = np.nan
    if "rho_l" not in pc_df.columns:
        pc_df["rho_l"] = np.nan
    if "rho_v" not in pc_df.columns:
        pc_df["rho_v"] = np.nan

    gamma_series = []
    for t_i in t_list:
        row = isotherm_df.loc[(n_key, t_i)]
        x = np.asarray(row["x"]).squeeze()
        frames = row["isotherm"]
        if not frames:
            print(f"[WARN] Empty isotherm list for N={n_key}, T={t_i}. Skipping.")
            continue

        frame_indices = [len(frames) - 1] if FRAME_MODE == "last" else range(1, len(frames))
        gamma_frame = []
        final_payload = None
        final_diagnostics = None

        for frame_idx in frame_indices:
            payload = frames[frame_idx]
            omega = np.asarray(payload.get("omega", payload.get("omegaX"))).squeeze()
            rho = np.asarray(payload["rho"]).squeeze()
            area, baseline, (i0, i1), tol = area_under_interface(
                x=x, omega=omega, tail_frac=TAIL_FRAC, nmin=NMIN, tol=None
            )
            gamma_i = area / 2.0
            gamma_frame.append(gamma_i)
            final_payload = payload
            final_diagnostics = (baseline, i0, i1, tol, omega, rho)

        if not gamma_frame:
            print(f"[WARN] No usable frames for N={n_key}, T={t_i}. Skipping.")
            continue

        gamma_selected = float(max(gamma_frame) if FRAME_MODE == "max" else gamma_frame[-1])
        gamma_series.append((t_i, gamma_selected))

        baseline, i0, i1, tol, omega_last, rho_last = final_diagnostics
        pc_df.loc[t_i, "rho_l"] = float(rho_last[rho_last.shape[-1] // 2])
        pc_df.loc[t_i, "rho_v"] = float(rho_last[int(rho_last.shape[-1] * 0.10)])
        pc_df.loc[t_i, "gamma_lg"] = gamma_selected

        # Diagnostic plot like notebook: omega-baseline and rho profile.
        plt.figure(figsize=(7, 4))
        w_corr = omega_last - baseline
        plt.plot(x, w_corr, label=r"$\omega(x)-\omega_b$")
        plt.plot(x, rho_last, label=r"$\rho(x)$")
        if i0 is not None and i1 is not None:
            plt.fill_between(x[i0 : i1 + 1], w_corr[i0 : i1 + 1], 0.0, alpha=0.25, label=r"$2\gamma_{lg}$")
        plt.axhline(0.0, ls="--", lw=1)
        plt.title(f"T={float(t_i):.2f}  gamma_lg={gamma_selected:.6f}")
        plt.grid(True)
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("omega / rho")
        fig_path = os.path.join(plot_dir, f"gamma_profile_N{n_key}_T{float(t_i):.2f}.png")
        ensure_plot_dir(fig_path)
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(
            f"T={float(t_i):.2f} gamma_lg={gamma_selected:.6f} "
            f"(baseline={baseline:.3e}, window=({i0},{i1}), tol={tol:.2e}, frames={len(gamma_frame)})"
        )

    if gamma_series:
        gamma_series = sorted(gamma_series, key=lambda p: p[0])
        plt.figure(figsize=(6, 4))
        plt.plot([t for t, _ in gamma_series], [g for _, g in gamma_series], marker="o")
        plt.grid(True)
        plt.xlabel("T")
        plt.ylabel(r"$\gamma_{lg}$")
        curve_path = os.path.join(plot_dir, f"gamma_curve_N{n_key}.png")
        ensure_plot_dir(curve_path)
        plt.savefig(curve_path, dpi=150, bbox_inches="tight")
        plt.close()

        plot_gamma_lg_cdf_vs_md(gamma_series, plot_dir, n_key)

    out_path = ensure_plot_dir(OUT_PKL)
    pc_df.sort_index().to_pickle(out_path)
    print(f"Saved updated phase curve with gamma_lg to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
