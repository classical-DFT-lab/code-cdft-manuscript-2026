#!/usr/bin/env python3
"""
Data preprocessing pipeline for adj_train and related scripts.

Generates z_profiles.pkl and z_profiles_0.pkl from:
  1. z_profile.csv files in N*/T_* directory tree (MD data -> z_profiles.pkl)
  2. LDA reference profiles (z_profiles_0): inline DFT solve or pre-computed LDA tree

Usage:
  python preprocess_data.py [--source SOURCE] [--output OUTPUT] [--wall WALL]
  e.g., python preprocess_data.py --skip-lda --wall wc --source ../data/dataset/lda_planar_wc
  --source: MD root (default) or LDA root when --skip-lda is set
  --md-source: MD root when --skip-lda (default: ../data/dataset/md_planar_<wall>)
  --output: Output dir for z_profiles*.pkl (default: ../data/dataset/pkl/profiles_wl_wc)
  --wall:   "wn2" or "wc" - sets output path if --output not given
"""
import os
import sys
import argparse
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import numpy as np
import pandas as pd
import torch

from libs.cdft_1d.augmented_lda import CDFT_MODEL as CDFT
from libs.cdft_1d.external_potentials import LJ93
from libs.solve_1d.newton import newton
from libs.ml.surrogates import setDNN, setDNNRep, setWDA
from libs.ml.loss import LossL1
from libs.utils import (
    dataframe_elementwise_map,
    resolve_training_device,
    sol_dtype,
    tensors_to_cpu_for_storage,
)


def gather_profiles(root: Path) -> pd.DataFrame:
    """Gather z_profile.csv files from N*/T_* tree. Returns DataFrame with MultiIndex (N, T)."""
    frames = []
    keys = []

    for N_dir in sorted(root.glob("N*/")):
        try:
            N = int(N_dir.name.lstrip("N"))
        except ValueError:
            continue
        for T_dir in sorted(N_dir.glob("T_*/")):
            try:
                T = float(T_dir.name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            csv = T_dir / "z_profile.csv"
            if csv.is_file():
                df = pd.read_csv(csv)
                frames.append(df)
                keys.append((N, T))

    if not frames:
        raise FileNotFoundError(
            f"No z_profile.csv files found under {root}\n"
            f"Expected structure: {root}/N8000/T_0.55/z_profile.csv"
        )

    combined = pd.concat(frames, keys=keys, names=["N", "T"])
    grouped = combined.groupby(level=["N", "T"]).agg(list)
    df_grouped = dataframe_elementwise_map(grouped, np.array)

    # Map columns: psi->x, rho_mean->rho
    col_map = {"psi": "x"}
    if "rho_mean" in df_grouped.columns:
        col_map["rho_mean"] = "rho"
    df = df_grouped.rename(columns=col_map)

    if "rho" not in df.columns and "rho_mean" in df.columns:
        df["rho"] = df["rho_mean"]
    if "rho_fit" not in df.columns:
        df["rho_fit"] = df["rho"]

    return df[["x", "rho", "rho_fit"]]


def default_dataset_root(name: str) -> Path:
    return (Path(script_dir) / ".." / "data" / "dataset" / name).resolve()


def build_z_profiles_0(df_md, mesh, eq_params, sol, device, Ew=1.2, sigmaw=1.2):
    """
    Compute LDA reference profiles (z_profiles_0) by solving DFT for each (N, T).
    Uses USE_MODEL=False for pure LDA.
    """
    L = mesh["L"]
    x = mesh["x"]
    dx = mesh["dx"]
    x_wall = mesh["x_wall"].item() if torch.is_tensor(mesh["x_wall"]) else mesh["x_wall"]
    cutoff_wall = 10.0

    Vext = (
        (LJ93(x, x_wall, Ew, sigmaw) - LJ93(cutoff_wall * torch.ones_like(x), x_wall, Ew, sigmaw))
        * torch.sigmoid(-(x - x_wall - cutoff_wall) / 1e-3)
    )[None, ...].to(device)

    records = []
    dt = sol_dtype(sol)
    for (N, T), row in df_md.iterrows():
        x_arr = np.array(row["x"]).flatten()
        rho_init = np.array(row["rho"]).flatten()
        # Initial guess: interpolate MD profile or use constant
        rho_guess = torch.tensor(rho_init, dtype=dt, device=device)
        if len(rho_guess.shape) == 1:
            rho_guess = rho_guess[None, None, :]
        elif len(rho_guess.shape) == 2:
            rho_guess = rho_guess[None, :, :] if rho_guess.shape[0] < rho_guess.shape[1] else rho_guess[:, None, :]

        # Interpolate to mesh if needed
        if rho_guess.shape[-1] != mesh["Nx"]:
            from libs.utils import linear_interpolate_1d_unsorted

            x_src = torch.tensor(x_arr, dtype=dt, device=device)
            rho_src = torch.tensor(rho_init, dtype=dt, device=device)
            rho_guess = linear_interpolate_1d_unsorted(x, x_src, rho_src)[None, None, :]

        beta = 1.0 / T
        mu_val = N / (2 * 40) ** 2  # approximate for NVT
        eq_params_local = dict(eq_params)
        eq_params_local["beta"] = torch.tensor(beta, dtype=dt, device=device)[None, None]
        eq_params_local["mu"] = torch.tensor(mu_val, dtype=dt, device=device)[None, None]
        eq_params_local["Vext"] = Vext

        sol_local = dict(sol)
        sol_local["USE_MODEL"] = False
        sol_local["rho_guess"] = rho_guess

        model = CDFT(eq_params_local, mesh, sol_local)
        setDNN(model, LR=0.0)
        setDNNRep(model, LR=0.0)
        setWDA(model, LR=0.0, modes=150)
        model.dnn_fn.eval()
        model.dnn_rep_fn.eval()
        model.wda_fn.eval()

        try:
            out = newton(
                model, U_guess=rho_guess,
                max_steps=800, tol=1e-6, alpha=0.5, verbose=False, detach_tensors=True
            )
            rho_sol = out["U"][0, 0, :].cpu().numpy()
            x_sol = x.cpu().numpy()
        except Exception as e:
            print(f"  Warning: LDA solve failed for N={N}, T={T}: {e}")
            rho_sol = np.array(row["rho"]).flatten()  # fallback to MD
            x_sol = np.array(row["x"]).flatten()

        records.append({"N": N, "T": T, "x": x_sol, "rho": rho_sol})

    df_0 = pd.DataFrame(records).set_index(["N", "T"])
    return df_0


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MD profiles and generate z_profiles.pkl, z_profiles_0.pkl"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="MD root with N*/T_*/z_profile.csv (default: md_planar_<wall>); "
        "with --skip-lda: LDA reference root (default: lda_planar_<wall>)",
    )
    parser.add_argument(
        "--md-source",
        type=Path,
        default=None,
        help="MD root when using --skip-lda (default: md_planar_<wall>)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output dir for z_profiles*.pkl (default: ../data/dataset/pkl/profiles_wl_<wall>)",
    )
    parser.add_argument(
        "--wall",
        choices=["wn2", "wc"],
        default="wc",
        help="Wall type for default output path",
    )
    parser.add_argument(
        "--skip-lda",
        action="store_true",
        help="Skip inline LDA solve; load pre-computed LDA reference via --source",
    )
    args = parser.parse_args()

    md_source = (
        args.md_source.expanduser().resolve()
        if args.md_source is not None
        else default_dataset_root(f"md_planar_{args.wall}")
    )
    if args.skip_lda:
        lda_source = (
            args.source.expanduser().resolve()
            if args.source is not None
            else default_dataset_root(f"lda_planar_{args.wall}")
        )
    else:
        md_source = (
            args.source.expanduser().resolve()
            if args.source is not None
            else default_dataset_root(f"md_planar_{args.wall}")
        )

    if args.output is not None:
        output = args.output.expanduser().resolve()
    else:
        output = (Path(script_dir) / ".." / "data" / "dataset" / "pkl" / f"profiles_wl_{args.wall}").resolve()

    output.mkdir(parents=True, exist_ok=True)
    DEVICE_KIND = "auto"  # or "cuda" | "mps" | "cpu"; override with env TORCH_DEVICE
    device = resolve_training_device(DEVICE_KIND)
    print(f"Using device: {device}")
    # Tensor dtype for DFT tensors (torch.float64 or torch.float32)
    TORCH_DTYPE = torch.float32
    torch.manual_seed(42)

    # 1. Gather z_profiles.pkl from MD z_profile.csv tree
    print(f"Scanning {md_source} for MD z_profile.csv ...")
    df_md = gather_profiles(md_source)
    z_profiles_path = output / "z_profiles.pkl"
    tensors_to_cpu_for_storage(df_md).to_pickle(z_profiles_path)
    Ns = df_md.index.get_level_values("N").unique()
    print(f"Wrote {z_profiles_path} with {len(df_md)} profiles")
    for N in Ns:
        Ts = df_md.loc[N].index.get_level_values("T").unique()
        print(f"  N={N}: T = [{', '.join(f'{t:.2f}' for t in Ts)}]")

    # 2. Build z_profiles_0.pkl (LDA reference)
    if args.skip_lda:
        print(f"Skipping LDA solve (--skip-lda); loading reference from {lda_source} ...")
        df_0 = gather_profiles(lda_source)[["x", "rho"]].copy()
    else:
        print("Computing LDA reference profiles (z_profiles_0)...")
        R, Lambda = 1.0, 1.0
        L = 30.0
        xmin, xmax = 0, 2*L
        Nx = int(2 * L / 0.2)
        x = torch.linspace(xmin, xmax, Nx, dtype=TORCH_DTYPE).to(device)
        dx = x[1] - x[0]
        Ew, sigmaw = (1.2, 1.2) if args.wall == "wn2" else (1.0, 2.0)
        x_wall = xmin - 0.001 * sigmaw

        mesh = {
            "BS": 1,
            "L": L,
            "Nx": Nx,
            "x_bc": [xmin, xmax],
            "x": x,
            "x_wall": torch.tensor(x_wall, dtype=TORCH_DTYPE, device=device),
            "dx": dx.to(device),
        }
        eq_params = {
            "R": torch.tensor(R, dtype=TORCH_DTYPE, device=device),
            "mu": torch.tensor(0.3, dtype=TORCH_DTYPE, device=device)[None, None],
            "beta": torch.tensor(1 / 0.95, dtype=TORCH_DTYPE, device=device)[None, None],
            "Lambda": torch.tensor(Lambda, dtype=TORCH_DTYPE, device=device),
            "dft_type": "LDA",
            "ensemble": "NVT",
            "str_param": "N",
            "sigma_attr": torch.tensor(1.0, dtype=TORCH_DTYPE, device=device),
            "eps_attr": torch.tensor(1.0, dtype=TORCH_DTYPE, device=device),
            "cutoff_attr": torch.tensor(2.5, dtype=TORCH_DTYPE, device=device),
            "Ew": torch.tensor(Ew, dtype=TORCH_DTYPE, device=device),
            "sigmaw": torch.tensor(sigmaw, dtype=TORCH_DTYPE, device=device),
            "Vext": None,
            "BC_R": "NONE",
            "fast": False,
        }
        sol = {
            "rho_guess": torch.zeros((1, 1, Nx), dtype=TORCH_DTYPE, device=device),
            "device": device,
            "dtype": TORCH_DTYPE,
            "outdir": "",
            "datadir": "",
            "pkldir": "",
            "RESTART_ML_MODEL": False,
            "USE_MODEL": False,
            "JACOBIAN": "EXACT",
            "LOSS": LossL1,
        }
        df_0 = build_z_profiles_0(df_md, mesh, eq_params, sol, device, Ew, sigmaw)

    z_profiles_0_path = output / "z_profiles_0.pkl"
    tensors_to_cpu_for_storage(df_0).to_pickle(z_profiles_0_path)
    print(f"Wrote {z_profiles_0_path}")

    print(f"\nDone. Run adj_train.py (pkldir={output})")


if __name__ == "__main__":
    main()
