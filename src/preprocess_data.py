#!/usr/bin/env python3
"""
Data preprocessing pipeline for adj_train and related scripts.

Generates z_profiles.pkl and z_profiles_0.pkl from:
  1. z_profile.csv files in N*/T_* directory tree (MD data)
  2. LDA DFT solutions for reference profiles (z_profiles_0)

Usage:
  python preprocess_data.py [--source SOURCE] [--output OUTPUT] [--wall WALL]
  e.g., python preprocess_data.py --skip-lda --wall wc
  --source: Root dir with N*/T_*/z_profile.csv (default: ../data/dataset/md_planar_wc)
  --output: Output dir for z_profiles*.pkl (default: ../data/dataset/pkl/profiles_wl_wn2)
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
    df_grouped = grouped.map(np.array)

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
    for (N, T), row in df_md.iterrows():
        x_arr = np.array(row["x"]).flatten()
        rho_init = np.array(row["rho"]).flatten()
        # Initial guess: interpolate MD profile or use constant
        rho_guess = torch.tensor(rho_init, dtype=torch.double, device=device)
        if len(rho_guess.shape) == 1:
            rho_guess = rho_guess[None, None, :]
        elif len(rho_guess.shape) == 2:
            rho_guess = rho_guess[None, :, :] if rho_guess.shape[0] < rho_guess.shape[1] else rho_guess[:, None, :]

        # Interpolate to mesh if needed
        if rho_guess.shape[-1] != mesh["Nx"]:
            from libs.utils import linear_interpolation
            x_src = torch.tensor(x_arr, dtype=torch.double, device=device)
            rho_src = torch.tensor(rho_init, dtype=torch.double, device=device)
            rho_guess = linear_interpolation(x, x_src, rho_src)[None, None, :]

        beta = 1.0 / T
        mu_val = N / (2 * L) ** 2  # approximate for NVT
        eq_params_local = dict(eq_params)
        eq_params_local["beta"] = torch.tensor(beta, dtype=torch.double, device=device)[None, None]
        eq_params_local["mu"] = torch.tensor(mu_val, dtype=torch.double, device=device)[None, None]
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
        default=Path(script_dir) / ".." / "data" / "dataset" / "md_planar_wc",
        help="Root dir with N*/T_*/z_profile.csv",
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
        help="Skip LDA solve; use MD rho as rho_0 (faster, less accurate)",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    if args.output is not None:
        output = args.output.expanduser().resolve()
    else:
        output = (Path(script_dir) / ".." / "data" / "dataset" / "pkl" / f"profiles_wl_{args.wall}").resolve()

    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # 1. Gather z_profiles.pkl from z_profile.csv
    print(f"Scanning {source} for z_profile.csv ...")
    df_md = gather_profiles(source)
    z_profiles_path = output / "z_profiles.pkl"
    df_md.to_pickle(z_profiles_path)
    Ns = df_md.index.get_level_values("N").unique()
    print(f"Wrote {z_profiles_path} with {len(df_md)} profiles")
    for N in Ns:
        Ts = df_md.loc[N].index.get_level_values("T").unique()
        print(f"  N={N}: T = [{', '.join(f'{t:.2f}' for t in Ts)}]")

    # 2. Build z_profiles_0.pkl (LDA reference)
    if args.skip_lda:
        print("Skipping LDA solve (--skip-lda); using MD rho as reference.")
        df_0 = df_md[["x", "rho"]].copy()
    else:
        print("Computing LDA reference profiles (z_profiles_0)...")
        R, Lambda = 1.0, 1.0
        L = 30.0
        xmin, xmax = -L, L
        Nx = int(2 * L / 0.2)
        x = torch.linspace(xmin, xmax, Nx, dtype=torch.double).to(device)
        dx = x[1] - x[0]
        Ew, sigmaw = (1.2, 1.2) if args.wall == "wn2" else (1.0, 2.0)
        x_wall = xmin - 0.001 * sigmaw

        mesh = {
            "BS": 1,
            "L": L,
            "Nx": Nx,
            "x_bc": [xmin, xmax],
            "x": x,
            "x_wall": torch.tensor(x_wall, device=device),
            "dx": dx.to(device),
        }
        eq_params = {
            "R": torch.tensor(R, device=device),
            "mu": torch.tensor(0.3, device=device)[None, None],
            "beta": torch.tensor(1 / 0.95, device=device)[None, None],
            "Lambda": torch.tensor(Lambda, device=device),
            "dft_type": "LDA",
            "ensemble": "NVT",
            "str_param": "N",
            "sigma_attr": torch.tensor(1.0, device=device),
            "eps_attr": torch.tensor(1.0, device=device),
            "cutoff_attr": torch.tensor(2.5, device=device),
            "Ew": torch.tensor(Ew, device=device),
            "sigmaw": torch.tensor(sigmaw, device=device),
            "Vext": None,
            "BC_R": "NONE",
            "fast": False,
        }
        sol = {
            "rho_guess": torch.zeros((1, 1, Nx), dtype=torch.double, device=device),
            "device": device,
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
    df_0.to_pickle(z_profiles_0_path)
    print(f"Wrote {z_profiles_0_path}")

    print(f"\nDone. Run adj_train.py (pkldir={output})")


if __name__ == "__main__":
    main()
