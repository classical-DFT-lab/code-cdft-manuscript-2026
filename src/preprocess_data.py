#!/usr/bin/env python3
"""
Build the pickles read by the planar scripts (adj_train.py, eval_planar.py) from the MD profiles.

For every wall, reads data/dataset/md_planar_<wall>/N*/T_*/z_profile.csv (columns x, rho: distance
from the wall on the cell centres of the dx = 0.2 grid, and MD density) and writes to
data/dataset/pkl/profiles_wl_<wall>/:
  - z_profiles.pkl:   the MD profiles, indexed by (N, T)
  - z_profiles_0.pkl: the mean-field (MF) profiles at the same (N, T), solved here; they are only
                      drawn next to the MD profile in the training plots

Usage:
  python preprocess_data.py                 # all four walls
  python preprocess_data.py --wall wn2      # one wall: wn2, wc, morse or harmonic
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
from libs.cdft_1d.external_potentials import LJ93, Morse, Harmonic
from libs.solve_1d.newton import newton
from libs.solve_1d.picard import picard
from libs.ml.surrogates import setDNN, setDNNRep, setWDA
from libs.ml.loss import LossL1
from libs.utils import (
    resolve_training_device,
    sol_dtype,
    tensors_to_cpu_for_storage,
)


def gather_profiles(root: Path) -> pd.DataFrame:
    """Gather the z_profile.csv files of an N*/T_* tree. Returns a DataFrame with MultiIndex (N, T)
    and array columns x, rho."""
    records = []
    for N_dir in sorted(root.glob("N*/")):
        N = int(N_dir.name.lstrip("N"))
        for T_dir in sorted(N_dir.glob("T_*/")):
            csv = T_dir / "z_profile.csv"
            if not csv.is_file():
                continue
            df = pd.read_csv(csv)
            records.append({"N": N, "T": float(T_dir.name.split("_", 1)[1]),
                            "x": df["x"].to_numpy(), "rho": df["rho"].to_numpy()})
    if not records:
        raise FileNotFoundError(
            f"No z_profile.csv files found under {root}\n"
            f"Expected structure: {root}/N8000/T_0.55/z_profile.csv"
        )
    return pd.DataFrame(records).set_index(["N", "T"])


def default_dataset_root(name: str) -> Path:
    return (Path(script_dir) / ".." / "data" / "dataset" / name).resolve()


def build_z_profiles_0(df_md, mesh, eq_params, sol, device, wall="wc"):
    """
    Mean-field profiles (z_profiles_0): solve the Euler-Lagrange equation of the baseline functional
    (USE_MODEL=False) at each (N, T), starting from the MD profile.
    """
    L = mesh["L"]
    x = mesh["x"]
    dx = mesh["dx"]
    x_wall = mesh["x_wall"].item() if torch.is_tensor(mesh["x_wall"]) else mesh["x_wall"]
    cutoff_wall = 10.0

    if wall == "morse":
        Vext = (
            (Morse(x, x_wall, 1.0, 3.0, 1.0) - Morse(cutoff_wall * torch.ones_like(x), x_wall, 1.0, 3.0, 1.0))
            * torch.sigmoid(-(x - x_wall - cutoff_wall) / 1e-3)
        )[None, ...].to(device)
    elif wall == "harmonic":
        Vext = Harmonic(x, x_wall, 100.0, 1.0, 0.01)[None, ...].to(device)
    else:
        Ew, sigmaw = (1.2, 1.2) if wall == "wn2" else (1.0, 2.0)
        Vext = (
            (LJ93(x, x_wall, Ew, sigmaw) - LJ93(cutoff_wall * torch.ones_like(x), x_wall, Ew, sigmaw))
            * torch.sigmoid(-(x - x_wall - cutoff_wall) / 1e-3)
        )[None, ...].to(device)

    records = []
    dt = sol_dtype(sol)
    for (N, T), row in df_md.iterrows():
        # Initial guess: the MD profile (already on the cell centres of the mesh)
        assert np.allclose(row["x"], x.cpu().numpy() - x_wall), "MD profile not on the dx = 0.2 cell-centred grid"
        rho_guess = torch.tensor(row["rho"], dtype=dt, device=device)[None, None, :]

        beta = 1.0 / T
        mu_val = N / 20 ** 2  # particles per unit wall area (20 x 20 MD box)
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
        setWDA(model, LR=0.0, modes=100)
        model.dnn_g_fn.eval()
        model.dnn_rep_fn.eval()
        model.wda_fn.eval()

        try:
            out = picard(
                model, U_guess=rho_guess,
                max_steps=2000, tol=1e-9, alpha=0.03, verbose=False
            )
            out = newton(
                model, U_guess=out["U"],
                max_steps=800, tol=1e-6, alpha=0.5, verbose=False, detach_tensors=True
            )
            rho_sol = out["U"][0, 0, :].cpu().numpy()
            x_sol = x.cpu().numpy()
        except Exception as e:
            print(f"  Warning: mean-field solve failed for N={N}, T={T}: {e}")
            rho_sol = np.array(row["rho"]).flatten()  # fallback to MD
            x_sol = np.array(row["x"]).flatten()

        records.append({"N": N, "T": T, "x": x_sol, "rho": rho_sol})

    df_0 = pd.DataFrame(records).set_index(["N", "T"])
    return df_0


WALLS = ("wn2", "wc", "morse", "harmonic")


def process_wall(wall, device, TORCH_DTYPE):
    md_source = default_dataset_root(f"md_planar_{wall}")
    output = (Path(script_dir) / ".." / "data" / "dataset" / "pkl" / f"profiles_wl_{wall}").resolve()
    output.mkdir(parents=True, exist_ok=True)

    # 1. MD profiles -> z_profiles.pkl
    print(f"\n[{wall}] reading {md_source}")
    df_md = gather_profiles(md_source)
    z_profiles_path = output / "z_profiles.pkl"
    tensors_to_cpu_for_storage(df_md).to_pickle(z_profiles_path)
    print(f"[{wall}] wrote {z_profiles_path} with {len(df_md)} profiles")
    for N in df_md.index.get_level_values("N").unique():
        Ts = df_md.loc[N].index.get_level_values("T").unique()
        print(f"  N={N}: T = [{', '.join(f'{t:.2f}' for t in Ts)}]")

    # 2. Mean-field profiles -> z_profiles_0.pkl
    print(f"[{wall}] solving the mean-field profiles ...")
    R, Lambda = 1.0, 1.0
    L = 30.0
    xmin, xmax = 0, 2*L
    Nx = int(2 * L / 0.2)
    dx = (xmax - xmin) / Nx
    x = torch.linspace(xmin + dx / 2, xmax - dx / 2, Nx, dtype=TORCH_DTYPE).to(device)
    dx = x[1] - x[0]
    if wall == "wn2":
        Ew, sigmaw = 1.2, 1.2
    elif wall == "wc":
        Ew, sigmaw = 1.0, 2.0
    else:
        Ew, sigmaw = 0.0, 1.0
    x_wall = xmin

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
    df_0 = build_z_profiles_0(df_md, mesh, eq_params, sol, device, wall)
    z_profiles_0_path = output / "z_profiles_0.pkl"
    tensors_to_cpu_for_storage(df_0).to_pickle(z_profiles_0_path)
    print(f"[{wall}] wrote {z_profiles_0_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build z_profiles.pkl and z_profiles_0.pkl from the MD profile CSVs"
    )
    parser.add_argument(
        "--wall",
        choices=WALLS + ("all",),
        default="all",
        help="Wall to process (default: all four)",
    )
    args = parser.parse_args()

    DEVICE_KIND = "auto"  # or "cuda" | "mps" | "cpu"; override with env TORCH_DEVICE
    device = resolve_training_device(DEVICE_KIND)
    print(f"Using device: {device}")
    # Tensor dtype for DFT tensors (torch.float64 or torch.float32)
    TORCH_DTYPE = torch.float32
    torch.manual_seed(42)

    for wall in (WALLS if args.wall == "all" else (args.wall,)):
        process_wall(wall, device, TORCH_DTYPE)
    print("\nDone.")


if __name__ == "__main__":
    main()
