#!/usr/bin/env python3
"""
Axisymmetric droplets on a wall: solve the canonical Euler-Lagrange equation in (r, z) with the
neural functional at the (N, T) of every MD droplet field of the selected series, and measure the
density error against MD. Wall parameters and temperature range are set per series in DROP_SETS.
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import torch

from libs.cdft_1d.external_potentials import LJ93
from libs.cdft_axisym.mesh import axisym_mesh, integrate, mesh_coords
from libs.cdft_axisym.functional import AXISYM_FREE_ENERGY
from libs.cdft_axisym.solver import canonical_solve
from libs.ml.surrogates import load_ml_state_dicts
from libs.plot_utils import get_plot_dir, ensure_plot_dir
from libs.utils import resolve_training_device, tensors_to_cpu_for_storage

plt.rcParams["text.usetex"] = False

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
# Droplet series: MD folder md_drop_<set> -> (Ew, sigmaw, T_max)
# wn2: training wall, N ~ 4.5e4; wc_2: wc wall, N ~ 4.5e4; wc_1: wc wall, N ~ 2.5e4
# (the wc_1 droplets evaporate above T = 0.70)
DROP_SETS = {
    "wn2": (1.2, 1.2, 0.90),
    "wc_2": (1.0, 2.0, 0.90),
    "wc_1": (1.0, 2.0, 0.70),
}

outdir = os.path.join(script_dir, "..", "output") + "/"
datadir = os.path.join(script_dir, "..", "data") + "/"
pkldir = os.path.join(datadir, "dataset", "pkl", "droplets_3d") + "/"

# Output
profiles_pkl_name = "droplets_3d.pkl"
plotdir = get_plot_dir(script_dir, "..", "output", "plot_droplets_3d")

DEVICE_KIND = "auto"  # or "cuda" | "mps" | "cpu"; override with env TORCH_DEVICE
device = resolve_training_device(DEVICE_KIND)
print(f"Using device: {device}")
TORCH_DTYPE = torch.float64
torch.manual_seed(42)

# Flags
USE_DBH_DIAMETER = 1  # 1: use Barker-Henderson diameter scaling
RESTART_ML_MODEL = 1

# ------------------------------------------------------------------------------
# Thermodynamic and wall parameters
# ------------------------------------------------------------------------------
R = 1.
sigma_attr = R
cutoff_wall = 10. * R

# ------------------------------------------------------------------------------
# DFT Solver Params
# ------------------------------------------------------------------------------
fp_max_steps = 30000
fp_tol = 1e-9
fp_alpha = 0.02      # Picard damping
fp_warmup = 200      # Picard steps before Anderson mixing
anderson_m = 5
fp_verbose = 1

# ------------------------------------------------------------------------------
# Mesh
# ------------------------------------------------------------------------------
Rmax, Lz = 60., 60.
nr = int(Rmax / (0.2 * R))
nz = int(Lz / (0.2 * R))
mesh = axisym_mesh(Rmax, Lz, nr, nz, device=device, dtype=TORCH_DTYPE)
Rc, Zc = mesh_coords(mesh)  # (nr*nz,) cell centres
r1, z1 = mesh["r"].cpu().numpy(), mesh["z"].cpu().numpy()
z_wall = 0.

# ------------------------------------------------------------------------------
# External potential (shifted LJ9-3 wall at z_wall)
# ------------------------------------------------------------------------------
def wall_potential(Ew, sigmaw):
    z = torch.tensor(Zc, dtype=TORCH_DTYPE, device=device)
    Vext = (
        (LJ93(z, z_wall, Ew, sigmaw) - LJ93(cutoff_wall*torch.ones_like(z), z_wall, Ew, sigmaw))
        *
        torch.sigmoid(-(z - z_wall - cutoff_wall) / (0.001))
    )[0]  # (nr*nz,)
    return torch.clamp(Vext, max=1e3)

# ------------------------------------------------------------------------------
# MD droplet fields
# ------------------------------------------------------------------------------
def load_md_drop(path):
    # MD density rho(r, z) on the cell centres of the mesh (rows z-major, r fastest)
    df = pd.read_csv(path)
    r_md, z_md = np.unique(df["r"].values), np.unique(df["z"].values)
    assert len(r_md) == nr and len(z_md) == nz and np.allclose(r_md, r1) and np.allclose(z_md, z1), \
        f"{path} is not on the {nr}x{nz} mesh"
    F = df["rho"].values.reshape(nz, nr)  # (nz, nr)
    return F.T.reshape(-1)                # (nr*nz,) raveled r-major

cases = []
for set_name, (Ew, sigmaw, T_max) in DROP_SETS.items():
    md_dir = os.path.join(datadir, "dataset", f"md_drop_{set_name}")
    for f in sorted(os.listdir(md_dir)):
        if f.startswith("rho_field_T"):
            T_val = round(float(f[len("rho_field_T"):-len(".csv")]), 2)
            if T_val <= T_max + 1e-6:
                cases.append((T_val, set_name, os.path.join(md_dir, f)))
cases.sort()
print(f"Loaded {len(cases)} MD droplet fields: " + ", ".join(f"{s} T={T:.2f}" for T, s, _ in cases))

Vext = {set_name: wall_potential(Ew, sigmaw) for set_name, (Ew, sigmaw, _) in DROP_SETS.items()}

ml_state_dicts = load_ml_state_dicts(datadir, device) if RESTART_ML_MODEL else None

# ------------------------------------------------------------------------------
# Instantiate the functional (the WDA operator is assembled once, the attraction per T)
# ------------------------------------------------------------------------------
ti = time.time()
model = AXISYM_FREE_ENERGY(mesh, ml_state_dicts, T=cases[0][0], use_dbh=USE_DBH_DIAMETER)
print(f"Assembled the {nr}x{nz} pair operators [{time.time()-ti:.1f}s]")

# ------------------------------------------------------------------------------
# Run over the droplet series
# ------------------------------------------------------------------------------
records = []
T_current = None
for T_val, set_name, md_path in cases:
    if T_val != T_current:
        model.set_T(T_val)
        T_current = T_val
    rho_md = torch.tensor(load_md_drop(md_path), dtype=TORCH_DTYPE, device=device)  # (nr*nz,)
    N_val = integrate(mesh, rho_md).item()
    print(f"\n\nRunning {set_name}, N={N_val:.0f}, T={T_val:.2f}... ")
    ti = time.time()

    out = canonical_solve(
        model,
        Vext[set_name] / T_val,
        N_val,
        rho0=torch.clamp(rho_md, min=1e-10),
        max_steps=fp_max_steps,
        tol=fp_tol,
        alpha=fp_alpha,
        warmup=fp_warmup,
        m=anderson_m,
        verbose=fp_verbose,
    )
    rho_nn = out["U"]  # (nr*nz,)
    eps_nn = (integrate(mesh, torch.abs(rho_nn - rho_md)) / N_val).item()  # int |rho - rho_md| dV / N
    print(f"eps_rho: NN = {eps_nn:.4f} (res {out['res_max']:.1e}, {out['steps']} steps)  [{time.time()-ti:.1f}s]")

    records.append({
        "set": set_name,
        "T": T_val,
        "N": N_val,
        "r": r1,
        "z": z1,
        "rho_md": rho_md.cpu().numpy(),
        "rho_nn": rho_nn.cpu().numpy(),
        "eps_nn": eps_nn,
        "res_nn": out["res_max"],
    })

    fields = [(records[-1]["rho_md"], "MD"), (records[-1]["rho_nn"], "ML cDFT"),
              (records[-1]["rho_nn"] - records[-1]["rho_md"], "ML cDFT - MD")]
    vmax = max(records[-1]["rho_md"].max(), records[-1]["rho_nn"].max())
    plt.figure(figsize=(15, 4.5))
    for k, (fld, label) in enumerate(fields):
        plt.subplot(1, 3, k + 1)
        if k < 2:
            plt.pcolormesh(r1, z1, fld.reshape(nr, nz).T, cmap="viridis", shading="nearest", vmin=0, vmax=vmax)
        else:
            lim = np.abs(fld).max()
            plt.pcolormesh(r1, z1, fld.reshape(nr, nz).T, cmap="RdBu_r", shading="nearest", vmin=-lim, vmax=lim)
        plt.colorbar(label=r"$\rho$")
        plt.xlabel("r")
        plt.ylabel("z")
        plt.gca().set_aspect("equal")
        plt.title(label)
    plt.suptitle(f"{set_name}: N={N_val:.0f}, T={T_val:.2f}, eps_rho={eps_nn:.3f}")
    plt.tight_layout()
    plot_path = os.path.join(plotdir, f"rho_{set_name}_T{T_val:.2f}.png")
    ensure_plot_dir(plot_path)
    plt.savefig(plot_path, dpi=120)
    plt.close()

    df = pd.DataFrame(records).set_index(["set", "T"]).sort_index()
    profiles_path = os.path.join(pkldir, profiles_pkl_name)
    ensure_plot_dir(profiles_path)
    tensors_to_cpu_for_storage(df).to_pickle(profiles_path)
    df[["N", "eps_nn", "res_nn"]].to_csv(os.path.join(plotdir, "eps_rho_droplets.csv"))

# Error vs temperature, one curve per series
df = pd.DataFrame(records).set_index(["set", "T"]).sort_index()
for set_name, mk in zip(DROP_SETS, "os^"):
    d = df.loc[set_name]
    plt.semilogy(d.index, d["eps_nn"], mk + "-", label=f"ML {set_name}")
plt.xlabel("T")
plt.ylabel(r"$\epsilon_\rho$")
plt.title("Droplet density error vs MD")
plt.legend()
plt.grid()
plt.savefig(os.path.join(plotdir, "eps_rho_droplets.png"))
plt.close()
print(df[["N", "eps_nn", "res_nn"]])
