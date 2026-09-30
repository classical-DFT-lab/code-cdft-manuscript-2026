#!/usr/bin/env python3
"""
Planar adsorption on a wall: solve the Euler-Lagrange equation at the (N, T) of every MD
profile of the dataset, with the neural functional and with the mean-field baseline, and
measure the density error against MD. Vext and pkldir are set via WALL.
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

from libs.cdft_1d.augmented_lda import CDFT_MODEL as CDFT
from libs.cdft_1d.external_potentials import LJ93, Morse, Harmonic
from libs.solve_1d.newton import newton
from libs.solve_1d.picard import picard
from libs.ml.dataset_pd import MY_DATASET_FROM_DF
from libs.ml.surrogates import setDNN, setWDA, setDNNRep, load_ml_state_dicts
from libs.ml.loss import LossL1
from libs.io_utils import load_pickle
from libs.plot_utils import get_plot_dir, ensure_plot_dir
from libs.utils import resolve_training_device, tensors_to_cpu_for_storage

plt.rcParams["text.usetex"] = False

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
# Wall type: "wn2", "wc", "morse" or "harmonic" (sets Vext and pkldir)
WALL = "wc"

outdir = os.path.join(script_dir, "..", "output") + "/"
datadir = os.path.join(script_dir, "..", "data") + "/"
pkldir = os.path.join(datadir, "dataset", "pkl", f"profiles_wl_{WALL}") + "/"

# Output
profiles_pkl_name = f"eval_profiles_wl_{WALL}.pkl"
plotdir = get_plot_dir(script_dir, "..", "output", f"plot_eval_planar_{WALL}")

DEVICE_KIND = "auto"  # or "cuda" | "mps" | "cpu"; override with env TORCH_DEVICE
device = resolve_training_device(DEVICE_KIND)
print(f"Using device: {device}")
TORCH_DTYPE = torch.float64
torch.manual_seed(42)

# Flags
JACOBIAN = "EXACT"
USE_MODEL = 1
USE_DBH_DIAMETER = 1  # 1: use Barker-Henderson diameter scaling when USE_MODEL
TRAIN_DNN = 0
TRAIN_WDA = 0
RESTART_ML_MODEL = 1
SAVE_MODEL = 0
fast = False

# ------------------------------------------------------------------------------
# Thermodynamic parameters
# ------------------------------------------------------------------------------
R, mu, m = 1., 1., 1.
beta = 1 / 0.95
dft_type = "LDA"
ensemble = "NVT"
str_param = "mu" if ensemble == "muVT" else "N"
Lambda = 1.

# ------------------------------------------------------------------------------
# DFT Solver Params
# ------------------------------------------------------------------------------
picard_max_steps = 2000
picard_tol = 1e-9
picard_alpha = 0.03

newton_max_steps = 1600
newton_tol = 1e-9
newton_alpha = 0.9
newton_verbose = 0

# ------------------------------------------------------------------------------
# External potential parameters (Ew, sigmaw depend on WALL)
# ------------------------------------------------------------------------------
sigma_attr = R
eps_attr = 1.
cutoff_attr = 2.5 * R
if WALL == "wn2":
    Ew, sigmaw = 1.2, 1.2
elif WALL == "wc":
    Ew, sigmaw = 1., 2.
else:
    Ew, sigmaw = 0., 1.
cutoff_wall = 10. * R

# ------------------------------------------------------------------------------
# Mesh
# ------------------------------------------------------------------------------
L = 30.
xmin, xmax = 0, 2*L
Nx = int(2 * L / (0.2 * R))
dx = (xmax - xmin) / Nx
x = torch.linspace(xmin + dx / 2, xmax - dx / 2, Nx, dtype=TORCH_DTYPE).to(device)
dx = x[1] - x[0]
x_wall = xmin
BS = 1

# ------------------------------------------------------------------------------
# External potential
# ------------------------------------------------------------------------------
if WALL == "morse":
    Vext = (
        (Morse(x, x_wall, 1., 3., 1.) - Morse(cutoff_wall*torch.ones_like(x), x_wall, 1., 3., 1.))
        *
        torch.sigmoid(-(x - x_wall - cutoff_wall) / (0.001))
    )[None,...]  # [B, 1, Nx]
elif WALL == "harmonic":
    Vext = Harmonic(x, x_wall, 100., 1., 0.01)[None,...]  # [B, 1, Nx]
else:
    # Shifted LJ9-3 wall potential at x_wall
    Vext = (
        (LJ93(x, x_wall, Ew, sigmaw) - LJ93(cutoff_wall*torch.ones_like(x), x_wall, Ew, sigmaw))
        *
        torch.sigmoid(-(x - x_wall - cutoff_wall) / (0.001))
    )[None,...]  # [B, 1, Nx]

BC_R = "NONE"  # Right BC: NONE, ZEROGRAD, SYMM available

rho_guess = torch.zeros((BS, 1, Nx), dtype=TORCH_DTYPE).to(device)

# ------------------------------------------------------------------------------
# Build parameter dicts
# ------------------------------------------------------------------------------
eq_params = {
    "R": torch.tensor(R, dtype=TORCH_DTYPE, device=device),
    "mu": torch.tensor(mu, dtype=TORCH_DTYPE, device=device)[None, None],
    "beta": torch.tensor(beta, dtype=TORCH_DTYPE, device=device)[None, None],
    "Lambda": torch.tensor(Lambda, dtype=TORCH_DTYPE, device=device),
    "dft_type": dft_type,
    "ensemble": ensemble,
    "str_param": str_param,
    "sigma_attr": torch.tensor(sigma_attr, dtype=TORCH_DTYPE, device=device),
    "eps_attr": torch.tensor(eps_attr, dtype=TORCH_DTYPE, device=device),
    "cutoff_attr": torch.tensor(cutoff_attr, dtype=TORCH_DTYPE, device=device),
    "Ew": torch.tensor(Ew, dtype=TORCH_DTYPE, device=device),
    "sigmaw": torch.tensor(sigmaw, dtype=TORCH_DTYPE, device=device),
    "Vext": Vext,
    "BC_R": BC_R,
    "fast": fast,
}

mesh = {
    "BS": BS,
    "L": L,
    "Nx": Nx,
    "x_bc": [xmin, xmax],
    "x": x,
    "x_wall": torch.tensor(x_wall, dtype=TORCH_DTYPE, device=device),
    "dx": dx.to(device),
}

sol = {
    "rho_guess": rho_guess,
    "device": device,
    "dtype": TORCH_DTYPE,
    "outdir": outdir,
    "datadir": datadir,
    "pkldir": pkldir,
    "JACOBIAN": JACOBIAN,
    "RESTART_ML_MODEL": RESTART_ML_MODEL,
    "SAVE_MODEL": SAVE_MODEL,
    "USE_MODEL": USE_MODEL,
    "USE_DBH_DIAMETER": USE_DBH_DIAMETER,
    "TRAIN_DNN": TRAIN_DNN,
    "TRAIN_WDA": TRAIN_WDA,
    "LOSS": LossL1,
}

ml_state_dicts = load_ml_state_dicts(datadir, device) if RESTART_ML_MODEL else None

# ------------------------------------------------------------------------------
# Instantiate CDFT Model and ML models
# ------------------------------------------------------------------------------
model = CDFT(eq_params, mesh, sol)
sd = ml_state_dicts
setDNN(model, LR=0.0, state_dicts=sd)
setDNNRep(model, LR=0.0, state_dict=sd["dnn_rep_fn"] if sd else None)
setWDA(model, LR=0.0, modes=100, state_dict=sd["wda_fn"] if sd else None)

for param in model.dnn_g_fn.parameters(): param.requires_grad = False
for param in model.dnn_rep_fn.parameters(): param.requires_grad = False
for param in model.wda_fn.parameters(): param.requires_grad = False
model.dnn_g_fn.eval()
model.dnn_rep_fn.eval()
model.wda_fn.eval()

# ------------------------------------------------------------------------------
# MD profiles on the mesh
# ------------------------------------------------------------------------------
df_md = load_pickle(
    pkldir + "z_profiles.pkl",
    description="MD density profiles (z_profiles.pkl)",
    hint=f"Run: python preprocess_data.py --wall {WALL}",
)
data_md = MY_DATASET_FROM_DF(df_md, df_md, mesh["x"], device=device, transform=True, dtype=TORCH_DTYPE, x_wall=x_wall)
print(f"Loaded {len(data_md)} MD profiles for wall {WALL}.")

# ------------------------------------------------------------------------------
# Evaluation Function
# ------------------------------------------------------------------------------
def run_evaluation(T_val, N_val, rho_md, use_model):
    eq_params["beta"] = torch.tensor(1/T_val, dtype=TORCH_DTYPE, device=device)[None, None]
    eq_params["mu"] = rho_md.sum(-1) * dx  # (1, 1) same particle number as MD
    sol["USE_MODEL"] = use_model

    # Damped Picard from the MD profile, then Newton
    out_p = picard(
        model,
        U_guess=rho_md,
        max_steps=picard_max_steps,
        tol=picard_tol,
        alpha=picard_alpha,
        verbose=False,
    )
    out_fwd = newton(
        model,
        U_guess=out_p["U"],
        max_steps=newton_max_steps,
        tol=newton_tol,
        alpha=newton_alpha,
        verbose=newton_verbose,
        detach_tensors=True,
    )
    rho = out_fwd["U"]  # (1, 1, Nx)

    eps = torch.abs(rho - rho_md).mean().item()  # (1/L) int |rho - rho_md| dx
    rel_l2 = torch.sqrt(torch.sum((rho - rho_md)**2) / torch.sum(rho_md**2)).item()
    return rho[0, 0, :].cpu().numpy(), eps, rel_l2, out_fwd["res_max"]

# ------------------------------------------------------------------------------
# Run over the dataset
# ------------------------------------------------------------------------------
records = []
for i in range(len(data_md)):
    b = data_md[i]
    N_val = int(b["N"].item())
    T_val = round(b["T"].item(), 2)
    rho_md = b["rho"][None, ...]  # (1, 1, Nx)
    print(f"\n\nRunning N={N_val}, T={T_val:.2f}... ")
    ti = time.time()

    rho_nn, eps_nn, l2_nn, res_nn = run_evaluation(T_val, N_val, rho_md, use_model=1)
    rho_mf, eps_mf, l2_mf, res_mf = run_evaluation(T_val, N_val, rho_md, use_model=0)
    print(f"eps_rho: NN = {eps_nn:.3e} (rel L2 {l2_nn:.3e}, newton {res_nn:.1e})"
          f"  MF = {eps_mf:.3e} (rel L2 {l2_mf:.3e}, newton {res_mf:.1e})  [{time.time()-ti:.1f}s]")

    records.append({
        "N": N_val,
        "T": T_val,
        "x": x.cpu().numpy(),
        "rho_md": rho_md[0, 0, :].cpu().numpy(),
        "rho_nn": rho_nn,
        "rho_mf": rho_mf,
        "eps_nn": eps_nn,
        "eps_mf": eps_mf,
        "rel_l2_nn": l2_nn,
        "rel_l2_mf": l2_mf,
    })

    plt.plot(x.cpu(), records[-1]["rho_md"], color="black", label="MD")
    plt.plot(x.cpu(), rho_nn, "--", color="red", label="ML cDFT")
    plt.plot(x.cpu(), rho_mf, ":", color="grey", label="MF cDFT")
    plt.xlabel("x")
    plt.ylabel(r"$\rho$")
    plt.title(f"{WALL}: N={N_val}, T={T_val:.2f}, eps_rho={eps_nn:.2e}")
    plt.legend()
    plt.grid()
    plot_path = os.path.join(plotdir, f"rho_N{N_val}_T{T_val:.2f}.png")
    ensure_plot_dir(plot_path)
    plt.savefig(plot_path)
    plt.close()

    df = pd.DataFrame(records).set_index(["N", "T"]).sort_index()
    profiles_path = os.path.join(pkldir, profiles_pkl_name)
    ensure_plot_dir(profiles_path)
    tensors_to_cpu_for_storage(df).to_pickle(profiles_path)
    df[["eps_nn", "eps_mf", "rel_l2_nn", "rel_l2_mf"]].to_csv(os.path.join(plotdir, f"eps_rho_{WALL}.csv"))

# Error vs temperature, one curve per N
df = pd.DataFrame(records).set_index(["N", "T"]).sort_index()
for N_val in df.index.get_level_values("N").unique():
    d = df.loc[N_val]
    line, = plt.semilogy(d.index, d["eps_nn"], "o-", label=f"ML N={N_val}")
    plt.semilogy(d.index, d["eps_mf"], "s--", color=line.get_color(), alpha=0.5, label=f"MF N={N_val}")
plt.xlabel("T")
plt.ylabel(r"$\epsilon_\rho$")
plt.title(f"Density error vs MD, wall {WALL}")
plt.legend(fontsize=7)
plt.grid()
plt.savefig(os.path.join(plotdir, f"eps_rho_{WALL}.png"))
plt.close()
print(df[["eps_nn", "eps_mf", "rel_l2_nn", "rel_l2_mf"]])
