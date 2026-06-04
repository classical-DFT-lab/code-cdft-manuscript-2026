#!/usr/bin/env python3
"""
Adsorption continuation: wall-liquid isotherm computation via Newton + continuation.
Vext and pkldir are set via WALL string (wn2 or wc) as in neural_cdft_dbh.
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
import pickle
import time
import torch

from libs.cdft_1d.augmented_lda import CDFT_MODEL as CDFT
from libs.cdft_1d.external_potentials import LJ93
from libs.solve_1d.newton import newton
from libs.solve_1d.continuation_gpu import continuation
from libs.ml.dataset_pd import setDatasetObject
from libs.ml.surrogates import setDNN, setWDA, setDNNRep, load_ml_state_dicts
from libs.ml.loss import LossL1
from libs.io_utils import load_pickle, DataNotFoundError
from libs.plot_utils import get_plot_dir, ensure_plot_dir
from libs import thermodynamics as tmd

plt.rcParams["text.usetex"] = False

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Wall type: "wn2" or "wc" (sets Vext params and pkldir)
WALL = "wc"

outdir = os.path.join(script_dir, "..", "output") + "/"
datadir = os.path.join(script_dir, "..", "data") + "/"
if WALL == "wn2":
    pkldir = os.path.join(datadir, "dataset", "pkl", "profiles_wl_wn2") + "/"
elif WALL == "wc":
    pkldir = os.path.join(datadir, "dataset", "pkl", "profiles_wl_wc") + "/"
else:
    pkldir = os.path.join(datadir, "dataset", "pkl", "profiles_wl") + "/"

plotdir = get_plot_dir(script_dir, "..", "output", "plot_adsorption_isotherm")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")
torch.manual_seed(42)

# Flags
JACOBIAN = "STABLE"
USE_MODEL = 1
USE_DBH_DIAMETER = 0  # 1: use Barker-Henderson diameter scaling when USE_MODEL
TRAIN_DNN = 0
TRAIN_WDA = 0
RESTART_ML_MODEL = 1
SAVE_MODEL = 1
fast = False

# ------------------------------------------------------------------------------
# Thermodynamic parameters
# ------------------------------------------------------------------------------
R, mu, m = 1., 5., 1.
beta = 1 / 0.95
dft_type = "LDA"
ensemble = "NVT"
guess_coex = [0.001, 0.9]
str_param = "mu" if ensemble == "muVT" else "N"
Lambda = 1.

# ------------------------------------------------------------------------------
# DFT Solver Params
# ------------------------------------------------------------------------------
newton_max_steps = 1600
newton_tol = 5e-8
newton_alpha = 0.5
newton_verbose = 1

cont_ds = 0.05 #* (1 - 2*(ensemble == "NVT"))
cont_continuation_steps = 30000
cont_max_corrector_steps = 500
cont_alpha = 0.9

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
    Ew, sigmaw = 1., 1.
cutoff_wall = 10. * R

# ------------------------------------------------------------------------------
# Mesh
# ------------------------------------------------------------------------------
L = 30.
xmin, xmax = -L, L
Nx = int(2 * L / (0.15 * R))
x = torch.linspace(xmin, xmax, Nx, dtype=torch.double).to(device)
dx = x[1] - x[0]
x_wall = xmin - 0.001 * sigmaw
BS = 1

# ------------------------------------------------------------------------------
# External potential and initial guess
# ------------------------------------------------------------------------------
# 6 // External Potential (e.g., Hard Wall) Setup
#Vext = HW(x, x_wall, Ew, sigmaw)
Vext = (
    (LJ93(x, x_wall, Ew, sigmaw) - LJ93(cutoff_wall*torch.ones_like(x), x_wall, Ew, sigmaw)) 
    * 
    torch.sigmoid(-(x - x_wall - cutoff_wall) / (0.001)) 
)[None,...]  # [B, 1, Nx].      # Shifted 

BC_R = "NONE"  # Right BC: NONE, ZEROGRAD, SYMM available

rho_guess = torch.zeros((BS, 1, Nx), dtype=torch.double).to(device)

# ------------------------------------------------------------------------------
# Build parameter dicts
# ------------------------------------------------------------------------------
eq_params = {
    "R": torch.tensor(R).to(device),
    "mu": torch.tensor(mu).to(device)[None, None],
    "beta": torch.tensor(beta).to(device)[None, None],
    "Lambda": torch.tensor(Lambda).to(device),
    "dft_type": dft_type,
    "ensemble": ensemble,
    "str_param": str_param,
    "sigma_attr": torch.tensor(sigma_attr).to(device),
    "eps_attr": torch.tensor(eps_attr).to(device),
    "cutoff_attr": torch.tensor(cutoff_attr).to(device),
    "Ew": torch.tensor(Ew).to(device),
    "sigmaw": torch.tensor(sigmaw).to(device),
    "Vext": Vext,
    "BC_R": BC_R,
    "fast": fast,
    "safe": 1.,  #1. for full jacobian, 0. for approx
}

mesh = {
    "BS": BS,
    "L": L,
    "Nx": Nx,
    "x_bc": [xmin, xmax],
    "x": x,
    "x_wall": torch.tensor(x_wall).to(device),
    "dx": dx.to(device),
}

sol = {
    "rho_guess": rho_guess,
    "device": device,
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
# Evaluation Function
# ------------------------------------------------------------------------------
def run_evaluation(T_val, N_val, N_start, target_density, rho_init, refine, plot_prefix="", plotdir_=None):
    plotdir_local = plotdir_ or plotdir

        
    eq_params["beta"] = torch.tensor(1/T_val, dtype=torch.double).to(device)[None, None]
    eq_params["mu"] = N_start / (20**2) 
    # eq_params["mu"] = torch.tensor(0.7 * N_val / (20**2), dtype=torch.double).to(device)[None, None]
    eq_params["target_density"] = target_density
    sol["rho_guess"] = rho_init.to(device)
    sol["rho_vl"] = torch.tensor([0.,0.], dtype=torch.double, device=device)

    model = CDFT(eq_params, mesh, sol)
    sd = ml_state_dicts
    setDNN(model, LR=0.0, state_dicts=sd)
    setDNNRep(model, LR=0.0, state_dict=sd["dnn_rep_fn"] if sd else None)
    setWDA(model, LR=0.0, modes=150, state_dict=sd["wda_fn"] if sd else None)

    for param in model.dnn_fn.parameters(): param.requires_grad = False
    for param in model.dnn_rep_fn.parameters(): param.requires_grad = False
    for param in model.wda_fn.parameters(): param.requires_grad = False
    model.dnn_fn.eval()
    model.dnn_rep_fn.eval()
    model.wda_fn.eval()
    
    # Newton solution
    out_fwd = newton(model, U_guess=sol["rho_guess"], max_steps=newton_max_steps, tol=newton_tol,
                     alpha=newton_alpha, verbose=newton_verbose, detach_tensors=True)

    if refine:
        # Continuation
        sol_0 = {
            "mu": out_fwd["U"].sum()*dx*torch.ones([mesh["BS"], 1], dtype=torch.double).to(device),
            "rho": out_fwd["U"], # (1, Nx)
            "rhocoex_vl": torch.tensor([0., 0.], dtype=torch.double).to(device),
        }

        sol_curve = continuation(
            sol_0, model,
            ds=cont_ds,
            continuation_steps=cont_continuation_steps,
            max_corrector_steps=cont_max_corrector_steps,
            alpha=cont_alpha,
            plotdir=plotdir_local,
            detach_tensors=True,
        )

        # mu_, rho_ = tmd.compute_mu_rho_curve(sol_curve, target_density)
        mu_, rho_ = sol_curve[-1]["mu"], sol_curve[-1]["rho"]
        mu_ = mu_.detach(); rho_ = rho_.detach()
    
        # Final Newton solve at corrected mu
        model.sol["rho_guess"] = rho_.squeeze().to(device).double()[None, None, :]
        model.eq_params["mu"] = mu_.squeeze().to(device).double()[None, None]
        # model.eq_params["mu"] = torch.tensor(target_density * 2*L, dtype=torch.double).to(device)[None, None] if ensemble == "NVT" else mu_
        
        out_final = newton(
            model, model.sol["rho_guess"],
            max_steps=newton_max_steps,
            tol=newton_tol,
            alpha=newton_alpha ,
            verbose=newton_verbose,
            detach_tensors=True
        )

    else:
        out_final = out_fwd
        sol_curve = None

    # Store and plot
    result = {
        "T": T_val,
        "N": N_val,
        "rho": out_final["U"][0, 0, :].cpu().numpy(),
        "x": model.mesh["x"].cpu().numpy(),
        "isotherm": sol_curve,
    }

    plt.plot(model.mesh["x"].cpu(), result["rho"], label=f"T={T_val}, N={N_val}")
    plt.plot(model.mesh["x"].cpu(), rho_init.detach().cpu().squeeze(), label=f"T={T_val}, N={N_val}")
    plt.legend()
    plt.grid()
    plt.title(f"rho(x) at T={T_val}, N={N_val}")
    plot_path = os.path.join(plotdir_local, f"{plot_prefix}rho_N{N_val}_T{T_val}.png")
    ensure_plot_dir(plot_path)
    plt.savefig(plot_path)
    plt.close()

    return result

# ------------------------------------------------------------------------------
# Run Full Sweep
# ------------------------------------------------------------------------------
# Initialize new df
data = pd.DataFrame(columns=["T", "N", "rho", "x"]).set_index(["N", "T"])
print("Initialized new empty DataFrame.")


T_vals = [0.55, 0.60, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
N_vals = [8000,]

a = -L + 5*sigmaw

for N_val in N_vals:
    new_results = []
    for T_val in T_vals:
        print(f"\n\nRunning N={N_val}, T={T_val:.2f}... ")
        rho_l = 0.8 #pc_nn.loc[T_val, "rho_l"]
        rho_v = 0.01 #pc_nn.loc[T_val, "rho_v"]

        # rho_init = (torch.sigmoid((mesh["x"]+a))*torch.sigmoid(-(mesh["x"]-a))*(rho_l - rho_v) + rho_v)[None, None, ...].to(device).double() 
        rho_init = 0.000001*(torch.sigmoid(10*(mesh["x"]+L))*torch.sigmoid(-10*(mesh["x"]-a))*(rho_l - rho_v) + rho_v)[None, None, ...].to(device).double() 

        target_density = 25/(2*L)
        N_start = rho_init.sum(-1)*mesh["dx"]*20**2

        result = run_evaluation(
                                    T_val, 
                                    N_val, 
                                    N_start,
                                    target_density,
                                    rho_init, 
                                    refine=1,
                                    plot_prefix="temp/"
                                    )
        
        new_results.append(result)

        df_new = pd.DataFrame(new_results).set_index(["N", "T"]).sort_index()
        df_new = pd.concat([data, df_new])
        isotherm_path = os.path.join(pkldir, f"isotherm_profiles_wl_{WALL}.pkl")
        ensure_plot_dir(isotherm_path)
        df_new.to_pickle(isotherm_path)
