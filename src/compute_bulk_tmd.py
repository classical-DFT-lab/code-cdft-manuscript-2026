#!/usr/bin/env python3
"""
Compute bulk thermodynamic properties: critical point and liquid-vapor coexistence
curve from the neural CDFT model. Saves phase curve data and plots to output/bulk_tmd/.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from libs.cdft_1d.augmented_lda import CDFT_MODEL as CDFT
from libs.cdft_1d.external_potentials import LJ93
from libs.ml.surrogates import setDNN, setWDA, setDNNRep, load_ml_state_dicts
from libs.ml.loss import LossL1
from libs.utils import resolve_training_device, sol_dtype, tensors_to_cpu_for_storage
from libs.thermodynamics import batched_coexistence_temperatures, newton_critical_point



# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    plt.rcParams["text.usetex"] = False

    script_dir = os.path.dirname(os.path.abspath("src/"))
    sys.path.insert(0, script_dir)
    output_dir = os.path.join(script_dir, "..", "output", "bulk_tmd")
    os.makedirs(output_dir, exist_ok=True)

    outdir = os.path.join(script_dir, "..", "output") + "/"
    datadir = os.path.join(script_dir, "..", "data") + "/"
    pkldir = datadir + "dataset/pkl/bulk_tmd/"

    DEVICE_KIND = "auto"
    device = resolve_training_device(DEVICE_KIND)
    print(f"Using device: {device}")

    TORCH_DTYPE = torch.float64
    torch.manual_seed(42)

    # Flags
    JACOBIAN = "EXACT"
    USE_MODEL = 1
    USE_DBH_DIAMETER = 0
    TRAIN_DNN = 0
    TRAIN_WDA = 0
    RESTART_ML_MODEL = 1
    SAVE_MODEL = 1
    fast = False

    # Thermodynamic parameters
    R, mu, m = 1.0, 1.0, 1.0
    beta = 1 / 1.6
    dft_type = "LDA"
    ensemble = "NVT"
    guess_coex = [0.001, 0.9]
    str_param = "mu" if ensemble == "muVT" else "N"
    Lambda = 1.0

    # External potential parameters
    sigma_attr = R
    eps_attr = 1.0
    cutoff_attr = 2.5 * R
    Ew = 1.0
    sigmaw = 1 * sigma_attr

    # Full mesh
    L = 30.0
    xmin, xmax = -L, L
    Nx = int(2 * L / (0.2 * R))
    x = torch.linspace(xmin, xmax, Nx, dtype=TORCH_DTYPE).to(device)
    dx = x[1] - x[0]
    x_wall = xmin - 0.01 * sigmaw
    BS = 1

    # Minimal bulk mesh
    Nx_bulk = 3
    L_bulk = 0.2 * R * (Nx_bulk - 1) / 2
    x_bulk = torch.linspace(-L_bulk, L_bulk, Nx_bulk, dtype=TORCH_DTYPE).to(device)
    dx_bulk = x_bulk[1] - x_bulk[0]
    x_wall_bulk = x_bulk.min() - 0.01 * sigmaw

    Vext = LJ93(x, x_wall, Ew, sigmaw)[None, ...].to(device) * 0.0
    rho_guess = torch.zeros((BS, 1, Nx), dtype=TORCH_DTYPE).to(device)
    rho_guess_bulk = torch.zeros((1, 1, Nx_bulk), dtype=TORCH_DTYPE).to(device)

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
        "BC_R": "NONE",
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

    mesh_bulk = {
        "BS": 1,
        "L": L_bulk,
        "Nx": Nx_bulk,
        "x_bc": [x_bulk.min().item(), x_bulk.max().item()],
        "x": x_bulk,
        "x_wall": x_wall_bulk.to(device=device, dtype=TORCH_DTYPE)
        if torch.is_tensor(x_wall_bulk)
        else torch.tensor(x_wall_bulk, dtype=TORCH_DTYPE, device=device),
        "dx": dx_bulk.to(device),
        "BULK_COMP": True,
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

    sol_bulk = {**sol, "rho_guess": rho_guess_bulk}

    ml_state_dicts = load_ml_state_dicts(datadir, device) if RESTART_ML_MODEL else None

    model = CDFT(eq_params, mesh, sol)
    sd = ml_state_dicts
    setDNN(model, LR=0, state_dicts=sd)
    setDNNRep(model, LR=0, state_dict=sd["dnn_rep_fn"] if sd else None)
    setWDA(model, LR=0, modes=150, state_dict=sd["wda_fn"] if sd else None)

    # --------------------------------------------------------------------------
    # Critical points
    # --------------------------------------------------------------------------
    sol_bulk["USE_MODEL"] = True
    rho_c_guess, T_c_guess = 0.3090, 1.0779
    rho_c_nn, T_c_nn = newton_critical_point(
        eq_params, mesh_bulk, sol_bulk, rho_c_guess, T_c_guess,
        max_iter=150, tol=1e-9, alpha=0.5, device=device, ml_state_dicts_=ml_state_dicts
    )
    print("Critical point (NN):")
    print("rho_c =", rho_c_nn)
    print("T_c   =", T_c_nn)

    sol_bulk["USE_MODEL"] = False
    rho_c_guess_mf, T_c_guess_mf = 0.08, 0.95
    rho_c_mf, T_c_mf = newton_critical_point(
        eq_params, mesh_bulk, sol_bulk, rho_c_guess_mf, T_c_guess_mf,
        max_iter=150, tol=1e-9, alpha=1.0, device=device, ml_state_dicts_=None
    )
    print("Critical point (MF):")
    print("rho_c =", rho_c_mf)
    print("T_c   =", T_c_mf)

    # --------------------------------------------------------------------------
    # Batched coexistence
    # --------------------------------------------------------------------------
    T_list = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    T_list0 = [0.55, 0.65, 0.75, 0.85, 0.95]

    # Mean-field model
    sol_bulk["USE_MODEL"] = False
    rho_v_b0, rho_l_b0 = batched_coexistence_temperatures(
        T_list=T_list0,
        eq_params=eq_params,
        mesh_bulk=mesh_bulk,
        sol_bulk=sol_bulk,
        ml_state_dicts=ml_state_dicts,
        use_model=False,
        rho_init_pair=(1e-15, 1.0 - 1e-15),
        tol=1e-6,
        max_iter=1000,
        alpha=0.9,
        chunk_size=len(T_list0),
        verbose=True,
    )

    # Neural model
    sol_bulk["USE_MODEL"] = True
    rho_v_b, rho_l_b = batched_coexistence_temperatures(
                                T_list=T_list,
                                eq_params=eq_params,
                                mesh_bulk=mesh_bulk,
                                sol_bulk=sol_bulk,
                                ml_state_dicts=ml_state_dicts,
                                use_model=True,
                                rho_init_pair=(1e-12, 1.0 - 1e-12),
                                tol=1e-7,
                                max_iter=100000,
                                alpha=0.25,
                                chunk_size=len(T_list),   # or smaller if memory becomes an issue
                                verbose=True,
                            )

    # Literature MD data
    T_md = [0.64, 0.67, 0.70, 0.73, 0.76, 0.79, 0.82, 0.85, 0.88, 0.91, 0.94, 0.97, 1.00, 1.03, 1.06]
    rho_l_md = [0.8176, 0.8024, 0.7866, 0.7704, 0.7538, 0.7361, 0.7181, 0.6986, 0.6784, 0.6556, 0.6309, 0.6032, 0.5712, 0.530, 0.463]
    rho_v_md = [0.00351, 0.00525, 0.00727, 0.01036, 0.01374, 0.01776, 0.0233, 0.0303, 0.0392, 0.0483, 0.0616, 0.0763, 0.096, 0.127, 0.168]
    T_md.append(1.085)
    rho_v_md.append(0.3170)
    rho_l_md.append(0.3170)

    # --------------------------------------------------------------------------
    # DataFrames
    # --------------------------------------------------------------------------
    pc_nn = pd.DataFrame({
        "T": T_list,
        "rho_v": rho_v_b.detach().cpu().numpy(),
        "rho_l": rho_l_b.detach().cpu().numpy(),
    }).set_index("T")

    pc_0 = pd.DataFrame({
        "T": T_list0,
        "rho_v": rho_v_b0.detach().cpu().numpy(),
        "rho_l": rho_l_b0.detach().cpu().numpy(),
    }).set_index("T")

    pc_md = pd.DataFrame({
        "T": T_md,
        "rho_v": rho_v_md,
        "rho_l": rho_l_md,
    }).set_index("T")

    # For plotting
    rho_v_l_plt = rho_v_b.detach().cpu().numpy().tolist()
    rho_l_l_plt = rho_l_b.detach().cpu().numpy().tolist()
    rho_v_l0_plt = rho_v_b0.detach().cpu().numpy().tolist()
    rho_l_l0_plt = rho_l_b0.detach().cpu().numpy().tolist()

    # --------------------------------------------------------------------------
    # Plot
    # --------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(rho_v_l0_plt[:-1], T_list0[:-1], color="blue")
    plt.plot(rho_l_l0_plt[:-1], T_list0[:-1], color="blue", label="rho - MF")
    plt.plot(rho_v_l_plt, T_list, color="red")
    plt.plot(rho_l_l_plt, T_list, color="red", label="rho - NN")
    plt.plot(rho_v_md[:-1], T_md[:-1], "--", color="black")
    plt.plot(rho_l_md[:-1], T_md[:-1], "--", color="black", label="rho - MD")

    plt.plot([rho_v_l_plt[-1], rho_c_nn], [T_list[-1], T_c_nn], ":", color="red", linewidth=2, zorder=5)
    plt.plot([rho_l_l_plt[-1], rho_c_nn], [T_list[-1], T_c_nn], ":", color="red", linewidth=2, zorder=5)

    plt.plot([rho_v_l0_plt[-2], rho_v_l0_plt[-1], rho_c_mf], [T_list0[-2], T_list0[-1], T_c_mf], ":", color="blue", linewidth=2, zorder=5)
    plt.plot([rho_l_l0_plt[-2], rho_l_l0_plt[-1], rho_c_mf], [T_list0[-2], T_list0[-1], T_c_mf], ":", color="blue", linewidth=2, zorder=5)

    plt.plot(rho_v_md[-2:], T_md[-2:], ":", color="black")
    plt.plot(rho_l_md[-2:], T_md[-2:], ":", color="black")

    plt.scatter(rho_c_nn, T_c_nn, c="red", label="T_c - NN")
    plt.scatter(rho_v_md[-1], T_md[-1], c="black", label="T_c - MD")
    plt.scatter(rho_c_mf, T_c_mf, c="blue", label="T_c - MF")

    plt.ylabel("T")
    plt.xlabel(r"$\rho$")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "phase_curve.svg"))
    plt.savefig(os.path.join(output_dir, "phase_curve.png"))
    plt.close()

    # --------------------------------------------------------------------------
    # Save
    # --------------------------------------------------------------------------
    tensors_to_cpu_for_storage(pc_nn).to_pickle(os.path.join(pkldir, "pc_nn_operative.pkl"))
    tensors_to_cpu_for_storage(pc_md).to_pickle(os.path.join(pkldir, "pc_md_operative.pkl"))
    tensors_to_cpu_for_storage(pc_0).to_pickle(os.path.join(pkldir, "pc_0_operative.pkl"))

    with open(os.path.join(output_dir, "critical_point.txt"), "w") as f:
        f.write(f"NN: rho_c = {rho_c_nn}\nNN: T_c   = {T_c_nn}\n")
        f.write(f"MF: rho_c = {rho_c_mf}\nMF: T_c   = {T_c_mf}\n")

    print(f"\nOutputs saved to {os.path.abspath(output_dir)}")
    print("\npc_nn (neural model):")
    print(pc_nn)