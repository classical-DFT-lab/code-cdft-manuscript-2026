#!/usr/bin/env python3
"""
Compute bulk thermodynamic properties: critical point and liquid-vapor coexistence
curve from the neural CDFT model. Saves phase curve data and plots to output/bulk_tmd/.
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
import torch

from libs.cdft_1d.augmented_lda import CDFT_MODEL as CDFT
from libs.cdft_1d.external_potentials import LJ93
from libs.ml.surrogates import setDNN, setWDA, setDNNRep, load_ml_state_dicts
from libs.ml.loss import LossL1

plt.rcParams["text.usetex"] = False

# ------------------------------------------------------------------------------
# Output directory
# ------------------------------------------------------------------------------
output_dir = os.path.join(script_dir, "..", "output", "bulk_tmd")
os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
outdir = os.path.join(script_dir, "..", "output") + "/"
datadir = os.path.join(script_dir, "..", "data") + "/"
pkldir = datadir + "dataset/pkl/profiles/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
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
R, mu, m = 1., 1., 1.
beta = 1 / 1.6
dft_type = "LDA"
ensemble = "NVT"
guess_coex = [0.001, 0.9]
str_param = "mu" if ensemble == "muVT" else "N"
Lambda = 1.

# External potential parameters
sigma_attr = R
eps_attr = 1.
cutoff_attr = 2.5 * R
Ew = 1.
sigmaw = 1 * sigma_attr

# Mesh (full-size, kept for compatibility with model init)
L = 30.
xmin, xmax = -L, L
Nx = int(2 * L / (0.2 * R))
x = torch.linspace(xmin, xmax, Nx, dtype=torch.double).to(device)
dx = x[1] - x[0]
x_wall = xmin - 0.01 * sigmaw
BS = 1

# Minimal mesh for bulk (0D problem: uniform density)
# Exploit homogeneity: use Nx=3 (minimal for FFT), output is uniform everywhere.
# Attractive uses fixed r_grid for integral; WDA supports small grids via BULK_COMP.
Nx_bulk = 3
L_bulk = 0.2 * R * (Nx_bulk - 1) / 2  # symmetric around 0
x_bulk = torch.linspace(-L_bulk, L_bulk, Nx_bulk, dtype=torch.double).to(device)
dx_bulk = x_bulk[1] - x_bulk[0]
x_wall_bulk = x_bulk.min() - 0.01 * sigmaw

# External potential and initial guess
Vext = LJ93(x, x_wall, Ew, sigmaw)[None, ...].to(device) * 0.
rho_guess = torch.zeros((BS, 1, Nx), dtype=torch.double).to(device)
rho_guess_bulk = torch.zeros((BS, 1, Nx_bulk), dtype=torch.double).to(device)

# Build parameter dicts
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
    "BC_R": "NONE",
    "fast": fast,
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

mesh_bulk = {
    "BS": BS,
    "L": L_bulk,
    "Nx": Nx_bulk,
    "x_bc": [x_bulk.min().item(), x_bulk.max().item()],
    "x": x_bulk,
    "x_wall": x_wall_bulk.to(device) if torch.is_tensor(x_wall_bulk) else torch.tensor(x_wall_bulk, device=device),
    "dx": dx_bulk.to(device),
    "BULK_COMP": True,  # 0D bulk: WDA supports small grids
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

# sol_bulk: same as sol but with minimal rho_guess for bulk (0D) computations
sol_bulk = {**sol, "rho_guess": rho_guess_bulk}

# Load ML state dicts once to avoid repeated disk I/O in coexistence/critical-point loops
ml_state_dicts = load_ml_state_dicts(datadir, device) if RESTART_ML_MODEL else None


def update_model(x_, rho_, T_, mesh_=None, sol_=None, ml_state_dicts_=None):
    """Build CDFT model for given mesh and temperature.
    If mesh_ and sol_ are provided (e.g. mesh_bulk, sol_bulk), use them for efficient bulk (0D) computations.
    """
    if mesh_ is not None and sol_ is not None:
        mesh_local = mesh_
        sol_local = sol_
    else:
        mesh_local = {
            "BS": 1,
            "L": x_.max().item(),
            "Nx": len(x_),
            "x_bc": [x_.min().item(), x_.max().item()],
            "x": x_,
            "x_wall": x_.min() - 0.001,
            "dx": (x_[1] - x_[0]).to(device),
        }
        sol_local = sol
    eq_params_local = dict(eq_params)
    eq_params_local["beta"] = torch.tensor(1 / T_, dtype=torch.double).to(device)[None, None]
    eq_params_local["mu"] = rho_.sum(-1) * mesh_local["dx"]
    eq_params_local["Vext"] = LJ93(mesh_local["x"], mesh_local["x_wall"], 1, 2)[None, ...].to(device) * 0
    model = CDFT(eq_params_local, mesh_local, sol_local)
    sd = ml_state_dicts_ if ml_state_dicts_ is not None else None
    setDNN(model, LR=0.0, state_dicts=sd)
    setDNNRep(model, LR=0.0, state_dict=sd["dnn_rep_fn"] if sd else None)
    setWDA(model, LR=0.0, modes=150, state_dict=sd["wda_fn"] if sd else None)
    return model


def chem_pot(rho, model):
    """Bulk chemical potential."""
    beta = model.eq_params["beta"][..., None]
    Lambda = model.eq_params["Lambda"]
    DF = model.gradients_FX(rho, detach_tensors=True, compute_D2FX=False)["DF"]
    mu = 1 / beta * (torch.log(Lambda**3 * rho)) + DF
    return mu


def coex_equations_auto(rho_vec, model):
    """Returns [eq1, eq2] enforcing mu(rho1)=mu(rho2) and P(rho1)=P(rho2)."""
    r1, r2 = rho_vec
    R1 = r1 * torch.ones_like(model.sol["rho_guess"])
    R2 = r2 * torch.ones_like(model.sol["rho_guess"])
    mu1 = model.GetChemPot(R1.sum() * model.mesh["dx"], R1).squeeze()
    mu2 = model.GetChemPot(R2.sum() * model.mesh["dx"], R2).squeeze()
    p1 = -model.GetOmega(R1)[0][0, 0, R1.shape[-1] // 2]
    p2 = -model.GetOmega(R2)[0][0, 0, R2.shape[-1] // 2]
    return [mu1 - mu2, p1 - p2]


def solve_newton(coex_equations, rho_init, model, tol=1e-5, max_iter=1000, alpha=0.3, verbose=True):
    """Newton-Raphson for coexistence equations."""
    rho0 = rho_init.clone().detach().to(model.sol["device"])
    for i in range(max_iter):
        rho0.requires_grad_(True)
        rho = rho0
        F = coex_equations(rho, model)
        J = torch.zeros((2, 2), dtype=torch.float64, device=rho.device)
        for j in range(2):
            J_i = torch.autograd.grad(
                F[j], rho, retain_graph=True, materialize_grads=True
            )[0]
            J[j, :] = J_i
        F_n = torch.tensor([F[0].item(), F[1].item()], dtype=torch.double, device=rho.device)
        with torch.no_grad():
            delta_rho = torch.linalg.solve(J, -F_n)
            rho0.add_(alpha * delta_rho)
        if verbose and i % 10 == 0:
            print(i, delta_rho.norm().item(), rho0)
        if torch.norm(delta_rho).item() < tol:
            if verbose:
                print("Number of Newton iterations:", i + 1)
            break
    return rho0.detach()


def newton_critical_point(eq_params, mesh, sol, rho0, T0, max_iter=20, tol=1e-8, alpha=1., device="cuda", ml_state_dicts_=None):
    """Solve for (rho_c, T_c) such that ∂p/∂rho=0 and ∂²p/∂rho²=0."""
    rho = torch.tensor(float(rho0), device=device).double()
    T = torch.tensor(float(T0), device=device).double()
    # Vext matching mesh (zero for bulk)
    Vext_mesh = LJ93(mesh["x"], mesh["x_wall"], 1, 2)[None, ...].to(device) * 0
    for it in range(max_iter):
        rho_var = rho.clone().detach().requires_grad_(True)
        T_var = T.clone().detach().requires_grad_(True)
        beta = 1.0 / T_var
        eq_params_local = dict(eq_params)
        eq_params_local["beta"] = beta * torch.ones_like(eq_params_local["mu"]).double()
        eq_params_local["Vext"] = Vext_mesh
        model = CDFT(eq_params_local, mesh, sol)
        sd = ml_state_dicts_
        setDNN(model, LR=0.0, state_dicts=sd)
        setDNNRep(model, LR=0.0, state_dict=sd["dnn_rep_fn"] if sd else None)
        setWDA(model, LR=0.0, modes=150, state_dict=sd["wda_fn"] if sd else None)
        model.dnn_fn.eval()
        model.dnn_rep_fn.eval()
        model.wda_fn.eval()
        R1 = rho_var * torch.ones_like(model.sol["rho_guess"])
        omega = model.GetOmega(R1)[0]
        Nx = omega.shape[-1]
        p = -omega[0, 0, Nx // 2]
        p_prime = torch.autograd.grad(p, rho_var, create_graph=True)[0]
        p_second = torch.autograd.grad(p_prime, rho_var, create_graph=True)[0]
        f1, f2 = p_prime, p_second
        F = torch.stack([f1, f2])
        if F.norm().item() < tol:
            print(f"Converged in {it} iterations (function norm).")
            break
        df1_drho = torch.autograd.grad(f1, rho_var, retain_graph=True)[0]
        df1_dT = torch.autograd.grad(f1, T_var, retain_graph=True)[0]
        df2_drho = torch.autograd.grad(f2, rho_var, retain_graph=True)[0]
        df2_dT = torch.autograd.grad(f2, T_var)[0]
        J = torch.stack([torch.stack([df1_drho, df1_dT]), torch.stack([df2_drho, df2_dT])])
        delta = torch.linalg.solve(J, -F)
        rho = rho + delta[0]
        T = T + delta[1]
        print("Norm:", delta.norm().item())
        if delta.norm().item() < tol:
            print(f"Converged in {it + 1} iterations (step size).")
            break
    return rho.item(), T.item()


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    model = CDFT(eq_params, mesh, sol)
    sd = ml_state_dicts
    setDNN(model, LR=0, state_dicts=sd)
    setDNNRep(model, LR=0, state_dict=sd["dnn_rep_fn"] if sd else None)
    setWDA(model, LR=0, modes=150, state_dict=sd["wda_fn"] if sd else None)

    # Critical point NN (bulk mesh with BULK_COMP for efficiency)
    sol_bulk["USE_MODEL"] = True
    rho_c_guess, T_c_guess = 0.3190, 1.0779
    rho_c_nn, T_c_nn = newton_critical_point(
        eq_params, mesh_bulk, sol_bulk, rho_c_guess, T_c_guess,
        max_iter=150, tol=1e-9, alpha=1., device=device, ml_state_dicts_=ml_state_dicts
    )
    print("Critical point (NN):")
    print("rho_c =", rho_c_nn)
    print("T_c   =", T_c_nn)

    # Critical point MF (USE_MODEL=False)
    sol_bulk["USE_MODEL"] = False
    rho_c_guess_mf, T_c_guess_mf = 0.08, 0.95  # MF/LDA critical point is at lower rho, T
    rho_c_mf, T_c_mf = newton_critical_point(
        eq_params, mesh_bulk, sol_bulk, rho_c_guess_mf, T_c_guess_mf,
        max_iter=150, tol=1e-9, alpha=1., device=device, ml_state_dicts_=None
    )
    print("Critical point (MF):")
    print("rho_c =", rho_c_mf)
    print("T_c   =", T_c_mf)

    # Restore USE_MODEL for coexistence loops
    sol_bulk["USE_MODEL"] = True

    # Coexistence with neural model (USE_MODEL=True) - bulk mesh with BULK_COMP
    T_list = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
    rho_l_l, rho_v_l = [], []
    for T_val in T_list:
        model = update_model(x_=mesh_bulk["x"], rho_=rho_guess_bulk, T_=T_val, mesh_=mesh_bulk, sol_=sol_bulk, ml_state_dicts_=ml_state_dicts)
        model.sol["USE_MODEL"] = True
        print(f"T={1/model.eq_params['beta'].item():.2f}")
        rho_v, rho_l = solve_newton(
            coex_equations_auto,
            rho_init=torch.tensor([1e-15, 1 - 1e-15]).double().to(device),
            model=model,
            tol=1e-6,
            max_iter=100000,
            alpha=0.1,
            verbose=False,
        )
        print(f"rho_v = {rho_v.item()}, rho_l = {rho_l.item()}")
        rho_v_l.append(rho_v), rho_l_l.append(rho_l)

    # Coexistence without neural model (USE_MODEL=False) - use minimal bulk mesh for ~35x speedup
    T_list0 = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
    rho_l_l0, rho_v_l0 = [], []
    for T_val in T_list0:
        model = update_model(x_=mesh_bulk["x"], rho_=rho_guess_bulk, T_=T_val, mesh_=mesh_bulk, sol_=sol_bulk, ml_state_dicts_=ml_state_dicts)
        model.sol["USE_MODEL"] = False
        print(f"T={1/model.eq_params['beta'].item():.2f} (MF)")
        rho_v, rho_l = solve_newton(
            coex_equations_auto,
            rho_init=torch.tensor([1e-15, 1 - 1e-15]).double().to(device),
            model=model,
            tol=1e-6,
            max_iter=1000,
            alpha=0.05,
            verbose=False,
        )
        print(f"rho_v = {rho_v.item()}, rho_l = {rho_l.item()}")
        rho_v_l0.append(rho_v), rho_l_l0.append(rho_l)

    # Literature MD data
    T_md = [0.64, 0.67, 0.70, 0.73, 0.76, 0.79, 0.82, 0.85, 0.88, 0.91, 0.94, 0.97, 1.00, 1.03, 1.06]
    rho_l_md = [0.8176, 0.8024, 0.7866, 0.7704, 0.7538, 0.7361, 0.7181, 0.6986, 0.6784, 0.6556, 0.6309, 0.6032, 0.5712, 0.530, 0.463]
    rho_v_md = [0.00351, 0.00525, 0.00727, 0.01036, 0.01374, 0.01776, 0.0233, 0.0303, 0.0392, 0.0483, 0.0616, 0.0763, 0.096, 0.127, 0.168]
    T_md.append(1.085)
    rho_v_md.append(0.3170)
    rho_l_md.append(0.3170)

    # Build DataFrames
    pc_nn = pd.DataFrame(columns=["T", "rho_l", "rho_v"]).set_index(["T"])
    pc_0 = pd.DataFrame(columns=["T", "rho_l", "rho_v"]).set_index(["T"])
    pc_md = pd.DataFrame(columns=["T", "rho_l", "rho_v"]).set_index(["T"])

    for i, T_val in enumerate(T_list):
        pc_nn.at[(T_val), "rho_v"] = rho_v_l[i].item()
        pc_nn.at[(T_val), "rho_l"] = rho_l_l[i].item()
    for i, T_val in enumerate(T_list0):
        pc_0.at[(T_val), "rho_v"] = rho_v_l0[i].item()
        pc_0.at[(T_val), "rho_l"] = rho_l_l0[i].item()
    for i, T_val in enumerate(T_md):
        pc_md.at[(T_val), "rho_v"] = rho_v_md[i]
        pc_md.at[(T_val), "rho_l"] = rho_l_md[i]

    # Convert tensors to floats for plotting (ensures correct MF bulk display)
    rho_v_l0_plt = [r.item() if torch.is_tensor(r) else r for r in rho_v_l0]
    rho_l_l0_plt = [r.item() if torch.is_tensor(r) else r for r in rho_l_l0]
    rho_v_l_plt = [r.item() if torch.is_tensor(r) else r for r in rho_v_l]
    rho_l_l_plt = [r.item() if torch.is_tensor(r) else r for r in rho_l_l]

    # Plot: solid curves (excl. last 2 pts for MF to avoid overlap with dotted), dotted to critical
    plt.figure(figsize=(8, 6))
    plt.plot(rho_v_l0_plt[:-1], T_list0[:-1], color="blue")
    plt.plot(rho_l_l0_plt[:-1], T_list0[:-1], color="blue", label="rho - MF")
    plt.plot(rho_v_l_plt, T_list, color="red")
    plt.plot(rho_l_l_plt, T_list, color="red", label="rho - NN")
    plt.plot(rho_v_md[:-1], T_md[:-1], "--", color="black")
    plt.plot(rho_l_md[:-1], T_md[:-1], "--", color="black", label="rho - MD")
    # Dotted: connect coexistence curve to critical point (NN and MF)
    # NN: last point to critical point
    plt.plot([rho_v_l_plt[-1], rho_c_nn], [T_list[-1], T_c_nn], ":", color="red", linewidth=2, zorder=5)
    plt.plot([rho_l_l_plt[-1], rho_c_nn], [T_list[-1], T_c_nn], ":", color="red", linewidth=2, zorder=5)
    # MF: last 2 points + critical point (longer dotted segment)
    plt.plot([rho_v_l0_plt[-2], rho_v_l0_plt[-1], rho_c_mf], [T_list0[-2], T_list0[-1], T_c_mf], ":", color="blue", linewidth=2, zorder=5)
    plt.plot([rho_l_l0_plt[-2], rho_l_l0_plt[-1], rho_c_mf], [T_list0[-2], T_list0[-1], T_c_mf], ":", color="blue", linewidth=2, zorder=5)
    # MD: last point is the critical point
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

    # Save pickle files
    pc_nn.to_pickle(os.path.join(output_dir, "pc_nn_operative.pkl"))
    pc_md.to_pickle(os.path.join(output_dir, "pc_md_operative.pkl"))
    pc_0.to_pickle(os.path.join(output_dir, "pc_0_operative.pkl"))

    # Save critical points
    with open(os.path.join(output_dir, "critical_point.txt"), "w") as f:
        f.write(f"NN: rho_c = {rho_c_nn}\nNN: T_c   = {T_c_nn}\n")
        f.write(f"MF: rho_c = {rho_c_mf}\nMF: T_c   = {T_c_mf}\n")

    print(f"\nOutputs saved to {os.path.abspath(output_dir)}")
    print("\npc_nn (neural model):")
    print(pc_nn)
