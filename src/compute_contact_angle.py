#!/usr/bin/env python3
"""
Apparent contact angle of the MD droplets from planar adsorption (spherical-cap construction).

1. Work of adsorption W(l) and film thickness l of every frame of the wall-liquid continuation
   isotherms written by adsorption_continuation.py, and the W(l), mu0(l) regressors
   (libs/omega_minimization/adsorption_work.py; written to data/dataset/pkl/profiles_wl_<wall>/).
2. Surface tensions gamma_wl = W_inf - gamma_lg, gamma_wg = W at the wall-gas state, Young angle.
3. Minimization of the excess free energy of a spherical cap at the (N, T) of each MD droplet
   (the cap radius follows from the volume constraint): theta_c = pi/2 - t0*, compared with the
   spherical-cap fit of the MD field (data/dataset/md_drop_<set>/cap_fit.csv).
4. Free-energy landscapes at fixed N and theta_c as a function of T and N.

Requires data/dataset/pkl/bulk_tmd/pc_md_operative_with_gamma.pkl (compute_gamma_lg.py) and the
isotherms of the wall (adsorption_continuation.py with the same WALL).
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.normpath(os.path.join(script_dir, ".."))
sys.path.insert(0, script_dir)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd

from libs.omega_minimization.adsorption_work import compute_adsorption_work
from libs.omega_minimization.utils import *
from libs.omega_minimization.spherical_cap.macro_omega_3D import Omega as OmegaCap
from libs.omega_minimization.spherical_cap.macro_omega_3D import R_constrained as R_constrainedCap


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
WALL = "wc"  # "wc" (V2) or "wn2" (V1): isotherms, regressors and MD droplet series
MD_DROP_DIR = {"wc": "md_drop_wc_2", "wn2": "md_drop_wn2"}[WALL]
pkldir = os.path.join(repo_root, "data", "dataset", "pkl", f"profiles_wl_{WALL}")
# Isotherms of adsorption_continuation.py: one pickle, or one per temperature when run with T_VALS
ISOTHERM_PKL = os.path.join(pkldir, f"isotherm_profiles_wl_{WALL}*.pkl")
PHASE_CURVE_PKL = os.path.join(repo_root, "data", "dataset", "pkl", "bulk_tmd", "pc_md_operative_with_gamma.pkl")
N_VALUE = 12000  # particle number (per 20 x 20 wall area) at the end of the continuation

plotdir = os.path.join(repo_root, "output", f"plot_contact_angle_{WALL}")
plotdir_work = os.path.join(repo_root, "output", f"plot_adsorption_work_wl_{WALL}")
for sub in ("", "sanitycheck", "drop"):
    os.makedirs(os.path.join(plotdir, sub), exist_ok=True)
T_list = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,]
KSIG = 3  # error bars of the MD contact angle, in sigma
# Droplet sizes for the free-energy landscapes and the theta_c(T, N) map
N_bar = 48500.
N_list = np.pi ** np.linspace(10, 20, 10)


# ------------------------------------------------------------------------------
# 1. Work of adsorption and regressors
# ------------------------------------------------------------------------------
compute_adsorption_work(WALL, ISOTHERM_PKL, PHASE_CURVE_PKL, pkldir, plotdir_work, n_value=N_VALUE)


# ------------------------------------------------------------------------------
# 2. Surface tensions and Young angle
# ------------------------------------------------------------------------------
with open(PHASE_CURVE_PKL, "rb") as f:
    pc_nn = pkl.load(f)
print(f"Loaded data with {len(pc_nn)} samples. \n\nPC datas are:")
print(pc_nn)
print(f"\n\nUpdate the datas with the work adsorption isotherm data. \n\nPC updated datas are:")
upload_gammas_in_pc_(pc_nn, T_list, plotpath=plotdir+"/gammas_vs_T.png", wall=WALL)
print(pc_nn)
print(f"\n\nPlotting the thetaY in:", plotdir+"/thetaY_vs_T.png")
def young_angle(T_j):
    cosY = (pc_nn.loc[(T_j), "gamma_wg"] - pc_nn.loc[(T_j), "gamma_wl"])/pc_nn.loc[(T_j), "gamma_lg"]
    return np.arccos(cosY)*180/np.pi if abs(cosY) <= 1. else np.nan  # nan above the wetting temperature
tYc = [young_angle(T_j) for T_j in T_list]
plt.plot(T_list, tYc, "o-",label=r"$\theta_Y$"), plt.xlabel("T")
plt.grid(); plt.legend()
plt.savefig(plotdir+"/thetaY_vs_T.png"); plt.close()


# -----------------------------------------------------------
# Upload work adsorption regressors and perform sanity checks

l = torch.linspace(0.0, 40, 1000)[...,None]
for i, T_i in enumerate(T_list):
    W_fn, dW_fn, W_nn = compute_W_fn(T_i, wall=WALL)
    muiso_fn, dmuiso_fn, mu0_nn = compute_mu0_fn(T_i, wall=WALL)
    with torch.no_grad():
        W_eval= W_nn(l)                # forward
        muiso_eval = mu0_nn(l)                # forward
    dW_eval = W_nn.dy_dx(l).detach()       
    dmuiso_eval = mu0_nn.dy_dx(l).detach()       

    plt.plot(l, W_eval, label="W_eval(l)")
    plt.plot(l, W_fn(l).detach(), "--", label="W(l)")
    plt.legend(); plt.grid(); plt.savefig(plotdir+f"/sanitycheck/W_vs_l_T{T_i:.2f}.png")
    plt.close()

    plt.plot(l, muiso_eval, label="muiso_eval(l)")
    plt.plot(l, muiso_fn(l).detach(), "--", label="muiso(l)")
    plt.legend(); plt.grid(); plt.savefig(plotdir+f"/sanitycheck/mu0_vs_l_T{T_i:.2f}.png")
    plt.close()


# -----------------------------------------------------------
# Parameters of the cap free energy and its minimization over t0 (theta_c = pi/2 - t0)
def cap_params(T_i, Npar, Lr, Lz):
    W_fn, dW_fn, W_nn = compute_W_fn(T_i, wall=WALL)
    mu0_fn, dmu0_fn, mu0_nn = compute_mu0_fn(T_i, wall=WALL)
    mu0_inf = mu0_fn(torch.tensor(mu0_nn.validity).double().unsqueeze(0)).squeeze().item()
    params = {
        "T" : T_i,
        "gamma_lg" : pc_nn.loc[(T_i),"gamma_lg"],
        "gamma_wg" : pc_nn.loc[(T_i),"gamma_wg"],
        "gamma_wl" : pc_nn.loc[(T_i),"gamma_wl"],
        "rho_l" : pc_nn.loc[(T_i),"rho_l"],
        "rho_v" : pc_nn.loc[(T_i),"rho_v"],
        "Npar" : Npar,
        "yb" : 1.4,
        "W_fn" : W_fn,
        "mu0_fn" : mu0_fn,
        "mu0_infty" : mu0_inf,
        "Winf" : pc_nn.loc[(T_i),"Winf"],
        "Wmin" : pc_nn.loc[(T_i),"Wmin"],
        "Wmax" : pc_nn.loc[(T_i),"Wmax"],
        "l_max" : pc_nn.loc[(T_i),"l_max"],
        "l_min" : pc_nn.loc[(T_i),"l_min"],
        "Lr" : Lr,
        "Lz" : Lz,
        "Vbox" :  (np.pi*Lr**2) * (Lz),
    }
    return params

def minimize_omega(params, scale=0.98, n_grid=2001):
    t0_grid = torch.linspace(-float(scale*torch.pi/2), float(scale*torch.pi/2), n_grid, dtype=torch.float64)
    OmCap = OmegaCap(t0_=t0_grid, R_=None, params=params)
    i = int(torch.argmin(OmCap))
    if i == 0 or i == n_grid - 1:
        print(f"T={params['T']:.2f}, N={params['Npar']:.0f}: the minimum of Omega is at the edge of the t0 grid")
    # Refine around the minimum
    dt = float(t0_grid[1] - t0_grid[0])
    t0_fine = torch.linspace(max(float(t0_grid[i]) - 2*dt, float(t0_grid[0])), min(float(t0_grid[i]) + 2*dt, float(t0_grid[-1])), 401, dtype=torch.float64)
    OmFine = OmegaCap(t0_=t0_fine, R_=None, params=params)
    j = int(torch.argmin(OmFine))
    return t0_grid, OmCap, t0_fine[j], OmFine[j]


# -----------------------------------------------------------
# MD droplets: N, box and spherical-cap fit of the MD density fields
md = pd.read_csv(os.path.join(repo_root, "data", "dataset", MD_DROP_DIR, "cap_fit.csv")).set_index("T")
md.index = md.index.round(2)

records = []
for T_i in T_list:
    if T_i not in md.index:
        print(f"No MD droplet at T={T_i:.2f}; skipping.")
        continue
    Lr, Lz, N_MD = md.loc[T_i, "Lr"], md.loc[T_i, "Lz"], md.loc[T_i, "N"]
    params = cap_params(T_i, N_MD, Lr, Lz)
    t0_grid, OmCap, t0Min, OmCmin = minimize_omega(params)
    Rt0 = R_constrainedCap(t0Min, params)
    theta_c = (np.pi/2 - t0Min.item())*180./np.pi
    theta_Y = young_angle(T_i)
    print(f"T={T_i:.2f}: theta_c = {theta_c:.2f} deg (t0 = {t0Min.item():.4f} rad), theta_Y = {theta_Y:.2f} deg, "
          f"theta_MD = {md.loc[T_i, 'theta_md']:.2f} +- {md.loc[T_i, 'sigma_md']:.2f} deg, R = {Rt0.item():.2f}, "
          f"h-zb = {(Rt0*(1 - torch.sin(t0Min))).item():.2f}")
    records.append({
        "T": T_i, "N": N_MD, "theta_c": theta_c, "theta_Y": theta_Y,
        "theta_md": md.loc[T_i, "theta_md"], "sigma_md": md.loc[T_i, "sigma_md"], "R": Rt0.item(),
        "gamma_lg": params["gamma_lg"], "gamma_wl": params["gamma_wl"], "gamma_wg": params["gamma_wg"],
    })

    # --- MD density field with the predicted cap ---
    md_csv = os.path.join(repo_root, "data", "dataset", MD_DROP_DIR, f"rho_field_T{T_i:.2f}.csv")
    if not os.path.isfile(md_csv):
        continue
    df_md = pd.read_csv(md_csv)
    eta, psi = np.unique(df_md["r"].values), np.unique(df_md["z"].values)
    rho = df_md["rho"].values.reshape(len(psi), len(eta))  # (nz, nr)
    th_hi = 0.5 * torch.pi
    Nplot = 1000
    thetas = torch.linspace(t0Min, th_hi, Nplot, dtype=torch.float64)
    zcMin = params["yb"] - Rt0 * torch.sin(t0Min)
    aMin = Rt0 * torch.cos(t0Min)
    plt.contourf(eta, psi, rho, levels=200, cmap='rainbow',)
    plt.clim(0., 1.2)
    plt.colorbar(label='rho_MD')
    circle = torch.tensor([0.0, zcMin], dtype=torch.float64)[:, None] + \
                torch.as_tensor(Rt0) * torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=0)
    plt.plot(circle[0].numpy(), circle[1].numpy(), linewidth=1.5, linestyle='-.', color="black", label=f'cDFT cap, {theta_c:.1f} deg')
    plt.hlines([params["yb"]], 0, aMin, linewidth=1.5, linestyle='-.', color="black",)
    plt.title(f'3D Droplet at T={T_i:.2f}, N={int(N_MD)} (MD: {md.loc[T_i, "theta_md"]:.1f} deg)')
    plt.legend()
    plt.xlim( 0, Lr)
    plt.savefig(plotdir+f"/drop/Drop_T{T_i:.2f}.png")
    plt.close()

df = pd.DataFrame(records).set_index("T")
df.to_csv(plotdir+"/theta_c.csv")
err = df["theta_c"] - df["theta_md"]
print(f"\ntheta_c vs MD on {WALL}: MAE = {err.abs().mean():.2f} deg, RMSE = {np.sqrt((err**2).mean()):.2f} deg")
print(df)

plt.errorbar(df.index, df["theta_md"], yerr=KSIG*df["sigma_md"], fmt="ko", capsize=3, label=f"MD ({KSIG}$\\sigma$)")
plt.plot(df.index, df["theta_c"], "rs-", label=r"cDFT $\theta_c$")
plt.plot(df.index, df["theta_Y"], "b--", label=r"$\theta_Y$")
plt.xlabel("T"); plt.ylabel("contact angle (deg)")
plt.title(f"{WALL}: MAE = {err.abs().mean():.1f} deg, RMSE = {np.sqrt((err**2).mean()):.1f} deg")
plt.legend(); plt.grid()
plt.savefig(plotdir+"/theta_vs_T.png"); plt.close()


# -----------------------------------------------------------
# Free-energy landscapes at fixed N and theta_c as a function of T and N
Lr, Lz = md["Lr"].mean(), md["Lz"].mean()
T_map = [T_i for T_i in T_list if T_i in md.index]
plt.figure(figsize=(6, 4.5))
for T_i in T_map:
    params = cap_params(T_i, N_bar, Lr, Lz)
    t0_grid, OmCap, t0Min, OmCmin = minimize_omega(params)
    theta_grid = 90. - t0_grid.numpy()*180/np.pi
    line, = plt.plot(theta_grid, OmCap.numpy(), label=f"T={T_i:.2f}")
    plt.plot(90. - t0Min.item()*180/np.pi, OmCmin.item(), "x", color=line.get_color(), markersize=8)
    theta_Y = young_angle(T_i)
    if np.isfinite(theta_Y):
        OmY = OmegaCap(t0_=torch.tensor([np.pi/2 - theta_Y*np.pi/180], dtype=torch.float64), R_=None, params=params)
        plt.plot(theta_Y, OmY.item(), "o", color=line.get_color(), markersize=4)
plt.xlabel(r"$\vartheta$ (deg)"); plt.ylabel(r"$\Delta F / A$")
plt.title(f"{WALL}: N = {N_bar:.0f} (x: minimum, o: Young angle)")
plt.legend(fontsize=7); plt.grid()
plt.savefig(plotdir+"/dF_landscape.png"); plt.close()

theta_TN = pd.DataFrame(index=T_map, columns=[f"{N_j:.0f}" for N_j in N_list], dtype=float)
for T_i in T_map:
    for N_j in N_list:
        params = cap_params(T_i, float(N_j), Lr, Lz)
        t0_grid, OmCap, t0Min, OmCmin = minimize_omega(params)
        theta_TN.loc[T_i, f"{N_j:.0f}"] = 90. - t0Min.item()*180/np.pi
theta_TN.to_csv(plotdir+"/theta_c_T_N.csv")
for col in theta_TN.columns:
    plt.plot(theta_TN.index, theta_TN[col], "o-", markersize=3, label=f"N={float(col):.2e}")
plt.plot(T_map, [young_angle(T_i) for T_i in T_map], "k--", label=r"$\theta_Y$")
plt.xlabel("T"); plt.ylabel(r"$\theta_c$ (deg)")
plt.title(f"{WALL}: apparent contact angle vs droplet size")
plt.legend(fontsize=6); plt.grid()
plt.savefig(plotdir+"/theta_vs_T_N.png"); plt.close()
print("Done.")
