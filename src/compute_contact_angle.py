import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.normpath(os.path.join(script_dir, ".."))
sys.path.insert(0, script_dir)

import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd

from libs.omega_minimization.utils import *
from libs.omega_minimization.spherical_cap.macro_omega_3D import Omega as OmegaCap
from libs.omega_minimization.spherical_cap.macro_omega_3D import R_constrained as R_constrainedCap



##---------- MAIN ----------##

plotdir = os.path.join(repo_root, "output", "plot_contact_angle")
for sub in ("", "sanitycheck", "drop", "dropPlan"):
    os.makedirs(os.path.join(plotdir, sub), exist_ok=True)
T_list = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,]
MD_DROP_DIR = "md_drop_wc_2"

# -----------------------------------------------------------
PHASE_CURVE_CANDIDATES = [
    os.path.join(repo_root, "data", "dataset", "pkl", "bulk_tmd", "pc_md_operative_with_gamma.pkl"),
    os.path.join(repo_root, "data", "dataset", "postproc", "phase_curve_nn.pkl"),
]
pc_path = next((p for p in PHASE_CURVE_CANDIDATES if os.path.isfile(p)), None)
if pc_path is None:
    raise FileNotFoundError(
        "Phase curve pickle not found. Tried:\n  " + "\n  ".join(PHASE_CURVE_CANDIDATES)
    )
print(f"Loading phase curve from: {pc_path}")
with open(pc_path, "rb") as f:
    pc_nn = pkl.load(f)
print(f"Loaded data with {len(pc_nn)} samples. \n\nPC datas are:")
print(pc_nn)
print(f"\n\nUpdate the datas with the work adsorption isotherm data. \n\nPC updated datas are:")
upload_gammas_in_pc_(pc_nn, T_list, plotpath=plotdir+"/gammas_vs_T.png")
print(pc_nn)
print(f"\n\nPlotting the thetaY in:", plotdir+"/thetaY_vs_T.png")
tYc = [np.arccos(np.clip((pc_nn.loc[(T_j), "gamma_wg"] - pc_nn.loc[(T_j), "gamma_wl"])/pc_nn.loc[(T_j), "gamma_lg"], -1.0, 1.0))    *180/np.pi for T_j in T_list]
plt.plot(T_list, tYc, "o-",label=r"$\theta_Y$"), plt.xlabel("T")
plt.grid(); plt.legend()
plt.savefig(plotdir+"/thetaY_vs_T.png"); plt.close()


# -----------------------------------------------------------
# Upload work adsorption regressors and perform sanity checks

l = torch.linspace(0.0, 40, 1000)[...,None]
for i, T_i in enumerate(T_list):
    W_fn, dW_fn, W_nn = compute_W_fn(T_i)
    muiso_fn, dmuiso_fn, mu0_nn = compute_mu0_fn(T_i)
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


# Evaluate Omega 
# -----------------------------------------------------------
# Upload md data
for T_i in T_list:
    md_csv = os.path.join(repo_root, "data", "dataset", MD_DROP_DIR, f"rho_field_T{T_i:.2f}.csv")
    bins_csv = os.path.join(repo_root, "data", "dataset", MD_DROP_DIR, "bins.csv")
    if not os.path.isfile(md_csv):
        print(f"MD drop field not found ({md_csv}); skipping T={T_i:.2f}.")
        continue
    data = np.loadtxt(md_csv, delimiter=",", skiprows=1)  # skip header
    bins = np.loadtxt(bins_csv, delimiter=",")  
    Nbins = (int(bins[0]), int(bins[1]))
    tensor = torch.from_numpy(data)
    eta = tensor[:,0].reshape(Nbins)
    psi = tensor[:,1].reshape(Nbins)
    rho = tensor[:,2].reshape(Nbins)
    deta = eta[0,1] - eta[0,0]
    dpsi = psi[1,0] - psi[0,0]
    Lr = eta.max() - eta.min()
    Lz = psi.max() - psi.min()
    N_MD = (rho*2*np.pi*eta).sum() * deta*dpsi 


    # -----------------------------------------------------------
    # Initialize parameters
    gamma_lg = pc_nn.loc[(T_i),"gamma_lg"]        # <-- NOTE: no trailing comma
    gamma_wl = pc_nn.loc[(T_i),"gamma_wl"]        # <-- NOTE: no trailing comma
    gamma_wg = pc_nn.loc[(T_i),"gamma_wg"]        # <-- NOTE: no trailing comma
    rho_v = pc_nn.loc[(T_i),"rho_v"]        # <-- NOTE: no trailing comma
    rho_l = pc_nn.loc[(T_i),"rho_l"]        # <-- NOTE: no trailing comma
    Winf = pc_nn.loc[(T_i),"Winf"]        # <-- NOTE: no trailing comma
    Wmax = pc_nn.loc[(T_i),"Wmax"]        # <-- NOTE: no trailing comma
    Wmin = pc_nn.loc[(T_i),"Wmin"]        # <-- NOTE: no trailing comma
    l_max = pc_nn.loc[(T_i),"l_max"]        # <-- NOTE: no trailing comma
    l_min = pc_nn.loc[(T_i),"l_min"]        # <-- NOTE: no trailing comma
    W_fn, dW_fn, W_nn = compute_W_fn(T_i)
    mu0_fn, dmu0_fn, mu0_nn = compute_mu0_fn(T_i)
    mu0_inf = mu0_fn(torch.tensor(mu0_nn.validity).double().unsqueeze(0)).squeeze().item()

    params = {
        "T" : T_i,
        "gamma_lg" : gamma_lg,
        "gamma_wg" : gamma_wg,
        "gamma_wl" : gamma_wl,
        "rho_l" : rho_l,
        "rho_v" : rho_v,
        "Npar" : N_MD,
        "yb" : 1.4,
        "W_fn" : W_fn,
        "mu0_fn" : mu0_fn,
        "mu0_infty" : mu0_inf,
        "Winf" : Winf,
        "Wmin" : Wmin,
        "Wmax" : Wmax,
        "l_max" : l_max,
        "l_min" : l_min,
        "Lr" : Lr,
        "Lz" : Lz,
        "Vbox" :  (np.pi*Lr**2) * (Lz),
    }


    scale = 0.99
    t0_grid = torch.linspace(-float(scale*torch.pi/2), float(scale*torch.pi/2), 1000, dtype=torch.float64)
    OmCap = OmegaCap(
        t0_=t0_grid, 
        R_=None, 
        params=params,
        )
    OmCmin, OmCMin_i = torch.min(OmCap, dim=-1)
    t0Min = t0_grid[OmCMin_i]
    t0Y = torch.pi/2 - np.arccos(np.clip((gamma_wg - gamma_wl)/gamma_lg, -1.0, 1.0))
    OmY = OmegaCap(
        t0_=torch.tensor([t0Y], dtype=torch.float64),
        R_=None,
        params=params,
        ).squeeze()  # Omega at the Young angle, so the t0Y marker sits on the curve
    Rt0 = R_constrainedCap(t0Min, params)
    print(f"Minimum Omega_Cap obtained at: t0 = ({t0Min.item():.4f} rad, ({t0Min.item()*180/np.pi:.2f} deg), t0Y={t0Y.item()*180/np.pi:.2f} deg, R={Rt0.item():.4e}, h-zb={(Rt0*(1 - torch.sin(t0Min))).item():.4e}")
    H0 = (l_max)
    height = torch.linspace(1, 40, 100) 
    A_ref = torch.pi*Lr**2
    DeltaN = params["Npar"] - rho_v * params["Vbox"]
    DeltaRho = params["rho_l"] - params["rho_v"]
    radius = torch.sqrt( DeltaN / (1e-6 + torch.pi*height*DeltaRho))
    A_r = torch.pi * radius**2
    W_h = W_fn(H0 + height.unsqueeze(-1))[...,0]
    #plt.plot(t0_grid.detach().numpy(), OmCap.detach().numpy(), label="Omega Cap")
    #plt.scatter(t0Min.detach().numpy(), OmCmin.detach().numpy(), color="red", label="Min")
    #plt.scatter(t0Y, OmY.detach().numpy(), marker="x", color="black", label="t0Y")
    #plt.legend(); plt.grid(); plt.savefig(plotdir+f"/drop/Omega_vs_t0_{T_i:.2f}.png")
    #plt.savefig(plotdir+f"/drop/Omega_vs_t0_temp.png")
    #plt.close()

    # --- Geometry for plotting the arc ---
    th_hi = 0.5 * torch.pi
    Nplot = 1000
    thetas = torch.linspace(t0Min, th_hi, Nplot, dtype=torch.float64)
    zcMin = params["yb"] - Rt0 * torch.sin(t0Min)
    aMin = Rt0 * torch.cos(t0Min)
    plt.contourf(eta.numpy(), psi.numpy(), rho.numpy(), levels=200, cmap='rainbow',)
    plt.clim(0., 1.2)
    plt.colorbar(label='rho_MD')
    circle = torch.tensor([0.0, zcMin], dtype=torch.float64)[:, None] + \
                torch.as_tensor(Rt0) * torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=0)
    theta_c = (np.pi/2 - t0Min)*180./np.pi
    plt.plot(circle[0].numpy(), circle[1].numpy(), linewidth=1.5, linestyle='-.', color="black", label='Laplace equation')
    plt.hlines([params["yb"]], 0, aMin, linewidth=1.5, linestyle='-.', color="black",)
    plt.title(f'3D Droplet at T={T_i:.2f}, N={int(N_MD)}')
    plt.legend()
    plt.xlim( 0, Lr)
    plt.savefig(plotdir+f"/drop/Drop.png")
    plt.savefig(plotdir+f"/drop/Drop_T{T_i:.2f}.png")
    plt.close()
    print("\n\n")
