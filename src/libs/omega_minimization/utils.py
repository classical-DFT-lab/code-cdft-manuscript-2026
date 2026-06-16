import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle as pkl

from libs.omega_minimization.spherical_cap.macro_omega_3D import Omega as OmegaCap
from libs.omega_minimization.spherical_cap.macro_omega_3D import R_constrained as R_constrainedCap


# ---------------------------------------------------------------------------
# Regressor locations: new layout written by src/compute_adsorption_work.py,
# with fallback to the legacy postproc/ layout (retro-compatibility with the
# old notebooks and any cwd-relative invocation from src/).
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
)

_W_REGRESSOR_DIRS = [
    os.path.join(_REPO_ROOT, "data", "dataset", "pkl", "profiles_wl_wc", "work_regressor_wl_wc"),
    os.path.join(_REPO_ROOT, "data", "dataset", "postproc", "work_regressor_Xb"),
    "../data/dataset/postproc/work_regressor_Xb",
]
_MU0_REGRESSOR_DIRS = [
    os.path.join(_REPO_ROOT, "data", "dataset", "pkl", "profiles_wl_wc", "mu0_regressor_wl_wc"),
    os.path.join(_REPO_ROOT, "data", "dataset", "postproc", "mu0_regressor"),
    "../data/dataset/postproc/mu0_regressor",
]


def _find_regressor(dirs, fname):
    for d in dirs:
        path = os.path.join(d, fname)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"{fname} not found in any of: {dirs}")


def load_work_regressor(T_star):
    path = _find_regressor(_W_REGRESSOR_DIRS, f"work_regressor_{T_star:.2f}.pt")
    return torch.jit.load(path), path


def load_mu0_regressor(T_star):
    path = _find_regressor(_MU0_REGRESSOR_DIRS, f"mu0_regressor_{T_star:.2f}.pt")
    return torch.jit.load(path), path


def upload_gammas_in_pc_(pc_nn, T_list, plotpath=None):
    gamma_wl_list, gamma_lg_list, gamma_wg_list = [], [], []
    for i, T_i in enumerate(T_list):
        W_fn, w_path = load_work_regressor(T_i)
        print("Loaded W regressor for T =", T_i, "from path:", w_path)
        Wmin = W_fn.min["W_min"]
        # Winf is stored by the fitting script at l = validity; evaluating the
        # spline mid-window (0.5*validity) underestimated the plateau.
        Winf = float(W_fn.Winf)
        W0=W_fn(torch.tensor([0.]).double()).squeeze().item()
        
        
        gamma_wg = W0
        #gamma_wg = Wmin
        #gamma_wg = W_fn.wg["W_wg"].item()

        l_max, Wmax = W_fn.max["l_max"], W_fn.max["W_max"]
        l_min, Wmin = W_fn.min["l_min"], W_fn.min["W_min"]
        # l_max = W_fn.max["l_max"]
        # l_min = W0
        gamma_lg = pc_nn.loc[(T_i), "gamma_lg"]
        gamma_wl = Winf - gamma_lg 

        pc_nn.loc[(T_i), "gamma_lg"] = gamma_lg
        pc_nn.loc[(T_i), "gamma_wg"] = gamma_wg
        pc_nn.loc[(T_i), "gamma_wl"] = gamma_wl
        pc_nn.loc[(T_i), "l_min"] = l_min
        pc_nn.loc[(T_i), "l_max"] = l_max
        pc_nn.loc[(T_i), "Winf"] = Winf
        pc_nn.loc[(T_i), "Wmax"] = Wmax
        pc_nn.loc[(T_i), "Wmin"] = Wmin
        gamma_wl_list.append(gamma_wl), gamma_lg_list.append(gamma_lg), gamma_wg_list.append(gamma_wg)
        print(l_min)
        # DW = Winf
    plt.plot(T_list, gamma_lg_list, "x-", label="gamma_lg")
    plt.plot(T_list, gamma_wl_list, "x-", label="gamma_wl")
    plt.plot(T_list, gamma_wg_list, "x-", label="gamma_wg")
    plt.grid(); plt.legend()
    if plotpath is not None:
        print(f"\n\nPlotting the gammas in:", plotpath,)
        plt.savefig(plotpath); 
    plt.close()

shift0 =0  # shift in l for W and mu0 functions
def compute_W_fn(T_star):
    W_nn, _ = load_work_regressor(T_star)
    W_nn.eval()

    lmax = W_nn.max["l_max"]
    lmin, Wmin, Winf = W_nn.min["l_min"], W_nn.min["W_min"], W_nn.Winf
    shift = lmax
    shift = lmin 
    shift = shift0

    # print(W_nn)

    # print(f"Min l* ≈ {lmin:.6f},  W(l*) ≈ {Wmin:.6f},  W(inf) ≈ {Winf:.6f}")

    # =========================
    # 1) Define W and W_nn
    # =========================
    # Define W(y) and its derivative
    def W_fn(y):
        """
        y: (...,1) tensor
        mu: (...,1) tensor
        """
        y = y + shift
        if y.shape[-1] != 1:
            raise ValueError("y must be of shape (...,1)")

        cutoff = 0.8*W_nn.validity
        mask = torch.sigmoid( -(y - cutoff)/0.5 )  # smooth mask    
        m = 0.  # slope at the cut-off
        q = W_nn(torch.tensor([cutoff]).double()) - m * cutoff  # intercept at the cut-off
        line = m * y + q
        return W_nn(y)*mask + line*(1 - mask)
        

    def dW_fn(y): 
        """
        y: (...,1) tensor
        mu: (...,1) tensor
        """
        y = y + shift  # keep consistent with W_fn when shift != 0
        if y.shape[-1] != 1:
            raise ValueError("y must be of shape (...,1)")
        cutoff = 0.8*W_nn.validity
        m = 0.  # slope at the cut-off
        mask = torch.sigmoid( -(y - cutoff)/0.5 )  # smooth mask    
        return W_nn.dy_dx(y)*mask + m*(1 - mask)

    return W_fn, dW_fn, W_nn

def compute_mu0_fn(T_star):
    mu0_nn, _ = load_mu0_regressor(T_star)
    W_nn, _ = load_work_regressor(T_star)
    mu0_nn.eval()
    shift = W_nn.min["l_min"]
    shift = shift0

    # shift = W_nn.max["l_max"]*0


    # =========================
    # 1) Define muiso and mu0_nn
    # =========================
    # Define W(y) and its derivative
    def mu0_fn(y):
        """
        y: (...,1) tensor
        mu: (...,1) tensor
        """
        y = y + shift

        if y.shape[-1] != 1:
            raise ValueError("y must be of shape (...,1)")

        cutoff = 0.8*mu0_nn.validity
        mask = torch.sigmoid( -(y - cutoff)/0.5 )  # smooth mask    
        m = 0.  # slope at the cut-off
        q = mu0_nn(torch.tensor([cutoff]).double()) - m * cutoff  # intercept at the cut-off
        line = m * y + q
        return mu0_nn(y)*mask + line*(1 - mask)
        

    def dmu0_fn(y): 
        """
        y: (...,1) tensor
        mu: (...,1) tensor
        """
        y = y + shift  # keep consistent with mu0_fn when shift != 0
        if y.shape[-1] != 1:
            raise ValueError("y must be of shape (...,1)")
        cutoff = 0.8*mu0_nn.validity
        m = 0.  # slope at the cut-off
        mask = torch.sigmoid( -(y - cutoff)/0.5 )  # smooth mask    
        return mu0_nn.dy_dx(y)*mask + m*(1 - mask)
    
    return mu0_fn, dmu0_fn, mu0_nn



def load_md_data(path, Nbins=(200, 200)):
    """Load an MD drop density field rho(eta, psi) from a CSV file."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)  # skip header
    tensor = torch.from_numpy(data).to(torch.float64)
    eta = tensor[:,0].reshape(Nbins)
    psi = tensor[:,1].reshape(Nbins)
    rho = tensor[:,2].reshape(Nbins)
    deta = eta[0,1] - eta[0,0]
    dpsi = psi[1,0] - psi[0,0]
    Lr = eta.max() - eta.min()
    Lz = psi.max() - psi.min()
    # Cylindrical-shell weight 2*pi*eta (same convention as the main scripts)
    N_MD = (rho*2*np.pi*eta).sum() * deta*dpsi
    return {
        "eta": eta, "psi": psi, "rho": rho,
        "deta": deta, "dpsi": dpsi,
        "Lr": Lr, "Lz": Lz, "N_MD": N_MD,
    }



def min_omega(T_i, N_j, pc_nn, use_binding, plotdir):
    # -----------------------------------------------------------
    # # Upload md data
    # data = np.loadtxt(f"../data/dataset/md_drop/rho_field_T{T_i:.2f}.csv", delimiter=",", skiprows=1)  # skip header
    # Nbins = (200, 200)
    # # data = np.loadtxt(f"../data/dataset/md_drop/rho_field.old.csv", delimiter=",", skiprows=1)  # skip header
    # # Nbins = (100, 100)
    # tensor = torch.from_numpy(data).to(torch.float64)
    # eta = tensor[:,0].reshape(Nbins)
    # psi = tensor[:,1].reshape(Nbins)
    # rho = tensor[:,2].reshape(Nbins)
    # deta = eta[0,1] - eta[0,0]
    # dpsi = psi[1,0] - psi[0,0]
    # # eta_cc = 0.5*(eta[:1,:] + eta[:-1,:])
    # Lr = eta.max() - eta.min()
    # Lz = psi.max() - psi.min()
    # N_MD = (rho*2*np.pi*eta).sum() * deta*dpsi 


    # -----------------------------------------------------------
    # Initialize parameters
    gamma_lg = pc_nn.loc[(T_i),"gamma_lg"]        # <-- NOTE: no trailing comma
    gamma_wl = pc_nn.loc[(T_i),"gamma_wl"]        # <-- NOTE: no trailing comma
    gamma_wg = pc_nn.loc[(T_i),"gamma_wg"]        # <-- NOTE: no trailing comma
    rho_v = pc_nn.loc[(T_i),"rho_v"]        # <-- NOTE: no trailing comma
    rho_l = pc_nn.loc[(T_i),"rho_l"]        # <-- NOTE: no trailing comma
    Winf = pc_nn.loc[(T_i),"Winf"]        # <-- NOTE: no trailing comma
    Wmin = pc_nn.loc[(T_i),"Wmin"]
    Wmax = pc_nn.loc[(T_i),"Wmax"]
    l_min = pc_nn.loc[(T_i),"l_min"]
    l_max = pc_nn.loc[(T_i),"l_max"]
    W_fn, dW_fn, W_nn = compute_W_fn(T_i)
    mu0_fn, dmu0_fn, mu0_nn = compute_mu0_fn(T_i)
    mu0_inf = mu0_fn(torch.tensor(mu0_nn.validity).double().unsqueeze(0)).squeeze().item()
    Lr=60
    Lz=60
    params = {
        "T" : T_i,
        "gamma_lg" : gamma_lg,
        "gamma_wg" : gamma_wg,
        "gamma_wl" : gamma_wl,
        "rho_l" : rho_l,
        "rho_v" : rho_v,
        "Npar" : N_j,
        "yb" : 1.4,
        "W_fn" : W_fn if use_binding else None,
        "mu0_fn" : mu0_fn,
        "mu0_infty" : mu0_inf,
        "Winf" : Winf,
        "Wmin" : Wmin,
        "Wmax" : Wmax,
        "l_min" : l_min,
        "l_max" : l_max,
        "Lr" : Lr,
        "Lz" : Lz,
        # "mu0_infty" : None,
        # "Npar" : 0.7321*N_MD,
        "Vbox" :  (np.pi*Lr**2) * (Lz),
    }

    scale = 0.95
    t0_grid = torch.linspace(-float(scale*torch.pi/2), float(scale*torch.pi/2), 3000, dtype=torch.float64)
    OmCap = OmegaCap(
        t0_=t0_grid, 
        R_=None, 
        params=params,
        )

    OmCmin, OmCMin_i = torch.min(OmCap, dim=-1)
    t0Min = t0_grid[OmCMin_i]
    cosY = np.clip((gamma_wg - gamma_wl)/gamma_lg, -1.0, 1.0)
    t0Y = torch.pi/2 - np.arccos(cosY)
    # Omega at the Young angle (for plotting the t0Y marker on the curve)
    OmY = OmegaCap(
        t0_=torch.tensor([t0Y], dtype=torch.float64),
        R_=None,
        params=params,
        ).squeeze()

    Rt0 = R_constrainedCap(t0Min, params)
    print(f"Minimum Omega_Cap obtained at: t0 = ({t0Min.item():.4f} rad, ({t0Min.item()*180/np.pi:.2f} deg), t0Y={t0Y.item()*180/np.pi:.2f} deg, R={Rt0.item():.4e}, h-zb={(Rt0*(1 - torch.sin(t0Min))).item():.4e}")



    OmPlan = None  # planar Omega needs W_fn


    # --- Geometry for plotting the arc ---
    th_hi = 0.5 * torch.pi
    Nplot = 10000
    thetas = torch.linspace(t0Min, th_hi, Nplot, dtype=torch.float64)
    zcMin = params["yb"] - Rt0 * torch.sin(t0Min)
    aMin = Rt0 * torch.cos(t0Min)

    if plotdir is not None:
        plt.plot(t0_grid.detach().numpy(), OmCap.detach().numpy(), label="Omega Cap")
        plt.scatter(t0Min.detach().numpy(), OmCmin.detach().numpy(), color="red", label="Min")
        plt.scatter(t0Y, OmY.detach().numpy(), marker="x", color="orange", label="t0Y")
        if OmPlan is not None:
            plt.hlines(OmPlan, xmin=t0_grid[0].item(), xmax=t0_grid[-1].item(), colors="green", label="Omega Planar")
        plt.legend(); plt.grid(); plt.savefig(plotdir+f"/drop/Omega_vs_t0_{T_i:.2f}.png")
        plt.savefig(plotdir+f"/drop/Omega_vs_t0_temp.png")
        plt.close()


        md_csv = os.path.join(_REPO_ROOT, "data", "dataset", "md_drop", f"rho_field_T{T_i:.2f}.csv")
        if os.path.isfile(md_csv):
            md = load_md_data(md_csv)

            plt.contourf(md["eta"].numpy(), md["psi"].numpy(), md["rho"].numpy(), levels=200, cmap='rainbow',)
            plt.colorbar(label='rho_MD')

            circle = torch.tensor([0.0, zcMin], dtype=torch.float64)[:, None] + \
                        torch.as_tensor(Rt0) * torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=0)

            theta_c = (np.pi/2 - t0Min)*180./np.pi
            # print(f"zcMin = {zcMin}, a_opt = {a_opt}, Rt0 = {Rt0}, t0Min = {t0Min}, contact angle={theta_c:}")
            plt.plot(circle[0].numpy(), circle[1].numpy(), linewidth=2, linestyle='--', color="black", label='Laplace equation')
            # plt.scatter([0], [zcMin], linestyle='--', color="black",)
            plt.hlines([params["yb"]], 0, aMin, linestyle='--', color="black", linewidth=2)
            plt.title(f'2D Droplet at T={T_i:.2f}, N={int(md["N_MD"])}')
            plt.legend()
            # plt.grid()
            plt.savefig(plotdir+f"/drop/Drop.png")
            plt.close()
        else:
            print(f"MD drop field not found ({md_csv}); skipping drop overlay plot.")

    out = {
        "T" : T_i,
        "N": N_j,
        "t0": t0Min,
        "Rt0": Rt0,
        "zc": zcMin,
        "a": aMin,
        "t0_grid": t0_grid,
        "OmCap": OmCap,
        "t0Y" : t0Y
        }
    return out