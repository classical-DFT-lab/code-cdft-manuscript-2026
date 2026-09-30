import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
import copy
from tqdm import tqdm

import torch

from libs.cdft_1d.augmented_lda import CDFT_MODEL as CDFT
from libs.cdft_1d.external_potentials import LJ126, LJ93, LJ71, HW
from libs.solve_1d.continuation_gpu import continuation
from libs.solve_1d.picard import picard
from libs.solve_1d.newton import newton
from libs.solve_1d.adjoint import adjoint

from libs.ml.surrogates import setDNN, setDNNRep, setWDA, load_ml_state_dicts, DNN
from libs.ml.dataset_pd import setDatasetObject
from libs.ml.loss import LossL1, LossL2, SpectralLoss, LossGradL2, MMD2Loss
from libs.thermodynamics import batched_coexistence_temperatures
from libs.utils import *
from libs.io_utils import load_pickle, DataNotFoundError

import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = False

import os
# Compute backend: "auto" | "cuda" | "mps" | "cpu". Override: export TORCH_DEVICE=mps
DEVICE_KIND = "auto"
device = resolve_training_device(DEVICE_KIND)
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    print("Available CUDA devices:", torch.cuda.device_count())
print(f"Using device: {device}")
# Floating-point dtype for grids, parameters, and ML (torch.float64 or torch.float32).
# Apple MPS often works best with torch.float32.
TORCH_DTYPE = torch.float64
torch.manual_seed(42)  # For reproducibility


# 1 // Configuration Flags & Output Directories
_script_dir = os.path.dirname(os.path.abspath(__file__))
outdir = os.path.join(_script_dir, "..", "output") + os.sep
datadir = os.path.join(_script_dir, "..", "data") + os.sep
plotdir = os.path.join(outdir, "plot_train")
plotdir_rho = os.path.join(plotdir, "rho")
os.makedirs(plotdir, exist_ok=True)
os.makedirs(plotdir_rho, exist_ok=True)

# Wall type: "wn2" or "wc" (sets pkldir and Vext params)
WALL = "wn2"
if WALL == "wn2":
    pkldir = os.path.join(datadir, "dataset", "pkl", "profiles_wl_wn2") + os.sep
elif WALL == "wc":
    pkldir = os.path.join(datadir, "dataset", "pkl", "profiles_wl_wc") + os.sep
else:
    pkldir = os.path.join(datadir, "dataset", "pkl", "profiles_wl") + os.sep
# Ensure ML model and reference solution directories exist for saving
ml_dicts_dir = os.path.join(datadir, "ml_model", "ml_dicts")
ml_intermediate_dir = os.path.join(datadir, "ml_model", "intermediate_models")
os.makedirs(ml_dicts_dir, exist_ok=True)
os.makedirs(ml_intermediate_dir, exist_ok=True)
USE_MODEL = 1
USE_DBH_DIAMETER = 1  # 1: use Barker-Henderson diameter scaling when USE_MODEL
TRAIN_DNN = 1
TRAIN_DNN_REP = 1
TRAIN_WDA = 1
RESTART_ML_MODEL = 0
SAVE_MODEL = 1
SAVE_INTERMEDIATE_MODELS = 0
JACOBIAN = "EXACT"


if TRAIN_DNN or TRAIN_WDA or TRAIN_DNN_REP: JACOBIAN = "EXACT" # Must be exact for training 

# 3 // Thermodynamic parameters
R           = 1.
mu          = 1.       # mu == chemical potential if Grand Canonical cDFT // mu == particles' number if Canonical cDFT
beta        = 1/0.95
m           = 1.
dft_type    = "LDA"
ensemble    = "NVT"  #"muVT" or "NVT"
guess_coex  = [0.001, 0.9]
str_param   = "mu" if ensemble == "muVT" else "N"



# Compute Lambda
# kb          = 1.38065e-23
# unit_mass   = 1.6726e-27
# mass_star   = 39.95 * unit_mass
# eps_star    = 120.*kb
# sigma_star  = 0.34e-9
# h_star      = np.sqrt(mass_star * eps_star * sigma_star**2)
# h_p         = 6.63e-34/h_star
Lambda      = 1. #np.sqrt(h_p**2 * beta / (2*np.pi*m))



# 2 // SOLVERS: Parameter for the DFT Model solvers
picard_steps = 50         # damped Picard prewarm from the MD profile
picard_steps_safe = 500   # longer prewarm if Newton does not converge after the short one
picard_alpha = 0.03
warm_start = 1            # 1: Newton from the previous step's solution, Picard prewarm only if it fails
newton_max_steps = 40
newton_tol = 1e-11     # on max|res|
newton_alpha = 0.9
newton_verbose = 0
 
cont_ds = 0.5 * (1 - 2*(ensemble=="NVT") )
cont_continuation_steps = 70
cont_max_corrector_steps = 500
cont_alpha = 1.
cont_plot = True



# 4 // Internal and external potential parameters (Ew, sigmaw depend on WALL)
sigma_attr  = 1*R
eps_attr    = 1.
cutoff_attr = 2.5 * R
if WALL == "wn2":
    Ew, sigmaw = 1.2, 1.2
elif WALL == "wc":
    Ew, sigmaw = 1., 2.
else:
    Ew, sigmaw = 1.2, 1.2
cutoff_wall = 10. * R



# 5 // Mesh: Spatial Discretization
L           = 30.
xmin, xmax  = 0., 2*L
Nx          = int(2*L/(0.2*R))
BS          = 5
x_bc        = [xmin, xmax]
dx          = (xmax - xmin)/Nx
x           = torch.linspace(xmin + dx/2, xmax - dx/2, Nx, dtype=TORCH_DTYPE).to(device)
x_wall      = xmin
dx          = x[1] - x[0]
print(f"dx = {dx.item():.4f} , Nx = {Nx}\n\n")


# 6 // External Potential (e.g., Hard Wall) Setup
#Vext = HW(x, x_wall, Ew, sigmaw)
Vext = (
    (LJ93(x, x_wall, Ew, sigmaw) - LJ93(cutoff_wall*torch.ones_like(x), x_wall, Ew, sigmaw)) 
    * 
    torch.sigmoid(-(x - x_wall - cutoff_wall) / (1e-3)) 
)[None,...]  # [B, 1, Nx].      # Shifted 

BC_R = "NONE"  # Right BC: NONE, ZEROGRAD, SYMM available



# 7 // MD
target_density = 0.2798



# 8 // Guess solution
rho_guess   = torch.zeros((BS, 1, Nx), dtype=TORCH_DTYPE).to(device)  # [B, 1, Nx]
#(Vext*0.) + Lambda**(-3) * torch.exp(beta*(torch.tensor(mu)))*0 # [B, 1, Nx]



# 9 // ML
# Two stages: smooth corrections (dnn_rep_fn, dnn_g_fn) with the weighted density switched off,
# then all the corrections with the WDA kernel. Each stage is a ladder of (learning rate, steps),
# with a fresh Adam at every rung.
stage_smooth = ((3e-3, 150), (1e-3, 130))
stage_wda = ((1e-3, 4000), (3e-4, 4000), (1e-4, 1350))
clip_norm = 1.0
guard_factor = 5.0     # loss > guard_factor*best (and above the loss the rung started at) -> back to best
max_guard_fires = 6
guard_backoff = 0.5
log_every = 50
train_T = [0.55, 0.65, 0.75, 0.85, 0.95]
#train_T = [0.55, ]
good_N = [8000]



# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
eq_params = {
    "R": torch.tensor(R, dtype=TORCH_DTYPE, device=device),
    "mu": torch.tensor(mu, dtype=TORCH_DTYPE, device=device)[None, None],  # [B, 1]
    "beta": torch.tensor(beta, dtype=TORCH_DTYPE, device=device)[None, None],  # [B, 1]
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
    "rho_guess": rho_guess.to(device),
    "device": device,
    "dtype": TORCH_DTYPE,
    "outdir": outdir,
    "datadir": datadir,
    "pkldir": pkldir,
    "RESTART_ML_MODEL": RESTART_ML_MODEL,
    "SAVE_MODEL": SAVE_MODEL,
    "SAVE_INTERMEDIATE_MODELS": SAVE_INTERMEDIATE_MODELS,
    "USE_MODEL": USE_MODEL,
    "USE_DBH_DIAMETER": USE_DBH_DIAMETER,
    "TRAIN_DNN": TRAIN_DNN,
    "TRAIN_DNN_REP": TRAIN_DNN_REP,
    "TRAIN_WDA": TRAIN_WDA,
    "JACOBIAN": JACOBIAN,
    "LOSS": None,  # set below on the mesh
}
sol["LOSS"] = MMD2Loss(mesh["x"], mesh["dx"])




# ------------------------------------------------------------------------------
# Instantiate CDFT Model -------------------------------------------
model = CDFT(eq_params, mesh, sol)
# Initial weights. The throwaway network below is not part of the model: it only consumes the random
# draws that preceded the corrections in the run of the published weights, so that seed 42 gives
# the same initial weights.
DNN(input_dim=1, output_dim=6, hidden_dim=24, num_layers=6)
setDNN(model, LR=0.)
setDNNRep(model, LR=0.)
setWDA(model, LR=0., modes=100)



# ------------------------------------------------------------------------------
# Instantiate the dataset ------------------------------------------------------
data_md, dataloader_md = setDatasetObject(
                                          good_N,
                                          train_T,
                                          pkldir,
                                          mesh,
                                          sol,
                                          transform = True,
                                          batch_size = BS,
                                          batches_in_list=False,
                                          )



# ------------------------------------------------------------------------------
# Instantiate ML Models -------------------------------------------
if not TRAIN_DNN:
    model.dnn_g_fn.eval()
    for param in model.dnn_g_fn.parameters():
        param.requires_grad = False
if not TRAIN_DNN_REP:
    model.dnn_rep_fn.eval()
    for param in model.dnn_rep_fn.parameters():
        param.requires_grad = False
if not TRAIN_WDA:
    model.wda_fn.eval()
    for param in model.wda_fn.parameters():
        param.requires_grad = False



# ------------------------------------------------------------------------------
# Training batch: the 5 MD states, solved from the MD profile at every step
batch = next(iter(dataloader_md))
rho_d = batch['rho']                             # [B, 1, Nx]
N_val = batch["N"]                               # [B, 1]
model.eq_params["beta"] = batch["beta"]          # [B, 1]
model.eq_params["mu"] = torch.sum(rho_d, dim=-1)*dx   # [B, 1]  N_MD/ 20**2 --> Area MD simulation box
print("Training states (N, T):", [(int(n), round(float(t), 2)) for n, t in zip(N_val[:, 0], batch["T"][:, 0])])



n_fallbacks = 0
rho_prev = None

def solve_forward(rho_guess):
    global n_fallbacks, rho_prev
    if warm_start and rho_prev is not None:
        out = newton(model, U_guess=rho_prev, detach_tensors=True, max_steps=newton_max_steps,
                     tol=newton_tol, alpha=newton_alpha, verbose=newton_verbose)
        if torch.isfinite(out["U"]).all() and out["converged"]:
            rho_prev = out["U"]
            return out
        n_fallbacks += 1
    for steps in (picard_steps, picard_steps_safe):
        out_p = picard(model, U_guess=rho_guess, max_steps=steps, tol=newton_tol, alpha=picard_alpha, verbose=False)
        out = newton(model, U_guess=out_p["U"], detach_tensors=True, max_steps=newton_max_steps,
                     tol=newton_tol, alpha=newton_alpha, verbose=newton_verbose)
        if torch.isfinite(out["U"]).all() and out["res_max"] <= out_p["res_max"]:
            rho_prev = out["U"]
            return out
        # Newton left the branch: retry with the longer prewarm
        n_fallbacks += 1
        if steps >= picard_steps_safe:
            break
    rho_prev = None
    return out_p

def rel_l2(rho, rho_ref):
    return torch.sqrt(torch.sum((rho - rho_ref)**2, dim=(1, 2)) / torch.sum(rho_ref**2, dim=(1, 2)))

def snapshot(modules):
    return [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in modules]

def restore(modules, snap):
    for m, sd in zip(modules, snap):
        m.load_state_dict(sd)

def save_model(dirname):
    os.makedirs(dirname, exist_ok=True)
    save_state_dict_cpu(model.dnn_g_fn.state_dict(), os.path.join(dirname, "dnn_g_fn.dict"))
    save_state_dict_cpu(model.dnn_rep_fn.state_dict(), os.path.join(dirname, "dnn_rep_fn.dict"))
    save_state_dict_cpu(model.wda_fn.state_dict(), os.path.join(dirname, "wda_fn.dict"))

def plot_state(rho, history, tag):
    for b in range(rho_d.shape[0]):
        plt.plot(model.mesh["x"].cpu(), rho_d[b,0,...].cpu().detach(), color = "black", label=f"rho_d")
        plt.plot(model.mesh["x"].cpu(), (rho[b,0,...]).cpu().detach(), "-.", color = "red", label=f"rho_nn")
        plt.plot(batch['x'].cpu().detach(), (batch['rho_0'][b,0,...]).cpu().detach(), "--", color = "grey", label=f"rho_0")
        plt.xlabel("x")
        plt.legend()
        T_b = 1/model.eq_params["beta"][b].item()
        plt.title(f"T = {T_b:.2f}, N = {N_val[b].cpu().item():.2f}")
        plt.grid()
        plt.savefig(os.path.join(plotdir_rho, f"rho_{tag}_N{N_val[b].item():.0f}_T_{T_b:.2f}.png"))
        plt.close()

    plt.plot(np.log10(history), label=f"loss_data")
    plt.xlabel("Step")
    plt.ylabel("log10(loss)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plotdir, f"loss_{tag}.png"))
    plt.close()



# ------------------------------------------------------------------------------
# ----------------------------- TRAINING LOOP ----------------------------------
# ------------------------------------------------------------------------------
all_modules = [model.dnn_g_fn, model.dnn_rep_fn, model.wda_fn]

def train_stage(tag, stages, modules):
    # Train mode / gradients for the modules of this stage only
    params = []
    for m in all_modules:
        on = m in modules
        m.train() if on else m.eval()
        for p in m.parameters():
            p.requires_grad = on
            if on: params.append(p)

    global rho_prev
    rho_prev = None
    out_fwd = solve_forward(rho_d)
    e = rel_l2(out_fwd["U"], rho_d).cpu().numpy()
    print(f"\n=== stage {tag}: {len(params)} tensors / {sum(p.numel() for p in params)} params, stages = {stages}")
    print(f"before: rel-L2 mean {e.mean():.4f} {np.round(e, 4).tolist()}   max|res| = {out_fwd['res_max']:.2e}")

    history = []
    best, best_snap = np.inf, snapshot(all_modules)
    for lr, n_steps in stages:
        optimizer = torch.optim.Adam(params, lr=lr)
        fires = 0
        floor = np.inf
        for step in range(n_steps):
            optimizer.zero_grad(set_to_none=True)
            ti = time.time()

            # ---------------------------------------------------------------- #
            # FORWARD PROBLEM
            out_fwd = solve_forward(rho_d)

            # ---------------------------------------------------------------- #
            # ADJOINT PROBLEM
            outA = adjoint(
                            model = model,
                            U_fwd = out_fwd["U"],
                            U_data = rho_d,
                            verbose = 0,
            )
            loss = outA["Loss"].item()
            if np.isfinite(loss) and not np.isfinite(floor):
                floor = loss

            # ---------------------------------------------------------------- #
            # GUARD: back to the best parameters with a smaller step if the loss blew up
            if not np.isfinite(loss) or (np.isfinite(best) and loss > guard_factor*max(best, 1e-12) and loss > floor):
                fires += 1
                lr *= guard_backoff
                print(f"guard {fires}/{max_guard_fires} at step {step}: loss = {loss:.3e}, best = {best:.6e}, lr -> {lr:.2e}")
                restore(all_modules, best_snap)
                rho_prev = None
                optimizer = torch.optim.Adam(params, lr=lr)
                if fires >= max_guard_fires:
                    break
                continue
            history.append(loss)

            # best parameters = the ones the loss was evaluated with, before the update
            if loss < best:
                best, best_snap = loss, snapshot(all_modules)
                if SAVE_MODEL:
                    save_model(ml_dicts_dir)

            # ---------------------------------------------------------------- #
            # GRADIENTS: outA["grad"] backpropagates dLoss/dtheta
            outA["grad"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, clip_norm).item()
            optimizer.step()

            if step%log_every==0 or step==n_steps-1:
                print(f"stage {tag} lr {lr:.1e} step {step:5d}: loss = {loss:.6e}, best = {best:.6e}, |grad| = {grad_norm:.3e}, "
                      f"newton max|res| = {out_fwd['res_max']:.1e}, fallbacks = {n_fallbacks}  [{time.time()-ti:.1f}s]", flush=True)
                plot_state(out_fwd["U"], history, tag)

    # leave the best parameters in the model
    restore(all_modules, best_snap)
    if SAVE_MODEL:
        save_model(ml_dicts_dir)
    if SAVE_INTERMEDIATE_MODELS:
        save_model(os.path.join(ml_intermediate_dir, f"ml_dicts_{tag}"))
    out_fwd = solve_forward(rho_d)
    e = rel_l2(out_fwd["U"], rho_d).cpu().numpy()
    print(f"after:  rel-L2 mean {e.mean():.4f} {np.round(e, 4).tolist()}   best loss = {best:.6e}  ({len(history)} steps)")
    plot_state(out_fwd["U"], history, tag)
    return best, history


t0 = time.time()
modules_smooth = [m for m, flag in ((model.dnn_g_fn, TRAIN_DNN), (model.dnn_rep_fn, TRAIN_DNN_REP)) if flag]
modules_wda = modules_smooth + ([model.wda_fn] if TRAIN_WDA else [])

if not RESTART_ML_MODEL:
    sol["USE_WDA"] = False
    train_stage("smooth", stage_smooth, modules_smooth)
sol["USE_WDA"] = True
train_stage("wda", stage_wda, modules_wda)

tf = time.time()
print(f"Total time:{tf-t0:.1f}")
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
