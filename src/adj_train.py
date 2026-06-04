import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
import copy
from tqdm import tqdm

import torch

import libs.thermodynamics as tmd
from libs.cdft_1d.augmented_lda import CDFT_MODEL as CDFT
from libs.cdft_1d.external_potentials import LJ126, LJ93, LJ71, HW
from libs.solve_1d.continuation_gpu import continuation
from libs.solve_1d.picard import picard
from libs.solve_1d.newton import newton
from libs.solve_1d.adjoint import adjoint

from libs.ml.surrogates import setDNN, setDNNRep, setWDA
from libs.ml.dataset_pd import setDatasetObject
from libs.ml.loss import LossL1, LossL2, SpectralLoss
from libs.utils import *
from libs.io_utils import load_pickle, DataNotFoundError

import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True

import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device==torch.device("cuda"): 
    torch.backends.cudnn.benchmark = True
    print("Available devices:", torch.cuda.device_count())
print(f"Using device: {device}")
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
USE_DBH_DIAMETER = 0  # 1: use Barker-Henderson diameter scaling when USE_MODEL
TRAIN_DNN = 1
TRAIN_DNN_REP = 1
TRAIN_WDA = 1
RESTART_ML_MODEL = 1
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
newton_max_steps = 1600
newton_tol = 1e-7
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
xmin, xmax  = -L, L
Nx          = int(2*L/(0.125*R))
BS          = 5
x_bc        = [xmin, xmax]
x           = torch.linspace(xmin, xmax, Nx, dtype=torch.double).to(device)
x_wall      = (xmin - 0.001*sigmaw)
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
rho_guess   = torch.zeros((BS, 1, Nx), dtype=torch.double).to(device)  # [B, 1, Nx]
#(Vext*0.) + Lambda**(-3) * torch.exp(beta*(torch.tensor(mu)))*0 # [B, 1, Nx]



# 9 // ML
lr = 1e-4
epochs = 10000
train_T = [0.55, 0.65, 0.75, 0.85, 0.95]
# train_T = [0.55, ]
good_N = [8000]



# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
eq_params = {
    "R": torch.tensor(R).to(device),
    "mu": torch.tensor(mu).to(device)[None, None],  # [B, 1]
    "beta": torch.tensor(beta).to(device)[None, None],  # [B, 1]
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
}

mesh = {
    "BS": BS,
    "L": L,
    "Nx": Nx,
    "x_bc": [xmin, xmax],
    "x": torch.linspace(xmin, xmax, Nx, dtype=torch.double).to(device),
    "x_wall": torch.tensor(x_wall).to(device),
    "dx": dx.to(device),
}

sol = {
    "rho_guess": rho_guess.to(device),
    "device": device,
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
    "LOSS": LossL2,  # LossL2, LossL1 or SpectralLoss
}




# ------------------------------------------------------------------------------
# Instantiate CDFT Model -------------------------------------------
model = CDFT(eq_params, mesh, sol)



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
setDNN(model, LR=lr)
setDNNRep(model, LR=lr)
setWDA(model, LR=lr, modes=150)

if not TRAIN_DNN:
    model.dnn_fn.eval()
    model.dnn_g_fn.eval()
    for param in model.dnn_fn.parameters():
        param.requires_grad = False
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
# Initialize the guess solution pandas DataFrame
if RESTART_ML_MODEL:
    U_guess_df = load_pickle(
        os.path.join(ml_dicts_dir, "U_guess.pkl"),
        description="U_guess (restart file)",
        hint="Train from scratch with RESTART_ML_MODEL=0, or ensure a previous run saved U_guess.pkl.",
    )
    U_guess_df[['rho', 'x']] = U_guess_df[['rho', 'x']].map(
        lambda arr: torch.tensor(arr, dtype=torch.double, device=model.sol["device"])
    )
else:
    # df_0 is initial reference MF LDA solution DataFrame with MultiIndex (N, T)
    U_guess_df = data_md.df_0.copy()[['rho', "x"]]   # Guess solution from LDA MF
    # U_guess_df = data_md.df_md.copy()[['rho', "x"]]   # Guess solution MD
    U_guess_df = U_guess_df.reset_index().set_index(['N', 'T']).sort_index()
    U_guess_df[['rho', 'x']] = U_guess_df[['rho', 'x']].applymap(
        lambda arr: torch.tensor(arr, dtype=torch.double, device=model.sol["device"])
    )


for idx, row in U_guess_df.iterrows():
    x_src = row['x']                # torch tensor (Nx,)
    rho_src = row['rho'].squeeze() # torch tensor (Nx,) or (1, Nx)
    
    rho_interp = data_md.interpolate_rho(
        x_src,
        rho_src,
        model.mesh["x"],
    ).to(model.sol["device"])       # shape: (1, Nx)

    U_guess_df.at[idx, 'rho'] = rho_interp
    U_guess_df.at[idx, 'x'] = model.mesh["x"]
print("Initialized guess for the training set: ",U_guess_df.head())


def update_U_guess_df(U_guess_df, batch, new_rho):
    # new_rho: tensor of shape [B, 1, Nx]
    for i in range(new_rho.size(0)):
        T = batch['T'][i].item()
        N_val = int(batch['N'][i].item())
        U_guess_df.at[(N_val, T), 'rho'] = new_rho[i,...]

def get_rho_from_U_guess_df(U_guess_df, batch):
    rho_list = []

    for i in range(batch['rho'].shape[0]):
        T = round(batch['T'][i].item(), 6)
        N_val = int(batch['N'][i].item())

        rho_val = U_guess_df.at[(N_val, T), 'rho']  # shape: (1, Nx) or (Nx,)
        
        # Ensure correct shape (1, Nx)
        if rho_val.ndim == 1:
            rho_val = rho_val.unsqueeze(0)

        rho_list.append(rho_val.to(batch['rho'].device))  # move to same device

    # Stack to shape: (B, 1, Nx)
    return torch.stack(rho_list, dim=0)

# ------------------------------------------------------------------------------
# ----------------------------- TRAINING LOOP ----------------------------------
# ------------------------------------------------------------------------------

logloss_history = []
loglossprior_history = []
loglossepoch_history = []
avgloss_history = [0.]
t0 = time.time()
for epoch in range(epochs):
    epoch_loss = 0
    epoch_lossprior = 0

    # Train mode
    if TRAIN_DNN:     model.dnn_fn.train(); 
    if TRAIN_DNN:     model.dnn_g_fn.train(); 
    if TRAIN_DNN_REP: model.dnn_rep_fn.train()
    if TRAIN_WDA:     model.wda_fn.train()

    model.optimizer_dnn.zero_grad()
    model.optimizer_dnn_rep.zero_grad()
    model.optimizer_fno.zero_grad()

    batch_count = 0
    batch_loss = 0.
    batch_lossprior = 0.
    n_batches = len(dataloader_md)
    for batch in dataloader_md:
        # Extract data
        rho_d = batch['rho']             # [B, 1, Nx]
        rho_guess_batch = batch['rho_0']       # [B, 1, Nx]
        beta_val = batch["beta"]         # [B, 1]
        T_val = batch["T"]         # [B, 1]
        N_val = batch["N"]               # [B, 1]
        N_target = torch.sum(batch["rho"], dim=-1)*dx   # [B, 1]
        
        # TMD coordinates
        beta = beta_val     
        N = N_target        #N_MD/ 20**2 --> Area MD simulation box
        model.eq_params["beta"] = beta  # [B, 1]
        model.eq_params["mu"] = N       # [B, 1]

        # Running simulation for T,N....
        ti = time.time()


        # ---------------------------------------------------------------- #
        # FORWARD PROBLEM
        rho_guess = get_rho_from_U_guess_df(U_guess_df, batch)  # [B, 1, Nx]        
        out_fwd = newton(
                    model=model,
                    U_guess=rho_guess, #U_guess_dict[beta_val] ,
                    detach_tensors=True,
                    max_steps=newton_max_steps,
                    tol=newton_tol ,
                    alpha=newton_alpha,
                    verbose= 1 if epoch==0 else newton_verbose,
                    plot=True,
                    )
        update_U_guess_df(U_guess_df, batch, out_fwd["U"])

        
        # ---------------------------------------------------------------- #
        # ADJOINT PROBLEM
        outA = adjoint(
                        model = model,
                        U_fwd = out_fwd["U"],
                        U_data = rho_d,
                        verbose = 1 if epoch==0 else newton_verbose,
        )


        # ---------------------------------------------------------------- #
        # LOSS & GRADIENTS
        # Prior WDA
        w_real = model.wda_fn.blocks[0].spectral_conv.real_weight
        w_imag = model.wda_fn.blocks[0].spectral_conv.imag_weight
        k = torch.linspace(0.1, 10, len(w_real), device=w_real.device)
        var =  (1/k)**2
        
        loss_prior_wda = 1e-6*((w_real**2 + w_imag**2) / var).mean()

        # Adjoint
        grad_adjoint = (outA["grad"] ).mean() 

        # Total
        grad = (grad_adjoint + loss_prior_wda)/len(data_md)     
        grad.backward(retain_graph=False)

        # Store the loss
        batch_loss += outA["Loss"].item()/len(data_md)
        batch_lossprior += loss_prior_wda.cpu().item()/len(data_md)
        batch_count +=1

        # Plot
        if (epoch)%10==0:
            os.makedirs(plotdir, exist_ok=True)
            os.makedirs(plotdir_rho, exist_ok=True)
            print("Plotting in", plotdir, "...")
            for b in range(rho_d.shape[0]):
                plt.plot(model.mesh["x"].cpu(), rho_d[b,0,...].cpu().detach(), color = "black", label=f"rho_d")
                plt.plot(model.mesh["x"].cpu(), (out_fwd["U"][b,0,...]).cpu().detach(), "-.", color = "red", label=f"rho_nn")
                # plt.plot(batch['x'].cpu().detach(), (rho_guess[b,0,...]).cpu().detach(), "-.", color = "orange", label=f"rho_guess")
                plt.plot(batch['x'].cpu().detach(), (batch['rho_0'][b,0,...]).cpu().detach(), "--", color = "grey", label=f"rho_0")
                plt.xlabel("x")
                plt.legend()
                plt.title(f"T = {1/model.eq_params["beta"][b].cpu().item():.2f}, N = {N_val[b].cpu().item():.2f}")
                plt.grid(); #plt.xlim(-30, -25)
                plt.savefig(os.path.join(plotdir_rho, f"rho_temp_N{N_val[b].item():.2f}_T_{1/model.eq_params["beta"][b].item():.2f}.png"))
                plt.close()
                

            plt.plot(logloss_history, label=f"loss_data")
            # plt.plot(loglossprior_history, label=f"loss_prior_wda")
            plt.xlabel("Epoch")
            plt.ylabel("log10(loss)")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(plotdir, "loss.png"))
            plt.close()

    logloss_history.append(np.log10(batch_loss))
    loglossprior_history.append(np.log10(batch_lossprior))
    # loglossepoch_history.append(sum(logloss_history[:-n_batches])/n_batches)

    

    # Scheduler
    if (100*epoch/epochs)%10==0: model.scheduler_dnn.step()
    if (100*epoch/epochs)%10==0: model.scheduler_dnn_rep.step()
    if (100*epoch/epochs)%10==0: model.scheduler_fno.step()

    model.optimizer_dnn.step()
    model.optimizer_dnn_rep.step()
    model.optimizer_fno.step()


    if SAVE_MODEL:
        torch.save(model.dnn_fn.state_dict(), os.path.join(ml_dicts_dir, "dnn_fn.dict"))
        torch.save(model.dnn_g_fn.state_dict(), os.path.join(ml_dicts_dir, "dnn_g_fn.dict"))
        torch.save(model.dnn_rep_fn.state_dict(), os.path.join(ml_dicts_dir, "dnn_rep_fn.dict"))
        torch.save(model.wda_fn.state_dict(), os.path.join(ml_dicts_dir, "wda_fn.dict"))
        with open(os.path.join(ml_dicts_dir, "U_guess.pkl"), 'wb') as file:
            pickle.dump(U_guess_df, file)
        if epoch%25==0 and model.sol["SAVE_INTERMEDIATE_MODELS"]:
            inter_dir = os.path.join(ml_intermediate_dir, f"ml_dicts{epoch}")
            os.makedirs(inter_dir, exist_ok=True)
            torch.save(model.dnn_fn.state_dict(), os.path.join(inter_dir, "dnn_fn.dict"))
            torch.save(model.dnn_g_fn.state_dict(), os.path.join(inter_dir, "dnn_g_fn.dict"))
            torch.save(model.dnn_rep_fn.state_dict(), os.path.join(inter_dir, "dnn_rep_fn.dict"))
            torch.save(model.wda_fn.state_dict(), os.path.join(inter_dir, "wda_fn.dict"))
            with open(os.path.join(inter_dir, "U_guess.pkl"), 'wb') as file:
                pickle.dump(U_guess_df, file)



    avg_loss = sum(logloss_history[:-n_batches]) 
    avg_lossprior = sum(loglossprior_history[:-n_batches])
    avgloss_history.append(avg_loss)
    
    if (epoch)%10==0: print(f"\n\n\n\nEpoch {epoch+1}: Lo ss = {avg_loss:.3e}, Loss prior = {avg_lossprior:.3e},   LR = {model.optimizer_dnn.param_groups[0]["lr"]:.2e}\n")#,\t\ttheta = {model.nn_fn.fc2.weights.data[...,0].item():.6f},{model.nn_fn.fc2.weights.data[...,1].item():.6f}")

tf = time.time()
print(f"Total time:{tf-t0:.1f}")
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
