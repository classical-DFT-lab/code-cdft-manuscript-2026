import matplotlib.pyplot as plt
from scipy.optimize import root
import numpy as np 
import torch

import libs.solve_1d.integrate.sum as integrate
from libs.utils import *

# -------------
# BULK COEXISTENCE
# -------------

# Carnaham-Sterling + BH
def coex_equations(rho_vec, model):
    """
    Returns the 2 equations that enforce:
      mu(rho1) = mu(rho2)
      P(rho1)  = P(rho2)
    Here mu_shift = 0 means we find the coexistence at the original T.
    """
    r1, r2 = rho_vec
    eq1 = bulk_mu_csbH(r1, model) - bulk_mu_csbH(r2, model)
    eq2 = bulk_pres_csbH(r1, model) - bulk_pres_csbH(r2, model)
    return [eq1, eq2]



def bulk_mu_csbH(rho, model, mu_target=0): # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    Bulk chemical potential for Carnahan-Starling + BH (approx).
    """

    rho = np.abs(rho)
    
    eta = np.pi * rho/6.
    beta = model.eq_params["beta"].detach().numpy()
    Lambda = model.eq_params["Lambda"].detach().numpy()
    sigma_attr = model.eq_params["sigma_attr"].detach().numpy()
    eps_attr = model.eq_params["eps_attr"].detach().numpy()
    #print("Rho:",rho)
    #print(np.log(rho + 1e-16))
    f = (
           1/beta * (
               np.log(Lambda**3 * rho + 1e-16) +  # dFid
               #eta * (8 - 9*eta + 3*eta**2)/(1 - eta)**3
               (3 - eta )/(1 - eta)**3  # dFXR
               )
           - 32*np.pi/9*rho * sigma_attr**3 * eps_attr #* 4/9  #dFXA
           - mu_target  
      )    
    return f



def bulk_pres_csbH(rho, model): # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """
    Bulk pressure for Carnahan-Starling + BH.
    """
    rho= np.abs(rho)
    eta = np.pi * rho/6.
    beta = model.eq_params["beta"].detach().numpy()
    sigma_attr = model.eq_params["sigma_attr"].detach().numpy()
    eps_attr = model.eq_params["eps_attr"].detach().numpy()

    out = (
        rho / beta * (
            1 + eta + (eta)**2 - (eta)**3
            )/(1 - eta)**3 
        - 
        16/9*np.pi * rho**2 * sigma_attr**3 * eps_attr #* 4/9
        #sigma_attr**3 * eps_attr * rho**2
      )

    return out

def find_coexistence_state(model, guess_coex = [0.00001, 200.51]):
    
    r1 = np.linspace(0.002, 10., 1000)
    r2 = np.linspace(0.002, 10., 1000)
    #r1 = np.linspace(0.01, 10, 1000)
    #r2 = np.linspace(0.01, 10, 1000)
    R2, R1 = np.meshgrid(r1, r2)
    res = coex_equations([R1, R2], model)
    plt.contourf((R1), (R2), np.log(np.abs(res[0]+1e-32) + np.abs(res[1])+1e-32)<-10 , cmap="plasma", levels=50)
    plt.colorbar()
    plt.savefig(model.sol["outdir"] + "plot/coex")
    plt.close()

    # Solve for (rho_vap, rho_liq).
    sol_coex = root(lambda x: coex_equations(x, model),
                    guess_coex, method='hybr', tol=1e-10)
    
    rho_liq, rho_vap = max(np.abs(sol_coex.x[0]), np.abs(sol_coex.x[1])), min(np.abs(sol_coex.x[0]), np.abs(sol_coex.x[1])) 
    model.sol["rho_vl"] = [rho_vap, rho_liq] 
    
    print("\nCOEXISTENCE ANALYSIS -------------------------------------------")
    print(sol_coex.message)
    print("Rho vapour:", rho_vap, "\nRho liquid:", rho_liq)
    # Chemical potential at coexistence

    mucoex = bulk_mu_csbH(rho_vap, model)
    print(f"\nmu coexistence: {mucoex}")
    print("------------------------------------------------------------------")



def find_rho_bulk(mu, model, guess=[1e-2]):
    #guess = model.sol["rho_vl"][0]
    rhoB = root(lambda x: bulk_mu_csbH(x, model, mu_target=mu),
                    guess, method='hybr', tol=1e-10)
    # rhoB_vap = min(np.abs(rhoB.x[0]), np.abs(rhoB.x[1]))
    # rhoB_liq = max(np.abs(rhoB.x[0]), np.abs(rhoB.x[1]))
    model.sol["rhoB_vap"] = np.abs(rhoB.x[0])
    
    print("\nBULK ANALYSIS -------------------------------------------")
    print(rhoB.message)
    print("Rho bulk vapour:", np.abs(rhoB.x[0]))#, "\nRho bulk liquid:", rhoB_liq)
    print(f"at starting mu: {mu}")
    print("----------------------------------------------------------------")




# ----------------------------------------------------------------------------------

# -------------
# BULK COEXISTENCE AUTODIFF
# -------------
# COMPUTE THE PRESSURE FROM THE FREE ENERGY MODEL AUTOMATICALLY 
def pres(rho, model, mu_target=0):
    """
    Bulk pressure for Carnahan-Starling + BH.
    """
    rho= torch.abs(rho).view(1,1)
    beta = model.eq_params["beta"]
    Lambda = model.eq_params["Lambda"]
    # sigma_attr = model.eq_params["sigma_attr"]
    # eps_attr = model.eq_params["eps_attr"]

    phiR = model.phiR(n=model.n(rho)).mean()
    phiA = model.phiA(rho) [0, model.mesh["Nx"]//2] #.mean() 
    f_ex =  (phiR + phiA).squeeze()

    DF = model.gradients_FX(rho, 
                            detach_tensors=False,
                            compute_D2FX=False,
                            )["DF"][0, model.mesh["Nx"]//2]
    
    mu =  1./beta*(torch.log(Lambda**3 * rho)) + DF
    p = rho*(mu) - rho/beta*(torch.log(Lambda**3 * rho) - 1) - f_ex

    # print("rho shape:", rho.shape)
    # print("phiA shape:", phiA.shape)
    # print("phiR shape:", phiR.shape)
    # print("f_ex shape:", f_ex.shape)
    # print("DF shape:", DF)
    # print("mu shape:", mu.shape)
    # print("p shape:", p.shape)
    return p


# COMPUTE THE CHEMICAL POTENTIAL FROM THE FREE ENERGY MODEL AUTOMATICALLY
def mu(rho, model):
    """
    Bulk chemical potential
    """
    rho= torch.abs(rho).view(1,1)
    beta = model.eq_params["beta"]
    Lambda = model.eq_params["Lambda"]

    DF = model.gradients_FX(rho, 
                            detach_tensors=True,
                            compute_D2FX=False,
                            )["DF"][0, model.mesh["Nx"]//2]
    
    mu =  1/beta*(torch.log(Lambda**3 * rho)) + DF
    return mu



def coex_equations_auto(rho_vec, model):
    """
    Returns the 2 equations that enforce:
      mu(rho1) = mu(rho2)
      P(rho1)  = P(rho2)
    Here mu_shift = 0 means we find the coexistence at the original T.
    """
    r1, r2 = rho_vec
    eq1 = mu(r1, model) - mu(r2, model)
    eq2 = pres(r1, model) - pres(r2, model)
    return [eq1, eq2]



def solve_newton(coex_equations, rho_init, model, tol=1e-5, max_iter=1000):
    rho = rho_init.clone().detach() # Initial guess
    k = 0
    for i in range(max_iter):
        rho.requires_grad_(True)
        F = coex_equations(rho, model)  # Compute function values
        J = torch.zeros((2, 2), dtype=torch.float64)  # Jacobian matrix

        # Compute Jacobian using autograd
        for j in range(2):
            J_i = torch.autograd.grad(F[j],
                                rho,
                                retain_graph=True,
                                materialize_grads=True)[0]  # (BS, Nx)
            J[j,:] = J_i

        F_n = torch.tensor([F[0].item(), F[1].item()], dtype=torch.double)
        # Newton-Raphson step: rho = rho - J⁻¹ * F
        with torch.no_grad():
            delta_rho = torch.linalg.solve(J, -F_n)  # Solve for update step
            rho += 0.1*delta_rho  # Update values
            
        # Convergence check
        if torch.norm(delta_rho) < tol:
            print("Number of Newton iterations:", k)
            break
        k += 1
    return rho.detach()



def find_coexistence_state_auto(model, guess_coex=[0.01, 0.99]):
    # Solve for (rho_vap, rho_liq).
    rho_init = torch.tensor(guess_coex)  # Initial guess for densities
    sol_coex = solve_newton(coex_equations_auto, 
                            rho_init, 
                            model,
                            max_iter=100,
                            )
    rho_liq, rho_vap = max(np.abs(sol_coex[0]), np.abs(sol_coex[1])), min(np.abs(sol_coex[0]), np.abs(sol_coex[1])) 
    model.sol["rho_vl"] = [rho_vap, rho_liq] 
    
    print("\nCOEXISTENCE ANALYSIS -------------------------------------------")
    print("Rho vapour:", rho_vap.item(), "\nRho liquid:", rho_liq.item())
    
    # Chemical potential at coexistence
    mucoex = mu(rho_vap, model).item()
    model.sol["mu_vl"] = mucoex
    print(f"\nmu coexistence: {mucoex}")
    print("------------------------------------------------------------------")

    mu1 = bulk_mu_csbH(rho_vap, model).item()
    mu2 = mu(rho_vap, model).item()
    pres1 = bulk_pres_csbH(rho_vap, model).item()
    pres2 = pres(rho_vap, model).item()
    
    print("Comparison between autotmd and tmd")
    print("mu \t/ auto:", mu2, "\ttmd:", mu1)
    print("pres \t/ auto:", pres2, "\ttmd:", pres1)



def compute_mu_rho_curve(sol_curve, rho_mean_target):
    """
    Compute the relationship mu = mu(mean_rho) and extract mu & rho(x) for a given rho_mean_target.

    Args:
        sol_curve (list of dict): Each dictionary has keys:
            - "rho": Tensor of shape (BS, Nx)
            - "mu": Scalar tensor (BS,)
        rho_mean_target (torch.Tensor): Target rho_mean values of shape (BS, 1)

    Returns:
        mu_target (torch.Tensor): Interpolated mu values for the given rho_mean_target (BS, 1)
        rho_x_target (torch.Tensor): Interpolated rho(x) profiles (BS, Nx)
    """
    device = sol_curve[0]["rho"].device
    rho_mean_target = torch.tensor(rho_mean_target).to(device)
    rho_means = []
    mu_values = []
    rho_profiles = []

    # Step 1: Extract and process data from the list of dictionaries
    for entry in sol_curve:
        rho = entry["rho"].to(device)[:,0,:]  # Shape (BS, Nx)
        mu = entry["mu"].to(device)     # Shape (BS, 1)

        mean_rho = rho.mean(dim=-1, keepdim=True)  # Shape (BS, 1,)

        rho_means.append(mean_rho)
        mu_values.append(mu)  # Ensure mu has shape (BS, 1)
        rho_profiles.append(rho)  # Full rho(x) profiles

    # Step 2: Concatenate across all samples
    rho_means = torch.cat(rho_means, dim=0).squeeze()  # Shape (Total_BS,)
    mu_values = torch.cat(mu_values, dim=0).squeeze()  # Shape (Total_BS,)
    rho_profiles = torch.cat(rho_profiles, dim=0)  # Shape (Total_BS, Nx)

    # Step 3: Sort values by rho_mean for interpolation
    sorted_indices = torch.argsort(rho_means)
    rho_means_sorted = rho_means[sorted_indices]  # Shape (Total_BS,)
    mu_values_sorted = mu_values[sorted_indices]  # Shape (Total_BS,)
    rho_profiles_sorted = rho_profiles[sorted_indices]  # Shape (Total_BS, Nx)

    # Step 4: Interpolate mu for the given rho_mean_target
    mu_target = linear_interpolation(rho_mean_target.squeeze(), rho_means_sorted, mu_values_sorted)
    mu_target = mu_target.view(-1, 1)  # Ensure correct shape (BS, 1)

    # Step 5: Interpolate rho(x) profiles for the corresponding mu_target
    rho_x_target = torch.zeros((rho.shape[0], rho.shape[1]), device=device)  # Shape (BS, Nx)

    for i in range(rho_profiles.shape[1]):  # Iterate over Nx dimension
        rho_x_target[:, i] = torch.maximum(
            linear_interpolation(rho_mean_target.squeeze().to(device), rho_means_sorted.to(device), rho_profiles_sorted[:, i].to(device)),
            rho_x_target[:, i])
    return mu_target, rho_x_target


