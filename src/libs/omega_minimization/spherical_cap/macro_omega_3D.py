import math
import torch
import numpy as np
from torch import Tensor
import matplotlib.pyplot as plt

# --------------------------------------------------------- #
# Drop case

def VolDrop(R, t0):
    V_sphere = (4/3) * math.pi * R**3
    frac = (2. - 3.*torch.sin(t0) + 1.*torch.sin(t0)**3)/4.
    return V_sphere * frac

def SurfDropCap(R, t0):
    Scap = (4) * math.pi * R**2 *(1. - torch.sin(t0))/2
    return Scap

def AreaContact(R, t0):
    Ac = math.pi * (R*torch.cos(t0))**2
    return Ac

def ParticlesNumber(R, t0, params):
    rhov = params["rho_v"]
    rhol = params["rho_l"]
    Vbox = params["Vbox"]
    Vdrop = VolDrop(R, t0)
    N_comp = Vdrop*(rhol - rhov) + Vbox*rhov
    return N_comp

def R_constrained(t0, params):
    rhov = params["rho_v"]
    rhol = params["rho_l"]
    Vbox = params["Vbox"]
    Npar_bar =params["Npar"]
    V_bar = (Npar_bar - Vbox*rhov)/(rhol - rhov)

    R =  (V_bar / (1e-12 + 4./3.*torch.pi * (0.5 - 3./4.*torch.sin(t0) + 1./4.*torch.sin(t0)**3)) ).pow(1/3)
    if not (R>0).all(): R  = torch.zeros_like(R) + 2e3
    assert (R>0).all(), f"R_constrained <=0: R={R}, Npar={Npar_bar}, Vbox={Vbox}, rhov={rhov}, rhol={rhol}, V_bar={V_bar}"
    return R


def Omega(
        t0_: Tensor,
        R_: Tensor,
        params: dict,
        ) -> Tensor:
    # Params
    mu0_fn = params["mu0_fn"]
    mu0_infty = params["mu0_infty"]
    W_fn = params["W_fn"]
    rhol = params["rho_l"]
    rhov = params["rho_v"]
    gamma_lg = params["gamma_lg"]
    gamma_wl = params["gamma_wl"]
    gamma_wg = params["gamma_wg"]
    l_max = params["l_max"]
    Winf = params["Winf"]
    Lr = params["Lr"]
    A_tot = torch.tensor(torch.pi * Lr**2).to(t0_.dtype)
    dGamma = (gamma_wg - gamma_wl)
    Ndisc = 10000

    # Constraint on R
    if R_ is None: R = R_constrained(t0_, params)
    else: 
        R = R_
        print(f"Using fixed R: R={R}")
    
    # R->infty contribution
    Om_infty = gamma_lg * SurfDropCap(R, t0_) - dGamma * AreaContact(R, t0_)
    
    # Binding potential contribution
    if W_fn is None: I1 = 0.
    else:
        l0 = l_max
        t0_abs = torch.abs(t0_) #(Nt,)
        dT = (torch.pi/2 - t0_abs)/Ndisc # (Nt,)
        t_grid = t0_abs[...,None] + dT[...,None]*torch.arange(0, Ndisc+1, dtype=t0_.dtype)[None,...]  #(Nt,Ndisc)
        l =  l0 + R[...,None]*(torch.sin(t_grid) - torch.sin(t0_)[...,None])  #(Nt,Ndisc)        
        gMu0 = (mu0_fn(l.unsqueeze(-1))[...,0] - mu0_infty)   
        W_infty = Winf  # W_infty = (gamma_lg+gamma_wl)
        gW = (
            + W_fn(l.unsqueeze(-1))[...,0] 
            - W_infty                           
            + gMu0*(rhol - rhov)*(l) 
        )  #(Nt,Ndisc)
        
        # Compute integral
        I1 = 2*math.pi*(R**2) * ( gW * torch.sin(2*t_grid)/2  * dT[...,None] ).sum(-1) #(Nt,)
        assert I1.shape == Om_infty.shape, f"I1 shape {I1.shape} != Om_infty shape {Om_infty.shape}"
    return (Om_infty  + I1)/A_tot 


