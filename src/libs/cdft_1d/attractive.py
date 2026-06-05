import torch
import torch.nn.functional as torch_f


# # Compute the kernel of the attractive part of the HFE
# def _psi_attr(self, x):
#     # psi: FA_ex = // rho(x)rho(y) psi(|y-x|) g(x,y) dxdy 
#     sigma_attr_0 = self.eq_params["sigma_attr"][..., None]  # (B, 1, 1)
#     eps_attr = self.eq_params["eps_attr"][..., None]  # (B, 1, 1)
#     cutoff_attr = self.eq_params["cutoff_attr"][..., None]  # (B, 1, 1)
#     x_abs = torch.abs(x)[None,...]  # (1, Nx, Nx)

#     if self.sol["USE_MODEL"]:
#         exp_rep_corr = self.eq_params["pred_dnn"][...,4:5]  [...,None] #(B,1,1)
#         exp_attr_corr = self.eq_params["pred_dnn"][...,5:6] [...,None] #(B,1,1)

#     sigma_attr = sigma_attr_0 
#     #              (1, Nx, Nx)       (B, 1, 1)
#     in_particle = torch.sigmoid(-(x_abs - sigma_attr)/0.01) # (B, Nx, Nx)
#     mask = torch.sigmoid(-(x_abs - cutoff_attr)/0.01) # (B, Nx, Nx)

#     comp1 = sigma_attr/(x_abs + 1e-6)

#     psi_attr = (   
#                  (
#                     4.*torch.pi * eps_attr * sigma_attr**2 * (

#                         + (comp1)**(
#                             10 * ((1. + 0.01*exp_rep_corr) if self.sol["USE_MODEL"] else 1.0)
#                             ) / 5 
                        
#                         - (comp1)**(
#                             4* ((1. + 0.01*exp_attr_corr) if self.sol["USE_MODEL"] else 1.0)
#                             )  / 2
                        
#                         ) * mask * (1 - in_particle)
#                  )
#                         +
#                 (
#                     -6.*torch.pi/5. * eps_attr * sigma_attr**2 * in_particle
#                 )       

#         )   # (B, Nx, Nx) 
#     return psi_attr  # (B, Nx, Nx)


# # Compute the kernel of the attractive part of the HFE
def _psi_attr(self, x):
    # Unpack parameters
    sigma_attr_0 = self.eq_params["sigma_attr"][..., None]  # (B, 1, 1)
    eps_attr = self.eq_params["eps_attr"][..., None]        # (B, 1, 1)
    cutoff_attr = self.eq_params["cutoff_attr"][..., None]  # (B, 1, 1)
    x_abs = torch.abs(x)[None, ...]                         # (1, Nx, Nx)

    # Optional learned exponents (disabled; set USE_MODEL and enable branch to use pred_dnn)
    exp_rep_corr = torch.zeros_like(sigma_attr_0)
    exp_attr_corr = torch.zeros_like(sigma_attr_0)

    # Smoothed transition width (increased from 0.01)
    smooth_width = 0.01

    # Define exponents with learned corrections (needed before geometric clamp)
    rep_exp = 10.0 * (1.0 + 0.01 * exp_rep_corr)    # (B, 1, 1)
    attr_exp = 4.0 * (1.0 + 0.01 * exp_attr_corr)  # (B, 1, 1)

    # Clamp |r| so comp1**rep_exp stays finite in float32. finf.max**(1/exp) is still too
    # large (e.g. 7131**10 overflows); cap comp1 via log: exp_hi*log(comp1) < log(fmax)*margin.
    finf = torch.finfo(x_abs.dtype)
    # MPS has no float64; keep intermediates in x_abs.dtype there.
    acc_dtype = x_abs.dtype if x_abs.device.type == "mps" else torch.float64
    exp_hi = torch.maximum(rep_exp, attr_exp).amax().to(dtype=acc_dtype).clamp(min=4.0)
    logmax = torch.log(torch.tensor(float(finf.max), dtype=acc_dtype, device=x_abs.device))
    log_comp_cap = (logmax * 0.78) / exp_hi
    comp_cap = torch.exp(torch.clamp(log_comp_cap, max=40.0)).to(dtype=x_abs.dtype, device=x_abs.device)
    min_sep = sigma_attr_0 / comp_cap
    x_abs_clamped = torch.clamp(x_abs, min=min_sep)

    # Compute effective sigma and comp1
    sigma_attr = sigma_attr_0
    comp1 = sigma_attr / x_abs_clamped
    comp_cutoff = sigma_attr / cutoff_attr

    # Smooth "in-particle" and "cutoff" masks
    in_particle = torch.sigmoid(-(x_abs - sigma_attr) / smooth_width)     # (B, Nx, Nx)
    mask = torch.sigmoid(-(x_abs - cutoff_attr) / (smooth_width))      # (B, Nx, Nx)
    # mask = 1.      # (B, Nx, Nx)
    # in_particle = 0.5 * (1.0 + torch.tanh(-(x_abs - sigma_attr) / smooth_width))  # (B, Nx, Nx)
    # mask = 0.5 * (1.0 + torch.tanh(-(x_abs - cutoff_attr) / (smooth_width)))  # (B, Nx, Nx)

    # Attractive-repulsive interaction
    attractive = (
        4.0 * torch.pi * eps_attr * sigma_attr**2 *
        (
            (comp1 ** rep_exp) / 5.0 -
            (comp1 ** attr_exp) / 2.0
        ) * (1.0 - in_particle)
    )

    attractive_cutoff = (
        4.0 * torch.pi * eps_attr * sigma_attr**2 *
        (
            (comp_cutoff ** rep_exp) / 5.0 -
            (comp_cutoff ** attr_exp) / 2.0
        ) * (1.0 - in_particle)
    )


    # In-particle contribution (softly activated)
    in_particle_term = -6.0 * torch.pi / 5.0 * eps_attr * sigma_attr**2 * in_particle

    # Total psi_attr
    psi_attr = (attractive - attractive_cutoff)*mask + in_particle_term   # (B, Nx, Nx)
    return psi_attr



# # Compute the kernel of the attractive part of the HFE
def _psi_attr_bulk(self, x):
    # Unpack parameters
    sigma_attr_0 = self.eq_params["sigma_attr"]   # (B,1) 
    eps_attr = self.eq_params["eps_attr"]   # (B,1) 
    cutoff_attr = self.eq_params["cutoff_attr"]   # (B,1) 
    x_abs = torch.abs(x)[None, ...]     # (1, N_r)

    # Optional learned exponents (disabled; set USE_MODEL and enable branch to use pred_dnn)
    exp_rep_corr = torch.zeros_like(sigma_attr_0)
    exp_attr_corr = torch.zeros_like(sigma_attr_0)

    # Smoothed transition width (increased from 0.01)
    smooth_width = 0.01

    # Clamp x_abs to avoid division by near-zero values
    x_abs_clamped = torch.clamp(x_abs, min=1e-6)   #(1, N_r)

    # Compute effective sigma and comp1
    sigma_attr = sigma_attr_0 #(B, 1)
    comp1 = sigma_attr / x_abs_clamped #(B, N_r)
    comp_cutoff = sigma_attr / cutoff_attr #(B, 1)

    # Smooth "in-particle" and "cutoff" masks
    in_particle = torch.sigmoid(-(x_abs - sigma_attr) / smooth_width)     # (B, N_r,)
    mask = torch.sigmoid(-(x_abs - cutoff_attr) / (smooth_width))      # (B, N_r,)
    # mask = 1.      # (B, N_r,)
    # in_particle = 0.5 * (1.0 + torch.tanh(-(x_abs - sigma_attr) / smooth_width))  # (B, N_r,)
    # mask = 0.5 * (1.0 + torch.tanh(-(x_abs - cutoff_attr) / (smooth_width)))  # (B, N_r,)

    # Define exponents with learned corrections
    rep_exp = 10.0 * (1.0 + 0.01 * exp_rep_corr)    # (B, 1)
    attr_exp = 4.0  * (1.0 + 0.01 * exp_attr_corr)  # (B, 1)

    # Attractive-repulsive interaction
    attractive = (
        4.0 * torch.pi * eps_attr * sigma_attr**2 *
        (
            (comp1 ** rep_exp) / 5.0 -
            (comp1 ** attr_exp) / 2.0
        ) * (1.0 - in_particle)
    ) # (B, N_r,)

    attractive_cutoff = (
        4.0 * torch.pi * eps_attr * sigma_attr**2 *
        (
            (comp_cutoff ** rep_exp) / 5.0 -
            (comp_cutoff ** attr_exp) / 2.0
        ) * (1.0 - in_particle)
    ) # (B, N_r,)

    # In-particle contribution (softly activated)
    in_particle_term = -6.0 * torch.pi / 5.0 * eps_attr * sigma_attr**2 * in_particle

    # Total psi_attr
    psi_attr = (attractive - attractive_cutoff)*mask + in_particle_term   # (B, Nr)
    return psi_attr


class ATTRACTIVE_FREE_ENERGY():
    def __init__(self):
        self._psi_attr = lambda x: _psi_attr(self, x)  # Store the kernel of the attractive part
        self._psi_attr_bulk = lambda x: _psi_attr_bulk(self, x)  # Store the kernel of the attractive part

    def phiA_bulk(self, rho,):
        """O(N_r) path for uniform (0D bulk) density when BULK_COMP. Replaces O(Nx^2) full kernel.
        Uses fixed r_grid covering kernel range (cutoff_attr), independent of mesh Nx.
        """
        cutoff = self.eq_params["cutoff_attr"]
        cutoff_val = cutoff.item() if hasattr(cutoff, "item") else float(cutoff)
        dx_val = 0.02
        n_r = max(2, int(cutoff_val / dx_val) + 1)
        r_grid = dx_val * torch.arange(n_r, dtype=rho.dtype, device=rho.device)
        r_grid = torch.clamp(r_grid, min=1e-8)  # avoid psi singularity at 0
        # Weight 1 for r=0, 2 for r>0 (1D symmetry)
        # psi and g at 1D distances (_psi_attr returns (B, 1, N_r) for 1D input; squeeze to (B, N_r))
        psi_r = self._psi_attr_bulk(r_grid)  # (B, N_r)
        if self.sol["USE_MODEL"]:
            g_r = self.dnn_g_fn(r_grid.unsqueeze(-1))[..., 0].unsqueeze(0)  # (1, N_r)
            kernel = (1 + 1e-3 * g_r) * psi_r  # (B, N_r)
        else:
            kernel = psi_r # (B, N_r)
        w = torch.ones_like(r_grid).unsqueeze(0)  # (B, N_r)
        if w.shape[-1] > 1:
            w[:, 1:] = 2.0
        integral = (w * kernel * dx_val).sum(dim=-1, keepdim=True)  # (B, 1)
        phiA_val = 0.5 * rho * rho * integral  # (B, 1) via broadcast
        return phiA_val
    

    def _phiA_bulk(self, rho, dx, x, Nx):
        """O(N_r) path for uniform (0D bulk) density when BULK_COMP. Replaces O(Nx^2) full kernel.
        Uses fixed r_grid covering kernel range (cutoff_attr), independent of mesh Nx.
        """
        cutoff = self.eq_params["cutoff_attr"]
        cutoff_val = cutoff.item() if hasattr(cutoff, "item") else float(cutoff)
        dx_val = dx.item() if hasattr(dx, "item") else float(dx)
        n_r = max(2, int(cutoff_val / dx_val) + 1)
        r_grid = dx * torch.arange(n_r, dtype=rho.dtype, device=rho.device)
        finf = torch.finfo(rho.dtype)
        sigma_m = self.eq_params["sigma_attr"].reshape(-1).amax()
        acc_dtype = rho.dtype if rho.device.type == "mps" else torch.float64
        logmax = torch.log(torch.tensor(float(finf.max), dtype=acc_dtype, device=rho.device))
        comp_cap = torch.exp(torch.clamp((logmax * 0.78) / 10.0, max=40.0)).to(dtype=rho.dtype, device=rho.device)
        min_r = sigma_m / comp_cap
        r_grid = torch.clamp(r_grid, min=min_r)
        # Weight 1 for r=0, 2 for r>0 (1D symmetry)
        # psi and g at 1D distances (_psi_attr returns (B, 1, N_r) for 1D input; squeeze to (B, N_r))
        psi_r = self._psi_attr(r_grid).squeeze(1)  # (B, N_r)
        if self.sol["USE_MODEL"]:
            g_r = self.dnn_g_fn(r_grid.unsqueeze(-1))[..., 0].unsqueeze(0)  # (1, N_r)
            kernel = (1 + 1e-3 * g_r) * psi_r
        else:
            kernel = psi_r
        w = torch.ones_like(r_grid)
        if w.numel() > 1:
            w[1:] = 2.0
        integral = (w * kernel * dx).sum(dim=-1, keepdim=True)  # (B, 1)
        phiA_val = 0.5 * rho * rho * integral.unsqueeze(-1)  # (B, 1, Nx) via broadcast
        return phiA_val
    
    # ATTRACTIVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# --> Attractive Free Energy density
    def phiA(self, rho):
        x = self.mesh["x"]  # (Nx,)
        dx = self.mesh["dx"]  # (,)
        Nx = x.shape[0]

        if self.mesh.get("BULK_COMP", False):
            return self._phiA_bulk(rho, dx, x, Nx)

        xy_abs = torch.abs(x[:, None] - x[None, :])  # (Nx, Nx)
        LJ_xy = self._psi_attr(xy_abs)   # (B, Nx, Nx)

        if self.sol["USE_MODEL"]:
            g = self.dnn_g_fn(xy_abs[..., None])[None, ..., 0]  # (B, Nx, Nx)
        else:
            g = 0

        phiA = torch.einsum("...uj,...uk,...jk->...uj", [0.5 * rho, rho, (1 + 1e-3 * g) * LJ_xy]) * dx  # (B, 1, Nx)
        return phiA  # (B, 1, Nx)



# --> Attractive Free Energy
    def FA(self, phiA):
        dx = self.mesh["dx"]  #(Nx,)
        FA = torch.einsum("...uj->...u", phiA) * dx  # (B,1)
        return FA



    # def DFA(self, rho):
    #     # dFA/drho:= conv(rho * psi_attr * g) [FMT]
    #     x = self.mesh["x"]
    #     dx = self.mesh["dx"]  #(Nx,)
    #     xy_abs = torch.abs(x[:, None] - x[None, :])  # (Nx,Nx)

    #     LJ_xy = self._psi_attr(xy_abs)   #(Nx,Nx)

    #     DFA = torch.einsum("ij,kj->ki", LJ_xy, rho) * dx  # (BS, Nx)
    #     return DFA  # (BS, Nx)



    # def D2FA(self):
    #     x = self.mesh["x"]
    #     xy_abs = torch.abs(x[:, None] - x[None, :])  # (Nx,Nx)
    #     LJ_xy = self._psi_attr(xy_abs)   #(Nx,Nx)

    #     return LJ_xy[None, :, :]  #(1, Nx, Nx)


