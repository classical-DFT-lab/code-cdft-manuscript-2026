import torch
import torch.fft
import torch.nn.functional as torch_f

# import pyro
# import pyro.distributions as dist
# import pyro.distributions.transforms as T

#from fmt import FMT

class CDFT():
    """
    classical Density Functional Theory model
    Contains the information for the solution of the cDFT problem
    
    Input:
        # eq_params: dictionary of model parameters (mu, beta, R, Lambda, etc.)
        # mesh: dictionary describing the spatial discretization (x, dx, Nx, etc.)
        # sol: dictionary storing current or initial solution guesses (e.g., rho_guess)
    """

    def __init__(self,
                 eq_params,
                 mesh,
                 sol,
                 ):
        
        self.eq_params = eq_params   # store physical/DFT parameters
        self.mesh = mesh             # store mesh-related data
        self.sol = sol               # store solution dictionary

        # Precomputes useful quantities
        self._psi_zerograd_BCR_()    # store the kernel of internal poterntial at the +inf  
        self._Vext_symm_BCR_()       # store the ext poterntial for symm bc

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def rho_ideal(self):
        # returns the bulk density expression as a function of 
        # eq_pamprams like mu, beta, Lambda, Vext. Ideal gas conditions.
        Lambda = self.eq_params["Lambda"]
        beta = self.eq_params["beta"]
        mu = self.eq_params["mu"]
        Vext = self.eq_params["Vext"]
        return Lambda**(-3) * torch.exp(beta*(mu - Vext))


    def _psi_zerograd_BCR_(self):
        # Store the kernel of the internal/external potential
        # to enforce zero ZERO_GRAD boundary condition at R side.
        x = self.mesh["x"]
        xmax = self.mesh["x_bc"][1]
        sigma_attr = self.eq_params["sigma_attr"]
        eps_attr = self.eq_params["eps_attr"]
        b = torch.max(xmax + 0.*x, sigma_attr + x) 
        term3 = (sigma_attr/(b - x))**3

        psi_ext_bc = ( 
                    +4.*torch.pi * eps_attr * sigma_attr**3 *(
                       + term3**3 / 45 
                       - term3    / 6
                    ) /1.5
        -  6./5.*torch.pi * eps_attr * sigma_attr**3 *(b - xmax)
        
        )[None,:]  # (1, Nx)

        self.psi_zerograd_bcR = psi_ext_bc  # (1, Nx)

    def _Vext_symm_BCR_(self):
        V = torch.flip(self.eq_params["Vext"], dims=(1,))
        self.Vext_symm_bcR = V
    
    def return_VextBCR(self,rho):
        if self.eq_params["BC_R"] == "SYMM":
            Vbc = self.Vext_symm_bcR
        if self.eq_params["BC_R"] == "ZEROGRAD":
            Vbc = self.psi_zerograd_bcR*rho[:,-1:]  # (1, Nx) x (BS, 1) = (BS, Nx)   
        if self.eq_params["BC_R"] == "NONE":
            Vbc = 0.
        return Vbc

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def h0(self, 
             rho_guess, 
             compute_Jac = False,
             detach_tensors=False,
             ):
        """
        h0: helper that computes h0 = rho_b(x) exp(-beta*DFX)
        and its Jacobian operator (Jac_op) (optional). This is used to compute the residual (rho - h0).
        
        self.gradients_FX:
            It must return the following quantities:
                -"DF"
                -"D2F" optional
        """
        
        out = self.gradients_FX( 
                     rho_guess, 
                     compute_D2FX = compute_Jac,
                     detach_tensors=False,
                    )
        
        # Model parameters
        dx = self.mesh["dx"]        
        beta = self.eq_params["beta"]   
        rho_id =  self.rho_ideal()  # ideal density expression
        
        # Boundary conditions
        Vext_BCR = self.return_VextBCR(rho_guess)  # (BS, Nx)

        # Compute the RHS of the EulerLagrange eq. rho = h0[rho] => rho_id * exp( - beta*DF ) 
        rho_h0 = rho_id * torch.exp(-beta*(out["DF"] + Vext_BCR)) # (BS, Nx)
        out["rho_h0"] = rho_h0

        if compute_Jac: 
            Jac_op = (
                torch.eye(rho_h0.shape[-1], device=rho_h0.device)[None,None, ...]  # (B, 1, Nx, Nx)
                + 
            #   (B, 1) x                (B, 1, Nx, 1) x    (B, 1, Nx, Nx)   (,)          
                beta[...,None, None] *  rho_h0[...,None] * dx
                )  # (B, 1, Nx, Nx)
            
            out["Jac_op"] = Jac_op
        return out


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def residual(self, 
                   rho_guess, 
                   compute_Jac = False,
                   detach_tensors=False,
                   ):
        
        # residual => returns (res, Jac_op) for Newton iteration
        # res = rho_guess - h0
        # Jac_op is from c_h0
        out = self.h0(  rho_guess, 
                        compute_Jac = compute_Jac,
                        detach_tensors=False,
                        )
        
        res = rho_guess - out["rho_h0"]
        out["res"] = res 
        return out


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def dfdmu(self, rho_h0):
        # c_dfdmu => partial derivative of the free energy w.r.t. chemical potential mu
        # Useful for the continuation algorithm
        return - self.eq_params["beta"] * rho_h0[:,:,None]


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
