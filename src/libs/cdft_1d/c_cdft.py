import torch

class CDFT():
    """
    Class for Canonical cDFT equation
    """
    def __init__(self,
                 eq_params,
                 mesh,
                 sol,
                 ):
        
        self.eq_params = eq_params   
        self.mesh = mesh             
        self.sol = sol               

        # Precomputes useful quantities
        self._Vext_symm_BCR_()       # store the ext poterntial for symm bc

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    def _Vext_symm_BCR_(self):
        V = torch.flip(self.eq_params["Vext"], dims=(1,))
        self.Vext_symm_BCR = V
        return V
        
    def return_VextBCR(self,rho):
        if self.eq_params["BC_R"] == "SYMM":
            Vbc = self._Vext_symm_BCR_()  # (B, 1, Nx)
        if self.eq_params["BC_R"] == "NONE":
            Vbc = 0.
        return Vbc

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    def h0(self, 
             rho_guess, 
             compute_Jac,
             detach_tensors,
             ):
        """
    EL:  rho(x) - h0[rho] = 0
        compute Residual and its Jacobian operator (Jac_op) (optional).
        
        self.gradients_FX:
            Return the following quantities:
                -"DF" : functional derivative for Helmholtz Free Energy wrt density
                -"D2F" optional : functional Jacobian for Helmholtz Free Energy wrt density
        """
        
        out = self.gradients_FX( 
                     rho_guess, 
                     compute_D2FX = compute_Jac,
                     detach_tensors = detach_tensors,
                    )
        
        # Model parameters
        dx = self.mesh["dx"]  # (,)        
        beta = self.eq_params["beta"]  # (B, 1)   
        mu = self.eq_params["mu"]   # (B, 1)
        Vext = self.eq_params["Vext"]  # (B, 1, Nx)
        
        # Boundary conditions
        Vext_BCR = self.return_VextBCR(rho_guess)  # (B, 1, Nx)

        # Compute the RHS of the EulerLagrange eq. rho = h0[rho] => rho_id * exp( - beta*DF ) 
        exp =  torch.exp(-beta[...,None]*(out["DF"] + Vext + Vext_BCR))  # (B, 1, Nx)
        int_exp = torch.einsum("...uj->...u", exp * dx)  # (B, 1)

        rho_h0 = mu[...,None] * exp / int_exp[...,None]  # (B, 1, Nx)
        out["rho_h0"] = rho_h0  # (B, 1, Nx)

        if compute_Jac: 
            # Jac_op => linear operator for (I + beta*rho_h0*hess_fwd*dx) -> Discretized of (dh(x)/drho(y), _ )_y            
            D = (
                out["D2F"]  # (B, 1, Nx, Nx)
                - 
                (
                    torch.einsum("...ui,...uij->...uj",[exp, out["D2F"] * dx])  # (B, 1, Nx) x (B, 1, Nx, Nx) => (B, 1, Nx)
                /
                    int_exp[...,None]   # (B, 1, 1)
                ).unsqueeze(-2)  # (B, 1, 1, Nx)
            ) # (B, 1, Nx, Nx)

            Jac_op = (
                torch.eye(rho_h0.shape[-1], device=rho_h0.device)[None,None, ...]  # (B, 1, Nx, Nx)
                + 
            #   (B, 1) x                (B, 1, Nx, 1) x    (B, 1, Nx, Nx)   (,)          
                beta[...,None, None] *  rho_h0[...,None] * D *              dx
                )  # (B, 1, Nx, Nx)
            
            out["Jac_op"] = Jac_op  # (B, 1, Nx, Nx)
            #print(Jac_op)

        return out


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def residual(self, 
                   rho_guess, # (B, 1, Nx)
                   compute_Jac,
                   detach_tensors,
                   ):
        
        # residual => returns (res, Jac_op) for Newton iteration
        # res = rho_guess - h0
        # Jac_op is from c_h0
        out = self.h0(  rho_guess,  # (B, 1, Nx)
                        compute_Jac = compute_Jac,
                        detach_tensors=detach_tensors,
                        )
        
        res = rho_guess - out["rho_h0"]  # (B, 1, Nx)
        out["res"] = res 
        return out


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # COMPUTATIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    def GetOmega(self, rho):
        # Parameters
        beta = self.eq_params["beta"]  # (B, 1)
        DF = self.gradients_FX(
                                rho ,
                                detach_tensors = False,
                                compute_D2FX = False,
                                )["DF"] #(B, 1, Nx)
        
        phiR = self.phiR(rho) #(B, 1, Nx)
        phiA = self.phiA(rho) #(B, 1, Nx)
        phiX = phiA + phiR # (B, 1, Nx)

        pres = rho/beta[...,None] + (rho*DF - phiX)
        omega = - pres
        Omega = torch.einsum("...uj->...u", omega * self.mesh["dx"] )
        # omegaX = - (pres - rho/beta)
        return omega, Omega

    # def GetOmegaX(self, rho):
    #     # Parameters
    #     beta = self.eq_params["beta"]  # (B, 1)
    #     Vext = self.eq_params["Vext"]  # (B, 1, Nx)
    #     N = self.eq_params["mu"]  # (B, 1)
    #     mu = self.GetChemPot(num_par=N, rho_guess=rho)  #(B, 1)
    #     dx = self.mesh["dx"]  #(,)i
    #     T = 1/beta  #(B, 1)

    #     # FA
    #     phiA = self.phiA(rho)  #(B, 1, Nx)
        
    #     # FR
    #     phiR = self.phiR(rho)  #(B, 1, Nx)

    #     omegaX = ( 
    #                 rho * (
    #                         T[...,None] * (torch.log(rho + 1e-16) - 1.0)  #(B, 1, Nx) 
    #                     + 
    #                         (Vext + self.return_VextBCR(rho) - mu[...,None])  #(B, 1, Nx)
    #                 )
    #                     +
    #                 (  
    #                     phiA + phiR  #(B, 1, Nx)
    #                 )
    #     )  #(B, 1, Nx)

    #     # Subtract reference: omegaX = omegaX - omegaX(1) in MATLAB => -omegaX[0] in torch
    #     # Ensure omegaX is at least 1D
    #     # if omegaX.dim() > 1:
    #     #     omegaX = omegaX - omegaX[:,0:1]
    #     OmegaX =  torch.einsum("...uj->...u", omegaX) * dx  #(B,1)
    #     return omegaX, OmegaX   #(B, 1, Nx), (B, 1)


    def GetChemPot(self, num_par, rho_guess):
        """
        Compute the chemical potential given the number of particles.

        Args:
            num_par (torch.Tensor): Number of particles (B, 1)
            model (object): CDFT model object

        Returns:
            torch.Tensor: Chemical potential values (B, 1)
        """
        DF = self.gradients_FX( 
                        rho_guess, 
                        compute_D2FX = False,
                        detach_tensors = True,
                    )["DF"]
        
        # Model parameters
        dx = self.mesh["dx"]  # (,)
        beta = self.eq_params["beta"]   # (B, 1)
        Vext = self.eq_params["Vext"]   # (B, 1)
        Lambda = self.eq_params["Lambda"]  #(,)
        
        # Boundary conditions
        Vext_BCR = self.return_VextBCR(rho_guess)  # (B, 1, Nx)

        # Compute the RHS of the EulerLagrange eq. rho = h0[rho] => rho_id * exp( - beta*DF ) 
        exp = torch.exp(-beta[...,None]*(DF + Vext + Vext_BCR))  # (B, 1, Nx)
        int_exp = torch.einsum("...uj->...u", exp * dx)
        chem_pot = 1/beta * torch.log(Lambda**3 * num_par / int_exp)  # (B, 1)
        return chem_pot


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    def dfdmu(self, rho_h0):
        # dfdN => partial derivative of the free energy w.r.t. number of particles N
        # Useful for the continuation algorithm
        #     (B, 1, Nx) (B, 1) ->       
        return -(rho_h0 / self.eq_params["mu"] [...,None])[...,None]  # (B, 1, Nx, 1)
