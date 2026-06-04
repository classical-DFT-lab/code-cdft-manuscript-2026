import torch
from typing import Optional, List


from libs.cdft_1d.c_cdft import CDFT as c_CDFT
from libs.cdft_1d.gc_cdft import CDFT as gc_CDFT
from libs.cdft_1d.attractive import ATTRACTIVE_FREE_ENERGY as AFE
from libs.cdft_1d.repulsive  import REPULSIVE_FREE_ENERGY  as RFE
import libs.solve_1d.integrate.sum as integrate



@torch.jit.script
def compute_grad_output_i(
                        e_i: torch.Tensor, 
                        y : List[torch.Tensor], 
                        rho : torch.Tensor,
                        ):
        """
        Utility function to vectorize the tensorial jacobian computation using backpropagation.
        """

        DFR, DFA = y  # (BS, 1, Ny=Nx)x2
        D2F_op = torch.autograd.grad(
            outputs=[DFR, DFA],  # (BS, 1, Ny=Nx)x2
            inputs=[rho,],  # (BS, 1, Nx) x1
            grad_outputs=torch.jit.annotate(Optional[List[Optional[torch.Tensor]]], [e_i,e_i]), # (BS, 1, Ny=Nx)
            create_graph=torch.jit.annotate(bool,True),  
            #materialize_grads=torch.jit.annotate(bool,True),
            )[0]            
        return D2F_op  # (BS, 1, Nx) 


def CDFT_MODEL(eq_params, mesh, sol):
    """
    Provide the class for the cDFT model assembling EL and F_ex.
    """

    ensemble = eq_params["ensemble"]

    CDFT = c_CDFT if ensemble == "NVT" else gc_CDFT
    ATTRACTIVE_FREE_ENERGY = AFE
    REPULSIVE_FREE_ENERGY = RFE

    class LDA_HardSpheres(CDFT, ATTRACTIVE_FREE_ENERGY, REPULSIVE_FREE_ENERGY):
        def __init__(self,
                    eq_params,
                    mesh,
                    sol,
                    ):

            CDFT.__init__(self,
                        eq_params,
                        mesh,
                        sol,
                        )

            ATTRACTIVE_FREE_ENERGY.__init__(self)
            REPULSIVE_FREE_ENERGY.__init__(self)



        def DF_auto(self, rho):
            """
            Computes F_ex's gradients using backpropagation.
            """

            if self.sol["USE_MODEL"]:
                self.eq_params["pred_dnn"] = self.dnn_fn(self.eq_params["beta"])#.double())  # [B, 1]
                self.eq_params["sigma_attr"] = 1. + 0.01*self.eq_params["pred_dnn"][...,0:1]  #[B, 1] 
                self.eq_params["eps_attr"] =   1. + 0.01*self.eq_params["pred_dnn"][...,1:2]  #[B, 1]

            # REPULSIVE
            phiR = self.phiR(rho)   # (B, 1, Nx)
            #                                     (B, 1, Nx)
            DFR = torch.autograd.grad(  outputs = [phiR * self.mesh["dx"] / self.mesh["dx"],],
                                        inputs=rho,  # (B, 1, Nx)
                                        grad_outputs=[torch.ones_like(rho),],  # (B, 1, Nx)
                                        create_graph=True,  # len(grad_output) x (B, 1, Nx)
                                        materialize_grads=True)[0]  # (B, 1, Nx)

            # ATTRACTIVE
            phiA = self.phiA(rho)  # shape (B, 1, Nx)
            DFA = torch.autograd.grad(  outputs = [phiA * self.mesh["dx"] / self.mesh["dx"],],
                                        inputs=rho,  # (B, 1, Nx)
                                        grad_outputs=[torch.ones_like(rho),],
                                        create_graph=True,  
                                        materialize_grads=True)[0]  # (B, 1, Nx)

            return DFR, DFA


        

        # MAIN OUTPUT: gradients_FX --> It allows the computation of the derivatives of the FREE ENERGY
        def gradients_FX(self,
                        rho_guess,
                        detach_tensors,
                        compute_D2FX,
                        ):
            """
            gradients_FX: uses vectorized autodiff to compute the functional derivative and hessian of the excess free-energy functional FX
            wrt rho.
            
            It returns a dict out with keys:
              - "DF":      functional derivative of Fx
              - "D2F":     functional hessian of FX (Optional)
            """
            # ----------------------------------------------------------------
            out = {}
            with torch.enable_grad():
                # Setup
                rho = rho_guess.requires_grad_(True)  # (BS, Nx)
                DFR, DFA = self.DF_auto(rho)  # (BS, Nx), (BS, Nx)
                DF = DFR + DFA  # (BS,Nx)


                if self.sol["JACOBIAN"] == "EXACT": DF_input = [DFR, DFA]
                elif self.sol["JACOBIAN"] == "STABLE": DF_input = [DFR/self.mesh["dx"], DFA]
                else: raise ValueError("Must specify Jacobian computation method")

                # JACOBIAN D2F: dF/(drho(x)drho(y)) --------------------------------
                if compute_D2FX:
                    jacobian_DF_op = torch.vmap(lambda e_i: compute_grad_output_i(
                        e_i,
                        DF_input,
                        rho,
                        )
                    )

                    # I = torch.eye(rho.shape[-1], device=rho.device, dtype=rho.dtype)
                    # I_batched = I[:,None, None,:].expand(-1, rho.shape[0], -1, -1)  # (Nx, B, 1, Nx)
                    D2F_temp = jacobian_DF_op(
                        torch.eye(
                            rho.shape[-1], device=rho.device, dtype=rho.dtype
                            )[:,None, None,:].expand(-1, rho.shape[0], -1, -1)  # (Nx, B, 1, Nx)
                        )/self.mesh["dx"]  # (Nx, B, 1, Nx) 
                    D2F = torch.einsum("i...j->...ij", D2F_temp)  # (B, 1, Nx, Nx)
                    out["D2F"] = D2F 
            out["DF"] = DF
            return out


    return LDA_HardSpheres(eq_params, mesh, sol)