import torch
import matplotlib.pyplot as plt
import pickle 


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PICARD SOLVER
def picard(model,
           U_guess,
           max_steps,
           tol = 1e-7,
           alpha = 0.2,
           verbose = True,

           **kwargs):
    """
    Returns an output dict with the interesting quantities obtained during
    the computations. It always has:
        - U:        newton solution
        - U_guess:  starting solution
        - deltaU:   norm of the discrepancy between Un - Un-1
    """
    U_old = U_guess
    delta_U_norm = 1
    i=0
    while i < max_steps and delta_U_norm > tol:
        out_k = model.residual( U_old,
                                compute_Jac = True,
                                )

        res = out_k["res"]

        # Solve linear system Jac_op * delta_rho = -res
        # -> delta_rho = - Jac_op^-1 * res
        delta_U = - res  # (h0 - rho_old)
        delta_U_norm = torch.abs(delta_U).mean().item()

        U_new = U_old + delta_U * alpha

        if verbose:
            print("Picard step:", i,
                    "\tdelta norm:", delta_U_norm,
                    "\t\tU_new norm:",torch.abs(U_new.detach()).mean().item())
        U_old = U_new  # update for next iteration
        i+=1

    # After final iteration, store results in self.sol
    out = model.h0( U_old,
                    compute_Jac = False,
                    )
    out["U"] = U_new
    out["U_guess"] = U_guess
    out["deltaU"] = delta_U_norm

    print("\nTotal Picard steps:", i,
        "\tdelta norm:", delta_U_norm,
        "\t\tU_new norm:",torch.abs(U_new.detach()).mean().item())
    print("----------------------------------------")
    return out

