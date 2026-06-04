import torch
import matplotlib.pyplot as plt
import pickle 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NEWTON SOLVER
def newton(model,
           U_guess,
           max_steps,
           detach_tensors,
           tol = 1e-7,
           verbose = True,
           alpha = 1.,
           
           **kwargs):
    """
    Returns an output dict with the interesting quantities obtained during
    the computations. It always has:
        - U:        newton solution
        - U_guess:  starting solution
        - deltaU:   norm of the discrepancy between Un - Un-1
    """
    U_old = U_guess  # (BS, 1, Nx)
    delta_U_norm = 1
    i=0
    while i < max_steps and delta_U_norm > tol:
        out_k = model.residual( U_old,
                                compute_Jac = True,
                                detach_tensors = detach_tensors,
                                )

        res = out_k["res"]
        Jac_op = out_k["Jac_op"]
        
        # Solve linear system Jac_op * delta_rho = -res
        # -> delta_rho = - Jac_op^-1 * res
        delta_U = torch.linalg.solve(Jac_op, -res)
        delta_U_norm = torch.abs(delta_U).mean().item()

        U_new = torch.maximum(U_old + delta_U*alpha, torch.zeros_like(U_old))

        if verbose:
            print("Newton step:", i,
                    "\tdelta norm:", delta_U_norm,
                    "\t\tU_new norm:",torch.abs(U_new.detach()).mean().item())
        U_old = U_new.detach()  # update for next iteration
        i+=1

    # After final iteration, store results in self.sol
    out = model.h0( U_old,
                    compute_Jac = True,
                    detach_tensors = detach_tensors,
                    )
    out["U"] = U_new.detach()
    out["U_guess"] = U_guess.detach()
    out["deltaU"] = delta_U_norm

    if verbose:
        print("\nTotal Newton steps:", i,
            "\tdelta norm:", delta_U_norm,
            "\t\tU_new*2L norm:",torch.abs(U_new.detach()).mean().item() * model.mesh["L"]*2)
        print("----------------------------------------")
    return out


