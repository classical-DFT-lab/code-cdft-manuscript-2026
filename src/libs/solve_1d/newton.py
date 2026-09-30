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
    Damped Newton iteration on the residual rho - h0[rho], stopped when max|res| <= tol.
    Returns an output dict with the interesting quantities obtained during
    the computations. It always has:
        - U:        newton solution
        - U_guess:  starting solution
        - deltaU:   norm of the discrepancy between Un - Un-1
        - res_max:  max|res| at the solution
        - converged, steps
    """
    U_old = U_guess.detach()  # (BS, 1, Nx)
    delta_U_norm = float("inf")
    res_max = float("inf")
    i=0
    while i < max_steps:
        out_k = model.residual( U_old,
                                compute_Jac = True,
                                detach_tensors = detach_tensors,
                                )

        res = out_k["res"]
        Jac_op = out_k["Jac_op"]
        res_max = res.detach().abs().max().item()
        if res_max != res_max:  # NaN
            if verbose: print(f"Newton step {i}: non-finite residual, stopping")
            break
        if res_max <= tol:
            break

        # Solve linear system Jac_op * delta_rho = -res
        # -> delta_rho = - Jac_op^-1 * res
        # RHS must be (..., n, k) per torch.linalg.solve; keep Jac_op (BS, 1, Nx, Nx), res (BS, 1, Nx).
        rhs = (-res).unsqueeze(-1)  # (BS, 1, Nx, 1)
        try:
            delta_U = torch.linalg.solve(Jac_op, rhs).squeeze(-1)  # (BS, 1, Nx)
        except torch._C._LinAlgError:
            if verbose: print(f"Newton step {i}: singular Jacobian, stopping")
            break
        if not torch.isfinite(delta_U).all():
            if verbose: print(f"Newton step {i}: non-finite step, stopping")
            break

        U_new = torch.maximum(U_old + delta_U*alpha, torch.zeros_like(U_old)).detach()
        delta_U_norm = (U_new - U_old).abs().mean().item()

        if verbose:
            print("Newton step:", i,
                    "\tmax|res|:", res_max,
                    "\tdelta norm:", delta_U_norm,
                    "\t\tU_new norm:",torch.abs(U_new).mean().item())
        U_old = U_new  # update for next iteration
        i+=1
        if delta_U_norm == 0.:  # step fully clamped
            break

    # After final iteration, store results in self.sol
    out = model.h0( U_old,
                    compute_Jac = False,
                    detach_tensors = detach_tensors,
                    )
    out["U"] = U_old.detach()
    out["U_guess"] = U_guess.detach()
    out["deltaU"] = delta_U_norm
    out["res_max"] = res_max
    out["converged"] = bool(res_max <= tol)
    out["steps"] = i

    if verbose:
        print("\nTotal Newton steps:", i,
            "\tmax|res|:", res_max,
            "\t\tU*2L norm:",torch.abs(U_old).mean().item() * model.mesh["L"]*2)
        print("----------------------------------------")
    return out

