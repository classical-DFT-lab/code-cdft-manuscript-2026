import torch
import matplotlib.pyplot as plt
import pickle 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# NEWTON SOLVER
def adjoint(model,
           U_fwd,
           U_data,
           verbose = False,
            ):
    """
    Returns an output dict with the interesting quantities obtained during
    the computations. It always has:
        - UA:        newton solution
        - UA_guess:  starting solution
        - deltaUA:   norm of the discrepancy between Un - Un-1
    """
    outA = {}

    out_fwd = model.residual( U_fwd,
                            compute_Jac = True,
                            detach_tensors = False,
                            )
    Jac_op = out_fwd["Jac_op"]
    res = out_fwd["res"]
        
    Jac_op_t = torch.einsum("buij->buji", Jac_op)  # (BS, 1, Nx, Nx)

    # Solve linear system Jac_op_t * UA = -dJ
    #dJ = 2*(U_fwd - U_data)/torch.norm(U_data + 1e-8)**2  # (BS, Nx)
    Loss = model.sol["LOSS"] (U_fwd, U_data)  # (,)

    dJ = torch.autograd.grad(outputs=Loss,
                            inputs=U_fwd, 
                            grad_outputs=torch.ones_like(Loss), 
                            create_graph=True, 
                            retain_graph=True)[0]  # (BS, 1, Nx)
    
    UA = torch.linalg.solve(Jac_op_t, -dJ)  # (BS, 1, Nx)

    # Compute gradients
    grad = torch.sum(UA*res) 
      
    outA["Loss"] = Loss  # (,)
    outA["grad"] = grad
    outA["UA"] = UA.detach()  # (BS, 1, Nx)


    if verbose:
        print("Adjoint step:"
                "\t\tUA norm =",torch.abs(UA.detach()).mean().item())
    print("----------------------------------------")
    return outA


