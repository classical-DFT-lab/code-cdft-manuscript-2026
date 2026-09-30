import torch

from libs.cdft_axisym.mesh import integrate


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CANONICAL FIXED-POINT SOLVER
# rho = N u / int u,  u = exp(-beta Vext - beta mu_ex[rho]): damped Picard, then Anderson mixing
def canonical_solve(model, betaV, N, rho0,
                    max_steps = 30000,
                    tol = 1e-9,
                    alpha = 0.02,
                    warmup = 200,
                    m = 5,
                    verbose = True,
                    print_every = 500,
                    ):
    """
    Returns an output dict with:
        - U:        converged density (nr*nz,)
        - res_max:  max |rho - h0(rho)| at the last step
        - steps:    number of iterations
    """
    mesh = model.mesh
    hot = betaV < 100.  # cells inside the wall are kept empty

    def h0(rho):
        u = torch.exp(torch.clamp(-betaV - model.mu_ex(rho), max=50.))
        u = torch.where(hot, u, torch.zeros_like(u))
        g = N * u / integrate(mesh, u)
        return g * (N / integrate(mesh, g))

    with torch.no_grad():
        rho = rho0.reshape(-1)
        rho = rho * (N / integrate(mesh, rho))
        n = rho.shape[0]
        dX = torch.zeros((m, n), dtype=rho.dtype, device=rho.device)
        dF = torch.zeros_like(dX)
        f_old, rho_old = torch.zeros_like(rho), rho
        eye = torch.eye(m, dtype=rho.dtype, device=rho.device)
        i, res, res_old, i_restart = 0, float("inf"), float("inf"), -m - 1
        while i < max_steps and res > tol:
            g = h0(rho)
            f = g - rho
            res = torch.abs(f).max().item()

            grew = res > 1.5 * res_old
            if grew:
                i_restart = i
                dX.zero_()
                dF.zero_()
            elif i > 0:
                slot = (i - 1) % m
                dX[slot] = rho - rho_old
                dF[slot] = f - f_old

            if i >= warmup and i - i_restart > m and not grew:
                A = dF @ dF.T
                lam = 1e-10 * torch.trace(A) / m + 1e-300
                gamma = torch.linalg.solve(A + lam*eye, dF @ f)
                rho_new = rho + f - (dX + dF).T @ gamma
            else:
                rho_new = rho + alpha * f

            if verbose and i % print_every == 0:
                print("Picard step:", i, "\tmax|res|:", f"{res:.2e}", "\t\trho max:", f"{rho.max().item():.4f}")
            f_old, rho_old = f, rho
            rho = torch.clamp(rho_new, min=1e-30)
            res_old = res
            i += 1
        rho = rho * (N / integrate(mesh, rho))

    if verbose:
        print("\nTotal Picard steps:", i, "\tmax|res|:", f"{res:.2e}")
        print("----------------------------------------")
    return {"U": rho, "res_max": res, "steps": i}
