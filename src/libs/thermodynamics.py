
import torch

from libs.cdft_1d.augmented_lda import CDFT_MODEL as CDFT
from libs.cdft_1d.external_potentials import LJ93
from libs.ml.surrogates import setDNN, setWDA, setDNNRep, load_ml_state_dicts
from libs.utils import resolve_training_device, sol_dtype, tensors_to_cpu_for_storage


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _as_batched_temperature(T_, dtype, device):
    """
    Convert temperature input to shape (B, 1).
    Accepts scalar, list, np.ndarray, or tensor.
    """
    T_ = torch.as_tensor(T_, dtype=dtype, device=device)
    if T_.ndim == 0:
        T_ = T_[None]
    return T_.reshape(-1, 1)


def _expand_rho_guess(rho_guess, batch_size):
    """
    Ensure rho_guess has shape (B, 1, Nx).
    """
    if rho_guess.shape[0] == batch_size:
        return rho_guess
    if rho_guess.shape[0] == 1:
        return rho_guess.expand(batch_size, -1, -1).clone()
    raise ValueError(
        f"rho_guess batch mismatch: got {rho_guess.shape[0]}, expected 1 or {batch_size}"
    )


def _make_zero_vext(mesh_local, batch_size, dtype, device):
    """
    Build zero external potential with shape (B, 1, Nx).
    """
    # keep same construction path as original code
    vext = LJ93(mesh_local["x"], mesh_local["x_wall"], 1, 2)[None, ...].to(device=device, dtype=dtype) * 0
    if vext.ndim == 2:
        vext = vext[:, None, :]
    return vext.expand(batch_size, -1, -1).clone()


def _clone_mesh_with_batch(mesh_local, batch_size):
    mesh_b = dict(mesh_local)
    mesh_b["BS"] = batch_size
    return mesh_b


def _clone_sol_with_rho_guess(sol_local, rho_guess_batched):
    sol_b = dict(sol_local)
    sol_b["rho_guess"] = rho_guess_batched
    return sol_b


def _default_bulk_mesh(eq_params, sol_local, nx_bulk=3):
    """
    Build a minimal bulk mesh when one is not explicitly provided.

    Matches the defaults used in compute_bulk_tmd.py:
      Nx_bulk = 3
      L_bulk = 0.2 * R * (Nx_bulk - 1) / 2
      x_bulk = linspace(-L_bulk, L_bulk, Nx_bulk)
    """
    td = sol_dtype(sol_local)
    dev = sol_local["device"]

    R = float(eq_params["R"].detach().item() if torch.is_tensor(eq_params["R"]) else eq_params["R"])
    L_bulk = 0.2 * R * (nx_bulk - 1) / 2
    x_bulk = torch.linspace(-L_bulk, L_bulk, nx_bulk, dtype=td, device=dev)
    dx_bulk = x_bulk[1] - x_bulk[0]

    x_wall = x_bulk.min() - 0.01 * R

    return {
        "BS": 1,
        "L": L_bulk,
        "Nx": nx_bulk,
        "x_bc": [x_bulk.min().item(), x_bulk.max().item()],
        "x": x_bulk,
        "x_wall": x_wall.to(device=dev, dtype=td),
        "dx": dx_bulk.to(device=dev),
        "BULK_COMP": True,
    }


def _default_sol_bulk(sol, mesh_bulk):
    """
    Build default bulk solver config as: {**sol, "rho_guess": rho_guess_bulk}.
    """
    if sol is None:
        raise ValueError("sol must be provided when sol_bulk is not passed.")
    if "device" not in sol:
        raise ValueError("sol must define 'device'.")

    td = sol_dtype(sol)
    dev = sol["device"]
    nx_bulk = int(mesh_bulk["Nx"])
    rho_guess_bulk = torch.zeros((1, 1, nx_bulk), dtype=td, device=dev)
    return {**sol, "rho_guess": rho_guess_bulk}


# ------------------------------------------------------------------------------
# Model builders
# ------------------------------------------------------------------------------
def update_model(x_, rho_, T_, eq_params_, mesh_=None, sol_=None, ml_state_dicts_=None):
    """
    Build CDFT model for given mesh and temperature(s).

    Supports:
        rho_: (B, 1, Nx)
        T_:   scalar or (B,)
    """
    if mesh_ is not None and sol_ is not None:
        mesh_local = mesh_
        sol_local = sol_
    else:
        batch_size = rho_.shape[0]
        if sol_ is None:
            raise ValueError("sol_ must be provided when mesh_ is not passed.")
        mesh_local = {
            "BS": batch_size,
            "L": x_.max().item(),
            "Nx": len(x_),
            "x_bc": [x_.min().item(), x_.max().item()],
            "x": x_,
            "x_wall": x_.min() - 0.001,
            "dx": (x_[1] - x_[0]).to(device=rho_.device),
        }
        sol_local = sol_

    td = sol_dtype(sol_local)
    dev = sol_local["device"]

    if rho_.ndim != 3:
        raise ValueError(f"rho_ must have shape (B, 1, Nx), got {tuple(rho_.shape)}")

    batch_size = rho_.shape[0]
    T_b = _as_batched_temperature(T_, dtype=td, device=dev)  # (B, 1)
    if T_b.shape[0] != batch_size:
        raise ValueError(
            f"Temperature batch mismatch: T has batch {T_b.shape[0]} but rho has batch {batch_size}"
        )

    mesh_b = _clone_mesh_with_batch(mesh_local, batch_size)
    sol_b = _clone_sol_with_rho_guess(sol_local, rho_)

    eq_params_local = dict(eq_params_)
    eq_params_local["beta"] = 1.0 / T_b                               # (B, 1)
    eq_params_local["mu"] = rho_.sum(dim=-1) * mesh_b["dx"]          # (B, 1)
    eq_params_local["Vext"] = _make_zero_vext(mesh_b, batch_size, td, dev)

    model = CDFT(eq_params_local, mesh_b, sol_b)

    sd = ml_state_dicts_ if ml_state_dicts_ is not None else None
    setDNN(model, LR=0.0, state_dicts=sd)
    setDNNRep(model, LR=0.0, state_dict=sd["dnn_rep_fn"] if sd else None)
    setWDA(model, LR=0.0, modes=150, state_dict=sd["wda_fn"] if sd else None)

    if hasattr(model, "dnn_fn") and model.dnn_fn is not None:
        model.dnn_fn.eval()
    if hasattr(model, "dnn_rep_fn") and model.dnn_rep_fn is not None:
        model.dnn_rep_fn.eval()
    if hasattr(model, "wda_fn") and model.wda_fn is not None:
        model.wda_fn.eval()

    return model


def chem_pot(rho, model):
    """
    Bulk chemical potential.

    rho: (B, 1, Nx)
    returns: same broadcasted shape as DF / rho
    """
    beta = model.eq_params["beta"][..., None]  # (B, 1, 1)
    Lambda = model.eq_params["Lambda"]
    DF = model.gradients_FX(rho, detach_tensors=True, compute_D2FX=False)["DF"]
    mu = (1.0 / beta) * torch.log(Lambda**3 * rho) + DF
    return mu


# ------------------------------------------------------------------------------
# Batched coexistence system
# ------------------------------------------------------------------------------
def coex_equations_auto(rho_vec, model):
    """
    Batched coexistence equations.

    Parameters
    ----------
    rho_vec : tensor, shape (B, 2)
        [:, 0] = vapor guess
        [:, 1] = liquid guess

    Returns
    -------
    F : tensor, shape (B, 2)
        F[:, 0] = mu(rho_v) - mu(rho_l)
        F[:, 1] = p(rho_v)  - p(rho_l)
    """
    if rho_vec.ndim != 2 or rho_vec.shape[1] != 2:
        raise ValueError(f"rho_vec must have shape (B, 2), got {tuple(rho_vec.shape)}")

    rho_template = model.sol["rho_guess"]              # (B, 1, Nx)
    dx = model.mesh["dx"]
    mid = rho_template.shape[-1] // 2

    r_v = rho_vec[:, 0].view(-1, 1, 1)                 # (B, 1, 1)
    r_l = rho_vec[:, 1].view(-1, 1, 1)                 # (B, 1, 1)

    R_v = r_v * torch.ones_like(rho_template)          # (B, 1, Nx)
    R_l = r_l * torch.ones_like(rho_template)          # (B, 1, Nx)

    N_v = R_v.sum(dim=-1) * dx                         # (B, 1)
    N_l = R_l.sum(dim=-1) * dx                         # (B, 1)

    mu_v = model.GetChemPot(N_v, R_v).reshape(-1)      # (B,)
    mu_l = model.GetChemPot(N_l, R_l).reshape(-1)      # (B,)

    omega_v = model.GetOmega(R_v)[0]                   # expected (B, 1, Nx)
    omega_l = model.GetOmega(R_l)[0]

    p_v = -omega_v[:, 0, mid]                          # (B,)
    p_l = -omega_l[:, 0, mid]                          # (B,)

    return torch.stack([mu_v - mu_l, p_v - p_l], dim=-1)  # (B, 2)


def solve_newton_batched(
    coex_equations,
    rho_init,
    model,
    tol=1e-5,
    max_iter=1000,
    alpha=0.3,
    clamp_min=1e-15,
    clamp_max=1.0 - 1e-15,
    verbose=True,
):
    """
    Batched Newton-Raphson for coexistence equations.

    Parameters
    ----------
    rho_init : tensor, shape (B, 2)

    Returns
    -------
    rho : tensor, shape (B, 2)
    """
    td = sol_dtype(model.sol)
    dev = model.sol["device"]

    rho0 = rho_init.clone().detach().to(device=dev, dtype=td)
    if rho0.ndim != 2 or rho0.shape[1] != 2:
        raise ValueError(f"rho_init must have shape (B, 2), got {tuple(rho0.shape)}")

    batch_size = rho0.shape[0]

    for i in range(max_iter):
        rho = rho0.detach().clone().requires_grad_(True)    # (B, 2)
        F = coex_equations(rho, model)                      # (B, 2)

        J = torch.zeros((batch_size, 2, 2), dtype=td, device=dev)

        # Because samples are independent across batch, grad(sum(F[:, j]), rho)
        # gives per-sample gradients stacked in shape (B, 2).
        for j in range(2):
            grad_j = torch.autograd.grad(
                F[:, j].sum(),
                rho,
                retain_graph=True,
                materialize_grads=True,
            )[0]                                            # (B, 2)
            J[:, j, :] = grad_j

        # Solve J * delta = -F independently for each batch item.
        delta_rho = torch.linalg.solve(J, -F.unsqueeze(-1)).squeeze(-1)   # (B, 2)

        with torch.no_grad():
            rho0 = rho0 + alpha * delta_rho
            rho0[:, 0].clamp_(min=clamp_min, max=clamp_max)
            rho0[:, 1].clamp_(min=clamp_min, max=clamp_max)

        step_norm = torch.linalg.norm(delta_rho, dim=-1)    # (B,)
        max_step = step_norm.max().item()

        if verbose and (i % 10 == 0 or max_step < tol):
            print(
                f"iter={i:4d} | max||delta||={max_step:.3e} | "
                f"mean||delta||={step_norm.mean().item():.3e}"
            )

        if max_step < tol:
            if verbose:
                print(f"Converged in {i + 1} Newton iterations.")
            break

    return rho0.detach()


def batched_coexistence_temperatures(
    T_list,
    eq_params,
    sol_bulk=None,
    ml_state_dicts=None,
    mesh_bulk=None,
    sol=None,
    use_model=True,
    rho_init_pair=(1e-15, 1.0 - 1e-15),
    tol=1e-7,
    max_iter=100000,
    alpha=0.9,
    chunk_size=None,
    verbose=True,
):
    """
    Solve coexistence for many temperatures in batches.

    Returns
    -------
    rho_v : tensor, shape (len(T_list),)
    rho_l : tensor, shape (len(T_list),)
    """
    if mesh_bulk is None:
        base_sol = sol_bulk if sol_bulk is not None else sol
        if base_sol is None:
            raise ValueError("Either sol_bulk or sol must be provided to build the default mesh_bulk.")
        mesh_bulk = _default_bulk_mesh(eq_params, base_sol)

    if sol_bulk is None:
        sol_bulk = _default_sol_bulk(sol, mesh_bulk)

    td = sol_dtype(sol_bulk)
    dev = sol_bulk["device"]

    T_all = torch.as_tensor(T_list, dtype=td, device=dev)
    nT = T_all.numel()

    if chunk_size is None:
        chunk_size = nT

    rho_v_chunks = []
    rho_l_chunks = []

    for start in range(0, nT, chunk_size):
        stop = min(start + chunk_size, nT)
        T_chunk = T_all[start:stop]
        B = T_chunk.shape[0]

        rho_guess_b = _expand_rho_guess(sol_bulk["rho_guess"], B)
        sol_chunk = _clone_sol_with_rho_guess(sol_bulk, rho_guess_b)
        model = update_model(
            x_=mesh_bulk["x"],
            rho_=rho_guess_b,
            T_=T_chunk,
            eq_params_=eq_params,
            mesh_=mesh_bulk,
            sol_=sol_chunk,
            ml_state_dicts_=ml_state_dicts,
        )
        model.sol["USE_MODEL"] = bool(use_model)

        rho_init = torch.tensor(rho_init_pair, dtype=td, device=dev).view(1, 2).expand(B, -1).clone()

        if verbose:
            mode = "NN" if use_model else "MF"
            print(f"\nSolving coexistence batch {start}:{stop} ({mode})")
            print("Temperatures:", [float(t) for t in T_chunk])

        rho_sol = solve_newton_batched(
            coex_equations=coex_equations_auto,
            rho_init=rho_init,
            model=model,
            tol=tol,
            max_iter=max_iter,
            alpha=alpha,
            verbose=verbose,
        )  # (B, 2)

        rho_v_chunks.append(rho_sol[:, 0])
        rho_l_chunks.append(rho_sol[:, 1])

        if verbose:
            for T_val, rv, rl in zip(T_chunk.tolist(), rho_sol[:, 0].tolist(), rho_sol[:, 1].tolist()):
                print(f"T={T_val:.3f} -> rho_v={rv:.8f}, rho_l={rl:.8f}")

    rho_v = torch.cat(rho_v_chunks, dim=0)
    rho_l = torch.cat(rho_l_chunks, dim=0)
    return rho_v, rho_l


# ------------------------------------------------------------------------------
# Critical point (kept scalar)
# ------------------------------------------------------------------------------
def newton_critical_point(
    eq_params, mesh, sol, rho0, T0, max_iter=20, tol=1e-8, alpha=1.0, device=None, ml_state_dicts_=None
):
    """Solve for (rho_c, T_c) such that ∂p/∂rho=0 and ∂²p/∂rho²=0."""
    if device is None:
        device = sol["device"]
    device = torch.device(device)
    td = sol_dtype(sol)

    rho = torch.tensor(float(rho0), dtype=td, device=device)
    T = torch.tensor(float(T0), dtype=td, device=device)

    Vext_mesh = LJ93(mesh["x"], mesh["x_wall"], 1, 2)[None, ...].to(device) * 0

    for it in range(max_iter):
        rho_var = rho.clone().detach().requires_grad_(True)
        T_var = T.clone().detach().requires_grad_(True)

        beta = 1.0 / T_var
        eq_params_local = dict(eq_params)
        eq_params_local["beta"] = beta * torch.ones_like(eq_params_local["mu"], dtype=td)
        eq_params_local["Vext"] = Vext_mesh

        model = CDFT(eq_params_local, mesh, sol)
        sd = ml_state_dicts_
        setDNN(model, LR=0.0, state_dicts=sd)
        setDNNRep(model, LR=0.0, state_dict=sd["dnn_rep_fn"] if sd else None)
        setWDA(model, LR=0.0, modes=150, state_dict=sd["wda_fn"] if sd else None)

        if hasattr(model, "dnn_fn") and model.dnn_fn is not None:
            model.dnn_fn.eval()
        if hasattr(model, "dnn_rep_fn") and model.dnn_rep_fn is not None:
            model.dnn_rep_fn.eval()
        if hasattr(model, "wda_fn") and model.wda_fn is not None:
            model.wda_fn.eval()

        R1 = rho_var * torch.ones_like(model.sol["rho_guess"])
        omega = model.GetOmega(R1)[0]
        Nx = omega.shape[-1]
        p = -omega[0, 0, Nx // 2]

        p_prime = torch.autograd.grad(p, rho_var, create_graph=True)[0]
        p_second = torch.autograd.grad(p_prime, rho_var, create_graph=True)[0]

        f1, f2 = p_prime, p_second
        F = torch.stack([f1, f2])

        if F.norm().item() < tol:
            print(f"Converged in {it} iterations (function norm).")
            break

        df1_drho = torch.autograd.grad(f1, rho_var, retain_graph=True)[0]
        df1_dT = torch.autograd.grad(f1, T_var, retain_graph=True)[0]
        df2_drho = torch.autograd.grad(f2, rho_var, retain_graph=True)[0]
        df2_dT = torch.autograd.grad(f2, T_var)[0]

        J = torch.stack([
            torch.stack([df1_drho, df1_dT]),
            torch.stack([df2_drho, df2_dT]),
        ])

        delta = torch.linalg.solve(J, -F)
        rho = rho + alpha * delta[0]
        T = T + alpha * delta[1]

        print("Norm:", delta.norm().item())
        if delta.norm().item() < tol:
            print(f"Converged in {it + 1} iterations (step size).")
            break

    return rho.item(), T.item()
