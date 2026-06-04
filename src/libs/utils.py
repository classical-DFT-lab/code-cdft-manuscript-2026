import torch
import torch.nn as nn


def linear_interpolation(x, xp, fp):
    """
    Perform linear interpolation for x given data points (xp, fp).

    Args:
        x (torch.Tensor): Query points for interpolation (BS,).
        xp (torch.Tensor): Known x-coordinates (sorted) (Total_BS,).
        fp (torch.Tensor): Known function values at xp (Total_BS,).

    Returns:
        torch.Tensor: Interpolated values (BS,).
    """
    x = x#.unsqueeze(-1)  # Ensure correct shape (Nx)
    
    # Find indices of nearest neighbors
    idx = torch.searchsorted(xp, x) - 1
    idx = idx.clamp(0, len(xp) - 2)  # Ensure valid indices

    x0, x1 = xp[idx], xp[idx + 1]
    f0, f1 = fp[idx], fp[idx + 1]

    # Linear interpolation formula
    return f0 + (f1 - f0) * (x - x0) / (x1 - x0)

def batched_linear_interpolation(x, xp, fp, *, assume_sorted=True, eps=0.0):
    """
    Linear interpolation for batched inputs.

    Args:
        x:  (..., Nx)   Query points.
        xp: (..., N )   Grid x-coordinates (monotonic along last dim).
        fp: (..., N )   Values on xp (same leading shape as xp).
        assume_sorted:  If False, will sort xp/fp along last dim first.
        eps:            Optional small clamp added to denominator to avoid div-by-zero
                        when xp has duplicate neighbors.

    Returns:
        (..., Nx) interpolated values
    """
    if xp.size(-1) < 2:
        raise ValueError("xp must have at least 2 points along the last dimension.")

    # Optionally sort xp/fp (robust if xp might not be sorted)
    if not assume_sorted:
        xp, sort_idx = torch.sort(xp, dim=-1)
        fp = torch.gather(fp, dim=-1, index=sort_idx)

    # Find interval indices for each x
    # searchsorted broadcasts over leading dims; output has shape (..., Nx)
    idx = torch.searchsorted(xp, x, right=False) - 1
    # Keep indices in [0, N-2]
    idx = idx.clamp(0, xp.size(-1) - 2)

    # Gather neighbors along the last dimension (works for arbitrary batch dims)
    x0 = torch.gather(xp, dim=-1, index=idx)
    x1 = torch.gather(xp, dim=-1, index=idx + 1)
    f0 = torch.gather(fp, dim=-1, index=idx)
    f1 = torch.gather(fp, dim=-1, index=idx + 1)

    # Avoid division by zero if xp has equal neighbors (rare but possible)
    denom = (x1 - x0)
    if eps != 0.0:
        denom = denom.clamp_min(eps)

    # Linear interpolation
    out = f0 + (f1 - f0) * (x - x0) / denom

    # Optional: exact endpoint handling (e.g., x < xp[...,0] or x > xp[..., -1])
    # Here we do clamp-to-endpoints; replace with extrapolation if you prefer.
    # out = torch.where(x < xp.select(-1, 0).unsqueeze(-1), fp.select(-1, 0).unsqueeze(-1), out)
    out = torch.where(x < xp.select(-1, 0).unsqueeze(-1), fp.select(-1, 0).unsqueeze(-1), out)
    out = torch.where(x > xp.select(-1, xp.size(-1) - 1).unsqueeze(-1),
                    #   fp.select(-1, fp.size(-1) - 1).unsqueeze(-1), 
                      0.,
                      out)

    return out

def find_local_maxima_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Given a 1D tensor of shape (Nx,), return a tensor of local maxima values.
    A local maximum is greater than its immediate neighbors.
    """
    if x.ndim != 1:
        raise ValueError("Input tensor must be 1D.")

    if x.shape[0] < 3:
        return torch.tensor([], dtype=x.dtype)  # Not enough points for local maxima

    # Compare each point with its neighbors
    left  = x[:-2]
    mid   = x[1:-1]
    right = x[2:]

    # Create mask for local maxima
    mask = (mid > left) & (mid > right)

    # Return the local maxima
    return mid[mask]



def move_to(obj, device):
  if torch.is_tensor(obj):
    return obj.to(device)
  elif isinstance(obj, dict):
    res = {}
    for k, v in obj.items():
        #try:
            res[k] = move_to(v, device)
        #except:
        #   continue
    return res
  elif isinstance(obj, list):
    res = []
    for v in obj:
      res.append(move_to(v, device))
    return res
  else:
    raise TypeError("Invalid type for move_to")
  


@torch.jit.script
def _select1d(v: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    # TorchScript-friendly 1D gather with arbitrary-shaped indices
    flat = torch.index_select(v, 0, idx.reshape(-1))
    return flat.reshape(idx.shape)

class NaturalCubicSpline(nn.Module):
    """
    Natural cubic spline y(x) through sorted 1D knots (x_i, y_i).
    Fit once from 1D tensors x[N], y[N]. Evaluate on any shape of xq
    (e.g., [B, M]) and get yq with the same shape.
    """
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        super().__init__()
        x = x.detach().reshape(-1).to(torch.double)
        y = y.detach().reshape(-1).to(torch.double)

        # sort & checks
        idx = torch.argsort(x)
        x = x.index_select(0, idx)
        y = y.index_select(0, idx)
        if x.numel() < 2:
            raise ValueError("Need at least two points.")
        if torch.any(torch.diff(x) <= 0):
            raise ValueError("x must be strictly increasing.")

        N = x.numel()
        h = torch.diff(x)                          # [N-1]
        delta = torch.diff(y) / h                  # [N-1]

        # Build tridiagonal system A m = rhs (natural BCs)
        A = torch.zeros((N, N), dtype=torch.double, device=x.device)
        rhs = torch.zeros((N,),    dtype=torch.double, device=x.device)
        A[0,0] = 1.0
        A[-1,-1] = 1.0
        A[1:-1, 0:-2] += torch.diag(h[:-1])
        A[1:-1, 1:-1] += torch.diag(2*(h[:-1] + h[1:]))
        A[1:-1, 2:   ] += torch.diag(h[1:])
        rhs[1:-1] = 6*(delta[1:] - delta[:-1])

        m = torch.linalg.solve(A, rhs)             # [N]

        # Buffers
        self.register_buffer("xk", x)
        self.register_buffer("yk", y)
        self.register_buffer("mk", m)

    def _interval_indices(self, xq: torch.Tensor) -> torch.Tensor:
        # returns i with same shape as xq, clamped to [0, N-2]
        i = torch.searchsorted(self.xk, xq, right=True) - 1
        i = torch.clamp(i, 0, self.xk.numel() - 2)
        return i

    def forward(self, xq: torch.Tensor) -> torch.Tensor:
        """
        Evaluate on any shape of xq (e.g., [B, M]) and return same shape.
        """
        xq = xq.to(self.xk.dtype)
        i = self._interval_indices(xq)

        xi  = _select1d(self.xk, i)
        xi1 = _select1d(self.xk, i+1)
        yi  = _select1d(self.yk, i)
        yi1 = _select1d(self.yk, i+1)
        mi  = _select1d(self.mk, i)
        mi1 = _select1d(self.mk, i+1)

        h   = xi1 - xi
        t   = (xq - xi) / h

        A = 1.0 - t
        B = t
        C = ((A**3 - A) * (h**2)) / 6.0
        D = ((B**3 - B) * (h**2)) / 6.0
        return A*yi + B*yi1 + C*mi + D*mi1

    @torch.jit.export
    def dy_dx(self, xq: torch.Tensor) -> torch.Tensor:
        xq = xq.to(self.xk.dtype)
        i = self._interval_indices(xq)

        xi  = _select1d(self.xk, i)
        xi1 = _select1d(self.xk, i+1)
        yi  = _select1d(self.yk, i)
        yi1 = _select1d(self.yk, i+1)
        mi  = _select1d(self.mk, i)
        mi1 = _select1d(self.mk, i+1)

        h   = xi1 - xi
        t   = (xq - xi) / h

        dA_dt = -1.0
        dB_dt =  1.0
        dC_dt = ((3.0*(1.0 - t)**2*(-1.0) - (-1.0)) * (h**2)) / 6.0
        dD_dt = ((3.0*(t**2)        - 1.0)          * (h**2)) / 6.0

        ds_dt = dA_dt*yi + dB_dt*yi1 + dC_dt*mi + dD_dt*mi1
        return ds_dt / h
