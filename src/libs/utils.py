import os
import warnings
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn


def pick_torch_device(kind: Optional[str] = None) -> torch.device:
    """
    Resolve a torch.device from a short name.

    Args:
        kind: ``"auto"`` (CUDA if available, else MPS on Apple Silicon if available, else CPU),
            ``"cuda"``, ``"mps"``, or ``"cpu"``. If ``None``, uses env ``TORCH_DEVICE`` or ``"auto"``.

    Note:
        MPS has limited float64 support; prefer ``TORCH_DTYPE = torch.float32`` when using ``mps``.
    """
    k = (kind if kind is not None else os.environ.get("TORCH_DEVICE", "auto")).strip().lower()
    mps_backend = getattr(torch.backends, "mps", None)
    mps_ok = mps_backend is not None and mps_backend.is_available()

    if k == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_ok:
            return torch.device("mps")
        return torch.device("cpu")
    if k == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        warnings.warn("CUDA requested but not available; using CPU.", UserWarning)
        return torch.device("cpu")
    if k == "mps":
        if mps_ok:
            return torch.device("mps")
        warnings.warn("MPS requested but not available; using CPU.", UserWarning)
        return torch.device("cpu")
    if k == "cpu":
        return torch.device("cpu")
    raise ValueError(
        f"Unknown device kind {kind!r}; use 'auto', 'cuda', 'mps', or 'cpu' "
        "(or set env TORCH_DEVICE)."
    )


def is_cuda_device(device: Union[torch.device, str]) -> bool:
    return torch.device(device).type == "cuda"


def resolve_training_device(script_default: str = "auto") -> torch.device:
    """``pick_torch_device`` using env ``TORCH_DEVICE`` if set, else ``script_default``."""
    return pick_torch_device(os.environ.get("TORCH_DEVICE", script_default))


def state_dict_to_cpu(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Copy ``state_dict`` tensors to CPU (detached clones) for ``torch.save`` portability
    across CUDA / MPS / CPU machines.
    """
    out: Dict[str, Any] = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().clone().contiguous()
        else:
            out[k] = v
    return out


def save_state_dict_cpu(state_dict: Dict[str, Any], path: str) -> None:
    """``torch.save`` a state dict with all tensor parameters stored on CPU."""
    torch.save(state_dict_to_cpu(state_dict), path)


def dataframe_elementwise_map(df: Any, func: Any) -> Any:
    """
    Apply ``func`` to every cell of a pandas ``DataFrame``.

    Pandas 2.2 removed ``DataFrame.applymap`` in favor of ``DataFrame.map``.
    Uses ``.map`` when present, otherwise falls back to ``.applymap`` (older pandas).
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for dataframe_elementwise_map") from None
    if not isinstance(df, pd.DataFrame):
        raise TypeError("dataframe_elementwise_map expects a pandas DataFrame")
    mapper = getattr(df, "map", None)
    if callable(mapper):
        return mapper(func)
    applymap = getattr(df, "applymap", None)
    if callable(applymap):
        return applymap(func)
    raise AttributeError(
        "DataFrame has neither .map nor .applymap; install pandas >= 2.2."
    )


def tensors_to_cpu_for_storage(obj: Any) -> Any:
    """
    Recursively move ``torch.Tensor`` leaves to CPU (detached clones) for pickle and
    nested structures (dicts, lists, tuples). Numpy arrays and scalars are unchanged.

    For ``pandas.DataFrame``, applies ``tensors_to_cpu_for_storage`` to every cell
    (via ``dataframe_elementwise_map`` so pandas 2.2+ and older releases both work).
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone().contiguous()
    if isinstance(obj, dict):
        return {k: tensors_to_cpu_for_storage(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(tensors_to_cpu_for_storage(x) for x in obj)
    if isinstance(obj, np.ndarray):
        return obj
    try:
        import pandas as pd
    except ImportError:
        pd = None  # type: ignore
    if pd is not None and isinstance(obj, pd.DataFrame):
        return dataframe_elementwise_map(obj.copy(), tensors_to_cpu_for_storage)
    return obj


def profile_nt_index_key(n, t, t_decimals: int = 2):
    """
    Stable ``(N, T)`` for profile DataFrames with a ``(N, T)`` MultiIndex.

    ``T`` is rounded to ``t_decimals`` so float32/MPS tensor round-trips match pandas keys.
    ``N`` is cast to ``int`` (particle counts are integral).
    """
    return int(n), round(float(t), t_decimals)


def normalize_profiles_nt_multiindex(df: Any) -> Any:
    """
    Rebuild a 2-level ``(N, T)`` MultiIndex using :func:`profile_nt_index_key`.
    No-op if ``df`` is not a DataFrame or does not have a 2-level MultiIndex.
    """
    try:
        import pandas as pd
    except ImportError:
        return df
    if not isinstance(df, pd.DataFrame) or not isinstance(df.index, pd.MultiIndex):
        return df
    if df.index.nlevels != 2:
        return df
    names = list(df.index.names)
    tuples = [profile_nt_index_key(n, t) for n, t in df.index]
    out = df.copy()
    out.index = pd.MultiIndex.from_tuples(tuples, names=names)
    return out


def sol_dtype(sol):
    """Tensor dtype for numerical work; default matches previous torch.double behavior."""
    return sol.get("dtype", torch.float64)


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
        fp: (..., N )   Values on xp (same leading shape as xp). Complex ``fp`` is
                        supported by interpolating real and imaginary parts separately
                        (``torch.gather`` does not support complex dtypes on all devices).
        assume_sorted:  If False, will sort xp/fp along last dim first.
        eps:            Optional small clamp added to denominator to avoid div-by-zero
                        when xp has duplicate neighbors.

    Returns:
        (..., Nx) interpolated values
    """
    if xp.size(-1) < 2:
        raise ValueError("xp must have at least 2 points along the last dimension.")

    if fp.is_complex():
        return torch.complex(
            batched_linear_interpolation(
                x, xp, torch.real(fp), assume_sorted=assume_sorted, eps=eps
            ),
            batched_linear_interpolation(
                x, xp, torch.imag(fp), assume_sorted=assume_sorted, eps=eps
            ),
        )

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


def linear_interpolate_1d_unsorted(
    x_query: torch.Tensor,
    xp: torch.Tensor,
    fp: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    1D linear interpolation where ``xp`` need not be sorted.

    ``torch.searchsorted`` (used in :func:`linear_interpolation`) requires increasing
    ``xp``. MD/CSV profiles often store ``x`` decreasing toward the bulk; without
    sorting, indices are wrong and values become NaN.
    """
    xq = x_query.reshape(-1)
    xk = xp.reshape(-1)
    fk = fp.reshape(-1)
    if xk.numel() != fk.numel():
        raise ValueError("xp and fp must have the same number of elements")
    order = torch.argsort(xk)
    xs = xk[order]
    fs = fk[order]
    out = batched_linear_interpolation(
        xq.unsqueeze(0),
        xs.unsqueeze(0),
        fs.unsqueeze(0),
        assume_sorted=True,
        eps=eps,
    ).squeeze(0)
    return out.reshape(x_query.shape)


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
    def __init__(self, x: torch.Tensor, y: torch.Tensor, dtype: Optional[torch.dtype] = None):
        super().__init__()
        dt = dtype if dtype is not None else torch.float64
        x = x.detach().reshape(-1).to(dt)
        y = y.detach().reshape(-1).to(dt)

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
        A = torch.zeros((N, N), dtype=dt, device=x.device)
        rhs = torch.zeros((N,),    dtype=dt, device=x.device)
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
