import numpy as np
import torch


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# AXISYMMETRIC MESH
# Cell-centred (r, z) mesh on [0, Rmax] x [0, Lz]: the axis r=0 and the wall z=0 are cell edges.
# Fields are stored raveled r-major, index = i*nz + j.
def axisym_mesh(Rmax, Lz, nr, nz, device, dtype=torch.float64):
    dr, dz = Rmax / nr, Lz / nz
    e = dr * np.arange(nr + 1)                       # (nr+1,) radial cell edges
    r = 0.5 * (e[:-1] + e[1:])                       # (nr,)
    z = (np.arange(nz) + 0.5) * dz                   # (nz,)
    w = np.repeat(np.pi * (e[1:]**2 - e[:-1]**2) * dz, nz)  # (nr*nz,) annular cell volumes
    mesh = {
        "r": torch.tensor(r, dtype=dtype, device=device),
        "z": torch.tensor(z, dtype=dtype, device=device),
        "dr": dr,
        "dz": dz,
        "w": torch.tensor(w, dtype=dtype, device=device),
        "nr": nr,
        "nz": nz,
        "Rmax": Rmax,
        "Lz": Lz,
        "device": device,
        "dtype": dtype,
    }
    return mesh


def integrate(mesh, f):
    return torch.sum(mesh["w"] * f)  # (,)


def mesh_coords(mesh):
    # Raveled cell-centre coordinates (numpy), same ordering as the fields
    R = np.repeat(mesh["r"].cpu().numpy(), mesh["nz"])  # (nr*nz,)
    Z = np.tile(mesh["z"].cpu().numpy(), mesh["nr"])    # (nr*nz,)
    return R, Z
