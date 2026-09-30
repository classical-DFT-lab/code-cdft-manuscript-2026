from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
import torch
import torch.nn.functional as torch_f


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# PAIR OPERATOR ON THE (r, z) MESH
# W rho = int K3(|x - x'|) rho(x') dx' for a radial kernel K3 truncated at cutoff: dense over the
# radial cells (exact annulus-annulus geometry), zero-padded FFT convolution along z.

def lens_area(d, a, b):
    # Area of the intersection of two disks of radii a, b with centres at distance d
    d, a, b = np.broadcast_arrays(np.asarray(d, float), np.asarray(a, float), np.asarray(b, float))
    mn = np.minimum(a, b)
    inside = d <= np.abs(a - b)
    lens = (~inside) & (d < a + b) & (mn > 0.0)
    dd = np.where(lens, d, 1.0)
    aa = np.where(lens, a, 1.0)
    bb = np.where(lens, b, 1.0)
    a1 = np.clip((dd*dd + aa*aa - bb*bb) / (2.0*dd*aa), -1.0, 1.0)
    a2 = np.clip((dd*dd + bb*bb - aa*aa) / (2.0*dd*bb), -1.0, 1.0)
    tri = (-dd + aa + bb) * (dd + aa - bb) * (dd - aa + bb) * (dd + aa + bb)
    val = aa*aa*np.arccos(a1) + bb*bb*np.arccos(a2) - 0.5*np.sqrt(np.maximum(tri, 0.0))
    return np.where(inside, np.pi*mn*mn, np.where(lens, val, 0.0))


# Transverse geometry table between the annular cells, reused by every kernel with the same cutoff
_lens_tables = {}
n_workers = min(16, os.cpu_count() or 1)

def _lens_table(e, S, nr):
    key = (nr, float(e[1]), float(S[-1]), S.size)
    if key not in _lens_tables:
        DD = np.empty((nr*nr, S.size))

        def fill(s0):
            s1 = min(s0 + 256, S.size)
            LQ = lens_area(S[s0:s1][None, None, :], e[:, None, None], e[None, :, None])  # (nr+1, nr+1, ds)
            DD[:, s0:s1] = (LQ[1:, 1:, :] - LQ[:-1, 1:, :] - LQ[1:, :-1, :] + LQ[:-1, :-1, :]).reshape(nr*nr, s1 - s0)

        with ThreadPoolExecutor(n_workers) as pool:
            list(pool.map(fill, range(0, S.size, 256)))
        _lens_tables[key] = DD
    return _lens_tables[key]


def pair_operator(mesh, kernel, cutoff, n_s=4096, n_fine=256):
    # kernel: numpy callable K3(u), evaluated at assembly time only
    c = float(cutoff)
    dz, nr, nz = mesh["dz"], mesh["nr"], mesh["nz"]
    Mc = int(np.ceil(c / dz - 1e-12))
    ms = np.arange(-Mc, Mc + 1)

    # Axial kernel integrated over the source cells, C[s, m] = int_{cell m} K3(sqrt(s^2 + zeta^2)) dzeta
    nf = max(2, 2 * (n_fine // 2))
    dzf = dz / nf
    zf = dzf * np.arange((Mc + 1) * nf + 1)
    S = np.linspace(0.0, c, n_s + 1)
    idx = nf // 2 + nf * np.arange(Mc + 1)
    C = np.zeros((S.size, 2*Mc + 1))
    s_chunk = max(1, int(8.0e6 / zf.size))
    for s0 in range(0, S.size, s_chunk):
        s1 = min(s0 + s_chunk, S.size)
        u = np.sqrt(S[s0:s1, None]**2 + zf[None, :]**2)
        g = np.where(u <= c, np.asarray(kernel(u), dtype=float), 0.0)
        G1 = np.zeros_like(g)
        G1[:, 1:] = np.cumsum(0.5 * (g[:, 1:] + g[:, :-1]), axis=1) * dzf
        G1h = G1[:, idx]                            # (chunk, Mc+1)
        C[s0:s1, Mc] = 2.0 * G1h[:, 0]
        pos = G1h[:, 1:] - G1h[:, :-1]
        C[s0:s1, Mc + 1:] = pos
        C[s0:s1, :Mc] = pos[:, ::-1]

    # Transverse integral over the annulus-annulus overlap, T[i, i', m] = 2 pi int s DD(s) C(s, m) ds
    e = mesh["dr"] * np.arange(nr + 1)
    wS = np.full(S.size, S[1] - S[0])
    wS[0] = wS[-1] = 0.5 * (S[1] - S[0])
    B = (S * wS)[:, None] * C                       # (n_s+1, 2Mc+1)
    T = 2.0 * np.pi * (_lens_table(e, S, nr) @ B).reshape(nr, nr, 2*Mc + 1)
    T = 0.5 * (T + T.transpose(1, 0, 2)[:, :, ::-1])

    # Spectrum along z (linear convolution through zero padding)
    n_fft = nz + 2*Mc
    ker = np.zeros((nr, nr, n_fft))
    for k, m in enumerate(ms):
        ker[:, :, (-int(m)) % n_fft] += T[:, :, k]
    ker_hat = np.fft.rfft(ker, axis=2)              # (nr, nr, n_fft//2+1)
    area = np.pi * (e[1:]**2 - e[:-1]**2)[:, None]  # (nr, 1) ring areas of the field cells

    # Bulk integral of the truncated kernel, 4 pi int u^2 K3 du
    uf = np.linspace(0.0, c, 4097)
    su = np.ones(4097)
    su[1:-1:2] = 4.0
    su[2:-1:2] = 2.0
    su *= (uf[1] - uf[0]) / 3.0
    ku = np.where(uf <= c, np.asarray(kernel(uf), dtype=float), 0.0)
    a_bulk = float(4.0 * np.pi * np.sum(su * uf * uf * ku))

    dev, dt = mesh["device"], mesh["dtype"]
    return ZConv(torch.tensor(ker_hat, dtype=torch.complex128, device=dev), n_fft, nr, nz, Mc,
                 torch.tensor(area, dtype=dt, device=dev), a_bulk)


class ZConv():
    def __init__(self, ker_hat, n_fft, nr, nz, Mc, area, a_bulk):
        self.ker_hat = ker_hat      # (nr, nr, n_fft//2+1)
        self.n_fft = n_fft
        self.nr = nr
        self.nz = nz
        self.Mc = Mc
        self.area = area            # (nr, 1)
        self.a_bulk = a_bulk

    def __matmul__(self, rho):
        x = rho.reshape(self.nr, self.nz)
        buf = torch_f.pad(x, (self.Mc, self.n_fft - self.Mc - self.nz))  # (nr, n_fft)
        xh = torch.fft.rfft(buf, dim=1)
        oh = torch.einsum("oif,if->of", self.ker_hat, xh)
        out = torch.fft.irfft(oh, n=self.n_fft, dim=1)
        out = out[:, self.Mc:self.Mc + self.nz] / self.area  # (nr, nz)
        return out.reshape(-1)

    def __mul__(self, c):
        return ZConv(self.ker_hat * c, self.n_fft, self.nr, self.nz, self.Mc, self.area, self.a_bulk * c)

    __rmul__ = __mul__
