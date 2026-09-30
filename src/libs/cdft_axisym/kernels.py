import numpy as np
import torch


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# RADIAL KERNELS FROM THE PLANAR ONES
# The planar model sees every radial kernel K3 through its transverse integral
# K1(x) = 2 pi int_|x|^inf K3(t) t dt, so K3(r) = -K1'(r) / (2 pi r).

# Same real-space window as SpectralConv1d.forward, normalised to unit integral
def wda_modes(real_weight, Nx0=300):
    wk = np.asarray(real_weight, dtype=np.float64).reshape(-1)  # (modes,)
    x_f = np.linspace(-1., 1., Nx0)
    dx_f = (x_f[1] - x_f[0]) * (1 - Nx0 % 2)
    H = np.roll(np.exp(-50*(x_f - 0.5*dx_f)**2), (Nx0 + 1)//2)
    w_hat = np.fft.rfft(H * np.fft.irfft(wk, n=Nx0)).real[:wk.size]
    return w_hat / (1e-12 + w_hat[0])


# K1(x) = (g_0 + 2 sum_m g_m cos(k_m x)) / L  ->  K3(r) = sum_m g_m k_m sin(k_m r) / (pi L r)
def wda_kernel_3d(modes, L=60., cutoff=3.0, n_tab=4001):
    m = np.arange(1, modes.size)
    km = 2.0 * np.pi * m / L
    r_tab = np.linspace(1e-3, cutoff, n_tab)
    K3_tab = (np.sin(km[None, :] * r_tab[:, None]) / r_tab[:, None] * (modes[1:] * km)[None, :]).sum(axis=1) / (np.pi * L)
    return lambda r: np.interp(np.asarray(r, float), r_tab, K3_tab, right=0.0)


# Planar attractive kernel (see cdft_1d/attractive.py), differentiable in r
def psi_attr_planar(r, sigma=1., eps=1., cutoff=2.5):
    r = torch.abs(r)
    smooth_width = 0.01
    comp1 = sigma / torch.clamp(r, min=1e-6)
    comp_cutoff = sigma / cutoff
    in_particle = torch.sigmoid(-(r - sigma) / smooth_width)
    mask = torch.sigmoid(-(r - cutoff) / smooth_width)
    attractive = 4.0*torch.pi*eps*sigma**2 * (comp1**10/5.0 - comp1**4/2.0) * (1.0 - in_particle)
    attractive_cutoff = 4.0*torch.pi*eps*sigma**2 * (comp_cutoff**10/5.0 - comp_cutoff**4/2.0) * (1.0 - in_particle)
    in_particle_term = -6.0*torch.pi/5.0 * eps*sigma**2 * in_particle
    return (attractive - attractive_cutoff)*mask + in_particle_term


# K1(r) = psi(r) (1 + 1e-3 g(r, beta)) differentiated by autograd
def attr_kernel_3d(dnn_g_fn, beta, cutoff=2.6, n_tab=2048):
    dev = next(dnn_g_fn.parameters()).device
    r = torch.linspace(1e-3, cutoff, n_tab, dtype=torch.float64, device=dev, requires_grad=True)
    g = dnn_g_fn(torch.stack([r, torch.full_like(r, beta)], dim=-1))[..., 0]  # (n_tab,)
    K1 = psi_attr_planar(r) * (1 + 1e-3*g)
    dK1, = torch.autograd.grad(K1.sum(), r)
    K3_tab = (-dK1 / (2.0*torch.pi*r)).detach().cpu().numpy()
    r_tab = np.linspace(1e-3, cutoff, n_tab)
    return lambda r_: np.interp(np.asarray(r_, float), r_tab, K3_tab, right=0.0)


# Barker-Henderson diameter, same quadrature as REPULSIVE_FREE_ENERGY.bh_diameter
def bh_diameter(T):
    grid = np.linspace(1e-6, 1., 100)
    dgrid = grid[1] - grid[0]
    u = 4*((1/grid)**12 - (1/grid)**6)
    return float(np.sum(1 - np.exp(-u/T)) * dgrid)
