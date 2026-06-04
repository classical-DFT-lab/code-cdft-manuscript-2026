import torch
import torch.nn as nn
import torch.fft
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl
from libs.utils import NaturalCubicSpline, linear_interpolation, batched_linear_interpolation


class SpectralConv1d(nn.Module):
    input_dim : int
    output_dim : int
    modes : int
    real_weight : nn.Parameter
    imag_weight : nn.Parameter

    def __init__(self, input_dim, output_dim, modes0, mesh, tabulated=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.modes0 = modes0
        self.tabulated = tabulated
        self.bulk_comp = mesh.get("BULK_COMP", False)  # 0D bulk: allow Nx < modes0
        self.L0 = 30.0   # Half-length of the domain used to train the model
        self.Nx0 = 300   # Discretiz. of the domain used to train the model
        self.L_new = mesh["L"]

        scale = 1. / (input_dim * output_dim)

        #self.real_weight = nn.Parameter(torch.empty(output_dim, input_dim, modes).uniform_(1. - 0.001*scale, 1. + 0.001*scale))
        self.real_weight = nn.Parameter(torch.ones(output_dim, input_dim, modes0))
        self.imag_weight = nn.Parameter(torch.empty(output_dim, input_dim, modes0).uniform_(-0.*scale, 0.*scale)).requires_grad_(False)
        self.auxiliary = nn.Parameter(torch.tensor([1.0, 10.0], dtype=torch.double))


    def complex_mult1d(self, x_complex, w_complex):
        # x_complex: (B, input_dim, modes)
        # w_complex: (output_dim, input_dim, modes)
        return torch.einsum("...jm,ijm->...im", x_complex, w_complex)
    
    
    # @torch.jit.trace
    def forward(self, x:torch.Tensor,):
        # x: (batch, input_dim, Nx)
        Nx = x.shape[-1]

        Nx0 = self.Nx0
        x_filter = torch.linspace(-1, 1., Nx0, dtype=torch.double, device=x.device)
        dx_filter = (x_filter[1]-x_filter[0])*(1 - Nx0%2)
        H_fwd = torch.roll(torch.exp(-50*(x_filter-0.5*dx_filter)**2), (Nx0+1)//2)
        real_weight_fwd = torch.fft.irfft(self.real_weight,n=Nx0, dim=-1)
        real_weight_filtered = torch.fft.rfft(H_fwd[None, None, :]*real_weight_fwd, dim=-1).real [..., :self.modes0] 
        real_weight0, imag_weight0 = real_weight_filtered, self.imag_weight

        # Common spacing from the *physical* domain length 2*L
        dk0   = 2*torch.pi / (2*self.L0)
        dknew = 2*torch.pi / (2*self.L_new)

        nf0      = torch.arange(self.modes0, device=x.device)
        k0       = (nf0 * dk0).double()                                       # (modes0,)

        Nf_rho   = x.shape[-1]//2 + 1
        nf_rho   = torch.arange(Nf_rho, device=x.device)
        k_rho    = (nf_rho * dknew).double()                                 # matches rfft(x)

        # Choose how many new bins you want (round, don’t truncate)
        NfM_rho  = int(round(self.modes0 * (self.L_new / self.L0)))
        NfM_     = min(self.modes0, NfM_rho)                                    # keep >=2
        nf_tilde = torch.arange(NfM_, device=x.device)
        k_tilde  = (nf_tilde * dknew).double()                               # (NfM_,)

        x_ft = torch.fft.rfft(x, dim=-1)  # -> (B, C, Nf_rho)
        assert x_ft.shape[-1] == Nf_rho, f"Expected {Nf_rho} frequencies, but got {x_ft.shape[-1]}"

        # BULK_COMP or Nf_rho < modes0: small grid, interpolate weights to k_rho
        if self.bulk_comp or Nf_rho < self.modes0:
            real_weight = batched_linear_interpolation(k_rho[None, None, :], k0[None, None, :], real_weight0)
            imag_weight = real_weight * 0
            x_ft_trunc = x_ft
        else:
            # Standard path: interpolate to k_tilde, then x_ft to k_tilde or truncate
            real_weight = batched_linear_interpolation(k_tilde[None, None, :], k0[None, None, :], real_weight0)
            imag_weight = real_weight * 0
            if (k_rho[: self.modes0] == k0).all():
                x_ft_trunc = x_ft[..., : self.modes0]
            else:
                x_ft_trunc = batched_linear_interpolation(k_tilde[None, None, :], k_rho[None, None, :], x_ft)

        weight = torch.complex(real_weight, imag_weight) / (1e-12 + real_weight[0, 0, 0])
        out_ft_trunc = self.complex_mult1d(x_ft_trunc, weight)
        x_out = torch.fft.irfft(out_ft_trunc, n=x.shape[-1], dim=-1)

        return x_out



class WDABlock1d(nn.Module):
    def __init__(self, input_dim, output_dim, modes, mesh, activation=nn.GELU(), use_bias=True):
        super().__init__()
        self.spectral_conv = SpectralConv1d(input_dim, output_dim, modes, mesh)
        self.linear_bypass = nn.Conv1d(input_dim, output_dim, kernel_size=1, bias=use_bias)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.spectral_conv(x) + 0.001*self.linear_bypass(x))


class WDA1d(nn.Module):
    def __init__(self, input_dim, output_dim, modes, width, num_blocks, mesh, activation=nn.GELU(), use_bias=True):
        super().__init__()
        self.lifting = nn.Conv1d(input_dim, width, kernel_size=1, bias=use_bias)
        self.blocks = nn.ModuleList([
            WDABlock1d(width, width, modes, mesh, activation=activation, use_bias=use_bias)
            for _ in range(num_blocks)
        ])
        self.projection = nn.Conv1d(width, output_dim, kernel_size=1, bias=use_bias)

    # #@torch.jit.trace
    def forward(self, x):
        #x = self.lifting(x)
        for block in self.blocks:
            x = block(x)
        #x = self.projection(x)
        return x