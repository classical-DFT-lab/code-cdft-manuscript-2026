import torch

from libs.ml.surrogates import DNN, DNNRep
from libs.cdft_axisym.convolution import pair_operator
from libs.cdft_axisym.kernels import wda_modes, wda_kernel_3d, attr_kernel_3d, bh_diameter


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EXCESS FREE ENERGY ON THE (r, z) MESH
# Same functional as cdft_1d (WDA + Carnahan-Starling with the learned corrections, mean-field
# attraction with the learned g), with the radial kernels. Everything is in units of kT:
# FX returns beta*F_ex, mu_ex returns beta*mu_ex.
class AXISYM_FREE_ENERGY():
    def __init__(self, mesh, ml_state_dicts, T, use_dbh=1):
        self.mesh = mesh
        self.use_dbh = use_dbh
        dev, dt = mesh["device"], mesh["dtype"]
        sd = ml_state_dicts

        self.dnn_g_fn = DNN(input_dim=2, output_dim=1, hidden_dim=24, num_layers=6, dtype=dt).to(dev)
        self.dnn_g_fn.load_state_dict(sd["dnn_g_fn"])
        self.dnn_rep_fn = DNNRep(input_dim=2, output_dim=1, hidden_dim=24, num_layers=6, dtype=dt).to(dev)
        self.dnn_rep_fn.load_state_dict(sd["dnn_rep_fn"])
        for param in self.dnn_g_fn.parameters(): param.requires_grad = False
        for param in self.dnn_rep_fn.parameters(): param.requires_grad = False
        self.dnn_g_fn.eval()
        self.dnn_rep_fn.eval()

        # WDA kernel (temperature independent): assembled once
        modes = wda_modes(sd["wda_fn"]["blocks.0.spectral_conv.real_weight"].cpu())
        self.W_wda = pair_operator(mesh, wda_kernel_3d(modes), cutoff=3.0)
        self.W_one = self.W_wda @ torch.ones(mesh["nr"]*mesh["nz"], dtype=dt, device=dev)  # (nr*nz,)
        self.bypass_w = sd["wda_fn"]["blocks.0.linear_bypass.weight"].item()
        self.bypass_b = sd["wda_fn"]["blocks.0.linear_bypass.bias"].item()

        self.set_T(T)

    def set_T(self, T):
        self.T = T
        self.beta = 1. / T
        self.dbh = bh_diameter(T) if self.use_dbh else 1.
        # Attractive kernel depends on T through g(r, beta)
        self.W_att = pair_operator(self.mesh, attr_kernel_3d(self.dnn_g_fn, self.beta), cutoff=2.6)

    # REPULSIVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def rho_bar(self, rho):
        return (self.W_wda @ rho) / self.W_one + 1e-3 * (self.bypass_w * rho + self.bypass_b)  # (nr*nz,)

    def phiR(self, rho):
        rho_bar = self.rho_bar(rho)
        n = torch.pi / 6. * self.dbh**3 * rho_bar     # packing fraction
        om_n = torch.clamp(1 - n, min=1e-12)
        phiR_0 = rho_bar * n * (4 - 3*n) / om_n**2   # (nr*nz,)
        inputs = torch.stack([n, torch.full_like(n, self.beta)], dim=-1)  # (nr*nz, 2)
        phi_corr = self.dnn_rep_fn.net(inputs)[..., 0]
        return phiR_0 * (1. + 0.01*phi_corr)

    def FR(self, rho):
        return torch.sum(self.mesh["w"] * self.phiR(rho))

    # ATTRACTIVE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def FA(self, rho):
        return 0.5 * self.beta * torch.sum(self.mesh["w"] * rho * (self.W_att @ rho))

    def FX(self, rho):
        return self.FR(rho) + self.FA(rho)

    # beta*mu_ex = (1/w) dF_ex/drho; the attractive part is linear in rho
    def mu_ex(self, rho):
        with torch.enable_grad():
            rho_ = rho.detach().requires_grad_(True)
            DFR, = torch.autograd.grad(self.FR(rho_), rho_)
        return DFR / self.mesh["w"] + self.beta * (self.W_att @ rho)  # (nr*nz,)
