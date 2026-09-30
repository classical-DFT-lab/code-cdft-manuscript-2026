import numpy as np
import torch

def LossL2(U_fwd, U_data):
    """
    Loss function to compute the loss between the forward solution and the target data.
    """
    return torch.mean(torch.abs(U_fwd - U_data)**2) / torch.mean(torch.abs(U_data)**2)


def LossL1(U_fwd, U_data):
    """
    Loss function to compute the loss between the forward solution and the target data.
    """
    return torch.mean(torch.abs(U_fwd - U_data)) / torch.mean(torch.abs(U_data))


def SpectralLoss(U_fwd, U_data):
    pred_f = torch.fft.rfft(U_fwd, dim=1)
    target_f = torch.fft.rfft(U_data, dim=1)
    # freq_weights = 2. - torch.linspace(0., 2., pred_f.shape[-1])[None,:]  # Emphasize lower frequencies
    # freq_weights = torch.linspace(1., 1.5, pred_f.shape[-1])[None,:]  # Emphasize higher frequencies
    freq_weights = 1.  # Dont emphasize higher frequencies
    return torch.mean(torch.square(pred_f - target_f)* (freq_weights))# / (torch.mean(U_data**2))


def LossGradL2(U_fwd, U_data):
    """
    Loss function to compute the loss between the forward solution and the target data.
    """
    dU_fwd = U_fwd[..., 1:] - U_fwd[..., :-1]
    dU_data = U_data[..., 1:] - U_data[..., :-1]

    Loss1 = torch.mean(torch.abs(U_fwd - U_data)**2) / torch.mean(torch.abs(U_data)**2)
    Loss2 = torch.mean(torch.abs(dU_fwd - dU_data)**2) / torch.mean(torch.abs(U_data)**2)
    return Loss1 + 0.5*Loss2


class MMD2Loss:
    """
    Multi-bandwidth MMD^2 between the forward solution and the target profile, normalised
    per state by MMD^2(0, target) and averaged over the batch:
        MMD^2(rho, ref) = (rho - ref)^T G (rho - ref),  G = sum_ks dx^2 exp(-(x-y)^2/(2 ks^2))/ks
    Build once on the mesh and pass the instance as sol["LOSS"].
    """
    def __init__(self, x, dx, bandwidths=tuple(0.5*k for k in range(1, 11)) + (0.01,)):
        z = np.asarray(x.detach().cpu(), dtype=np.float64)
        w = float(dx.item() if torch.is_tensor(dx) else dx)
        D2 = (z[:, None] - z[None, :])**2
        G = np.zeros_like(D2)
        for ks in bandwidths:
            G += (w*w) * np.exp(-D2 / (2.*ks*ks)) / ks
        self.G = torch.tensor(G, dtype=torch.float64, device=x.device)

    def _quad(self, d):
        # d: (B, 1, Nx) -> (B,)
        return torch.einsum("bui,ij,buj->b", d, self.G.to(d.dtype), d)

    def __call__(self, U_fwd, U_data):
        val = self._quad(U_fwd - U_data)
        norm = self._quad(U_data)
        return torch.mean(val / norm)
