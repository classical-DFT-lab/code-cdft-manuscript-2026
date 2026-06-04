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