import torch
from torch import nn


from libs.ml.wda import WDA1d


def load_ml_state_dicts(datadir, device):
    """
    Load all ML surrogate state dicts from disk once. Use with setDNN/setDNNRep/setWDA
    to avoid repeated disk I/O when creating many models (e.g. bulk coexistence loop).
    Returns dict with keys: dnn_fn, dnn_g_fn, dnn_rep_fn, wda_fn.
    """
    base = datadir + "ml_model/ml_dicts/"
    return {
        "dnn_fn": torch.load(base + "dnn_fn.dict", map_location=device),
        "dnn_g_fn": torch.load(base + "dnn_g_fn.dict", map_location=device),
        "dnn_rep_fn": torch.load(base + "dnn_rep_fn.dict", map_location=device),
        "wda_fn": torch.load(base + "wda_fn.dict", map_location=device),
    }


class DNN(nn.Module):
    def __init__(self, input_dim=1, output_dim=3, hidden_dim=8, num_layers=3):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim).double())
        layers.append(nn.GELU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim).double())
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    #@torch.compile
    def forward(self, x):
        return self.net(x)


class DNNRep(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim).double())
        layers.append(nn.GELU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim).double())
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x): #in: (B, 1, Nx) 
        x_ = torch.einsum("bui->biu", x)  # (B, Nx, 1)
        x_out_ = self.net(x_)  # (B, Nx, 1)
        x_out = torch.einsum("biu->bui", x_out_)  # (B, 1, Nx)
        return x_out
    
def setDNN(dft_model, LR, state_dicts=None):
    """
    Args:
        dft_model: CDFT model to attach DNN surrogates to
        LR: learning rate (unused when not training)
        state_dicts: optional dict with 'dnn_fn' and 'dnn_g_fn' keys; if provided, use instead of loading from disk
    """
    B = dft_model.mesh["BS"]
    dnn_fn = DNN(input_dim=1, output_dim=6, hidden_dim=24, num_layers=6).double().to(dft_model.sol["device"])
    dnn_g_fn = DNN(input_dim=1, output_dim=1, hidden_dim=24, num_layers=6).double().to(dft_model.sol["device"])
    dft_model.dnn_fn = torch.jit.trace(dnn_fn, torch.zeros([B, 1], dtype=torch.double, device=dft_model.sol["device"]))
    dft_model.dnn_g_fn = torch.jit.trace(dnn_g_fn, torch.zeros([B, 1], dtype=torch.double, device=dft_model.sol["device"]))

    # Optimizer for DNN model --------------
    dft_model.optimizer_dnn = torch.optim.Adam(
                                                [
                                                    {"params": dft_model.dnn_fn.parameters(), "lr": LR},
                                                    {"params": dft_model.dnn_g_fn.parameters(), "lr": 0.1*LR},
                                                ],
                                                weight_decay=0*1e-5,
                                              )
    
    #Scheduler - ExpDecay
    dft_model.scheduler_dnn = torch.optim.lr_scheduler.ExponentialLR(dft_model.optimizer_dnn, gamma=0.95)        
    LRs = [dft_model.optimizer_dnn.param_groups[0]["lr"],]

    if state_dicts is not None:
        dft_model.dnn_fn.load_state_dict(state_dicts["dnn_fn"])
        dft_model.dnn_g_fn.load_state_dict(state_dicts["dnn_g_fn"])
    elif dft_model.sol["RESTART_ML_MODEL"]:
        dft_model.dnn_fn.load_state_dict(torch.load(dft_model.sol["datadir"] + "ml_model/ml_dicts/dnn_fn.dict", map_location=dft_model.sol["device"]))
        dft_model.dnn_g_fn.load_state_dict(torch.load(dft_model.sol["datadir"] + "ml_model/ml_dicts/dnn_g_fn.dict", map_location=dft_model.sol["device"]))


def setDNNRep(dft_model, LR, state_dict=None):
    """
    Args:
        dft_model: CDFT model to attach DNNRep surrogate to
        LR: learning rate (unused when not training)
        state_dict: optional pre-loaded state dict; if provided, use instead of loading from disk
    """
    B = dft_model.mesh["BS"]
    Nx = dft_model.mesh["Nx"]
    dnn_rep_fn = DNNRep(input_dim=2, output_dim=1, hidden_dim=24, num_layers=6).double().to(dft_model.sol["device"])
    dft_model.dnn_rep_fn = torch.jit.trace(dnn_rep_fn, torch.zeros([B, 2, Nx], dtype=torch.double, device=dft_model.sol["device"]) )

    # Optimizer for DNN model --------------
    dft_model.optimizer_dnn_rep = torch.optim.Adam(dft_model.dnn_rep_fn.parameters(), 
                                              lr=LR, 
                                              weight_decay=0*1e-5,
                                              )
    
    #Scheduler - ExpDecay
    dft_model.scheduler_dnn_rep = torch.optim.lr_scheduler.ExponentialLR(dft_model.optimizer_dnn_rep, gamma=0.95)        
    LRs = [dft_model.optimizer_dnn_rep.param_groups[0]["lr"],]

    if state_dict is not None:
        dft_model.dnn_rep_fn.load_state_dict(state_dict)
    elif dft_model.sol["RESTART_ML_MODEL"]:
        dft_model.dnn_rep_fn.load_state_dict(torch.load(dft_model.sol["datadir"] + "ml_model/ml_dicts/dnn_rep_fn.dict", map_location=dft_model.sol["device"]))


def setWDA(dft_model, LR,
            in_dim = 1,
            out_dim = 1,
            modes = 100,
            width = 1,
            num_blocks = 1,
            use_bias = True,
            activation = lambda x: x,
            state_dict = None,
            ):
    """
    Args:
        state_dict: optional pre-loaded state dict for wda_fn; if provided, use instead of loading from disk
    """
    dft_model.wda_fn = WDA1d(
        in_dim, out_dim, modes, width, num_blocks, dft_model.mesh, activation, use_bias
        ).double().to(dft_model.sol["device"])

    # Optimizer for FNO model --------------
    dft_model.optimizer_fno = torch.optim.Adam(dft_model.wda_fn.parameters(), 
                                              lr=LR, 
                                              weight_decay=0*1e-5,
                                              )
    
    #Scheduler - ExpDecay
    dft_model.scheduler_fno = torch.optim.lr_scheduler.ExponentialLR(dft_model.optimizer_fno, gamma=0.95)        
    LRs = [dft_model.optimizer_fno.param_groups[0]["lr"],]

    if state_dict is not None:
        dft_model.wda_fn.load_state_dict(state_dict)
    elif dft_model.sol["RESTART_ML_MODEL"]:
        dft_model.wda_fn.load_state_dict(torch.load(dft_model.sol["datadir"] + "ml_model/ml_dicts/wda_fn.dict", map_location=dft_model.sol["device"]))


