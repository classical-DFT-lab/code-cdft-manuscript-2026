import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from libs.io_utils import load_pickle, DataNotFoundError
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from libs.utils import (
    linear_interpolate_1d_unsorted,
    normalize_profiles_nt_multiindex,
    profile_nt_index_key,
    sol_dtype,
)
def _to_numpy_1d(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64, copy=False).reshape(-1)
    return np.asarray(x, dtype=np.float64).reshape(-1)

import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from libs.utils import *
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from libs.utils import linear_interpolation


class MY_DATASET_FROM_DF(Dataset):
    def __init__(self, dataframe, dataframe_0, target_grid, device, transform=True, dtype=torch.float64):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with MultiIndex or tuple column ('N', 'T') and columns x, rho, rho_fit
            target_grid (Tensor): Grid to interpolate onto
            transform (bool): If True, interpolate onto target grid
            dtype: torch dtype for tensors (should match sol["dtype"] / TORCH_DTYPE in training script)
        """
        self.df = dataframe.reset_index(drop=False)
        self.df_md = dataframe
        self.df_0 = dataframe_0
        self.target_grid = target_grid.to(device)
        self.transform = transform
        self.device = device
        self.dtype = dtype

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        N = row['N']
        T = row['T']
        x = torch.as_tensor(row['x'], dtype=self.dtype, device=self.device)  # (Nx)
        rho = torch.as_tensor(row['rho'], dtype=self.dtype, device=self.device)  # (Nx)
        rho_fit = torch.as_tensor(row['rho_fit'], dtype=self.dtype, device=self.device)  # (Nx)
        rho_0 = torch.as_tensor(self.df_0.loc[(N, T)]["rho"], dtype=self.dtype, device=self.device)  # (Nx)
        x_0 = torch.as_tensor(self.df_0.loc[(N, T)]["x"], dtype=self.dtype, device=self.device)  # (Nx)

        N = torch.tensor(N, dtype=self.dtype, device=self.device)[None]
        beta = torch.tensor(1.0 / T, dtype=self.dtype, device=self.device)[None]
        T = torch.tensor(T, dtype=self.dtype, device=self.device)[None]

        if self.transform:
            rho = self.interpolate_rho(x, rho, self.target_grid)  # (1, Nx)
            rho_0 = self.interpolate_rho(x_0, rho_0, self.target_grid)  # (1, Nx)
            rho_fit = self.interpolate_rho(x, rho_fit, self.target_grid)  # (1, Nx)

        return {
            'x': self.target_grid if self.transform else x,  # (Nx)
            'rho': rho,  # (1, Nx)
            'rho_0': rho_0,  # (1, Nx)
            'rho_fit': rho_fit,  # (1, Nx)
            'N': N, # (1,)
            'beta': beta, # (1,)
            'T': T, # (1,)
        }

    def interpolate_rho(self, x_src, rho_src, x_target):
        rho_interp = linear_interpolation(x_target, (x_src - x_src[0] + (x_src[1] - x_src[0])/2) + x_target[0], rho_src)[None,...]
        return rho_interp 

def my_collate_fn_old(batch):
    return batch  # Now each element is a single sample dict

def my_collate_fn(batch):
    collated = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        if key == 'x':
            # Assume all x are the same → take just the first one
            collated[key] = vals[0]
        elif isinstance(vals[0], torch.Tensor):
            collated[key] = torch.stack(vals, dim=0)
        else:
            collated[key] = vals
    return collated


def setDatasetObject(good_N, good_T, datadir, mesh, sol, batch_size, transform, batches_in_list):
    
    with open(datadir + "z_profiles.pkl", "rb") as f:
        df = pickle.load(f)
        mask = ( # with a ~( is the negative mask
            df.index.get_level_values("N").isin(good_N) &
            df.index.get_level_values("T").isin(good_T)
        )
        df = df[mask]  

    print(f"Loaded data with {len(df)} samples.")

    with open(datadir + "z_profiles_0.pkl", "rb") as f:
        df_0 = pickle.load(f)
        mask = ( # with a ~( is the negative mask
            df_0.index.get_level_values("N").isin(good_N) &
            df_0.index.get_level_values("T").isin(good_T)
        )
        df_0 = df_0[mask]  

    print(f"Loaded data with {len(df_0)} samples.")

    target_grid = mesh["x"]
    td = sol_dtype(sol)
    dataset = MY_DATASET_FROM_DF(
        df, df_0, target_grid, device=sol["device"], transform=transform, dtype=td
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, #pin_memory=True,
                            collate_fn=(my_collate_fn_old if batches_in_list else my_collate_fn))

    # Optional plot for verification
    for batch in dataloader:
        if batches_in_list:
            b = batch[0]
            plt.plot(b['x'].cpu(), b['rho'][0].cpu(), label=f"rho_md | T={1/b['beta'].cpu().item():.2f}, N={b['N'].cpu().item():.0f}")
            plt.plot(b['x'].cpu(), b['rho_0'][0].cpu(), label=f"rho_0 | T={1/b['beta'].cpu().item():.2f}, N={b['N'].cpu().item():.0f}")
        else:
            b = batch
            plt.plot(b['x'].cpu(), b['rho'][0, 0].cpu(), label=f"rho_md | T={1/b['beta'][0].cpu().item():.2f}, N={b['N'][0].cpu().item():.0f}")
            plt.plot(b['x'].cpu(), b['rho_0'][0, 0].cpu(), label=f"rho_0 | T={1/b['beta'][0].cpu().item():.2f}, N={b['N'][0].cpu().item():.0f}")
        break  # just one batch

    plt.legend()
    plt.savefig(sol["outdir"] + "plot_train/rho_df_data.png")
    plt.close()

    return dataset, dataloader
