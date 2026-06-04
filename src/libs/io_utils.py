"""
I/O utilities with robust handling for missing reference data.
Raises DataNotFoundError with helpful messages when data files are not found.
"""
import os
import pickle
import numpy as np


class DataNotFoundError(FileNotFoundError):
    """Raised when required reference/data file is not found."""

    def __init__(self, path: str, description: str = "", hint: str = ""):
        self.path = path
        self.description = description or "Reference data"
        abs_path = os.path.abspath(path)
        msg = f"{self.description} not found: {abs_path}"
        if hint:
            msg += f"\n{hint}"
        super().__init__(msg)


def load_pickle(path: str, description: str = "", hint: str = "") -> object:
    """
    Load object from pickle file. Raises DataNotFoundError if file is missing.

    Args:
        path: Path to the .pkl file.
        description: Short description of the data (e.g. "Phase curve", "z_profiles").
        hint: Optional hint for the user (e.g. "Run compute_bulk_tmd.py first").

    Returns:
        The unpickled object.
    """
    path = os.path.normpath(path)
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError as e:
        raise DataNotFoundError(
            path,
            description=description or "Pickle file",
            hint=hint or "Ensure the data file exists in the data/ folder.",
        ) from e


def load_torch_jit(path: str, description: str = "", hint: str = "", **kwargs) -> object:
    """
    Load TorchScript model via torch.jit.load. Raises DataNotFoundError if file is missing.
    """
    import torch
    path = os.path.normpath(path)
    try:
        return torch.jit.load(path, **kwargs)
    except FileNotFoundError as e:
        raise DataNotFoundError(
            path,
            description=description or "TorchScript model",
            hint=hint or "Ensure the model file exists.",
        ) from e


def load_numpy_txt(path: str, description: str = "", hint: str = "", **kwargs) -> np.ndarray:
    """
    Load array from text file via np.loadtxt. Raises DataNotFoundError if file is missing.

    Args:
        path: Path to the text/csv file.
        description: Short description of the data.
        hint: Optional hint for the user.
        **kwargs: Passed to np.loadtxt (e.g. delimiter, skiprows).

    Returns:
        The loaded numpy array.
    """
    path = os.path.normpath(path)
    try:
        return np.loadtxt(path, **kwargs)
    except FileNotFoundError as e:
        raise DataNotFoundError(
            path,
            description=description or "Data file",
            hint=hint or "Ensure the reference data exists in the data/ folder.",
        ) from e
