"""
Shared plotting utilities: robust directory creation and figure saving.
Use ensure_plot_dir() before saving to any path; use save_figure() for one-liner save.
"""
import os


def ensure_plot_dir(path: str) -> str:
    """
    Ensure the directory for the given path exists. Creates parent directories as needed.
    If path is a directory (no extension or ends with /), creates it directly.
    If path is a file, creates its parent directory.

    Returns the normalized path (with os.path.normpath).
    """
    path = os.path.normpath(path)
    if os.path.splitext(path)[1] and not path.endswith(os.sep):
        # Looks like a file path: ensure parent dir exists
        dirpath = os.path.dirname(path)
    else:
        dirpath = path.rstrip(os.sep)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    return path


def save_figure(path: str, fig=None, **kwargs):
    """
    Save a figure to path, ensuring the directory exists.
    Uses matplotlib.pyplot.gcf() if fig is None.
    """
    import matplotlib.pyplot as plt
    ensure_plot_dir(path)
    (fig if fig is not None else plt.gcf()).savefig(path, **kwargs)


def get_plot_dir(script_dir: str, *subdirs: str) -> str:
    """
    Return a plot directory path relative to script_dir, creating it if needed.
    E.g. get_plot_dir(script_dir, "..", "output", "plot_minOmega") -> .../output/plot_minOmega
    """
    path = os.path.normpath(os.path.join(script_dir, *subdirs))
    os.makedirs(path, exist_ok=True)
    return path
