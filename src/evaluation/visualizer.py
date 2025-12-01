"""
Visualization utilities for spectral data:
- Raw vs preprocessed spectra
- Average spectra per preprocessing method
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def plot_raw_vs_preprocessed(
    raw_sample,
    preprocessed_sample,
    wavelengths=None,
    title="Raw vs Preprocessed",
    labels=("Raw", "Preprocessed"),
    save_path=None
):
    """
    Plot one sample before and after preprocessing.

    raw_sample: 1D array (wavelengths)
    preprocessed_sample: 1D array (same shape)
    wavelengths: 1D array of wavelength indices or values
    """
    if wavelengths is None:
        wavelengths = np.arange(len(raw_sample))

    plt.figure(figsize=(8, 4))
    plt.plot(wavelengths, raw_sample, label=labels[0], alpha=0.8)
    plt.plot(wavelengths, preprocessed_sample, label=labels[1], alpha=0.8)
    plt.xlabel("Wavelength index")
    plt.ylabel("Intensity")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


def plot_average_spectra(
    spectra_dict,
    wavelengths=None,
    title="Average spectra comparison",
    save_path=None
):
    """
    Plot average spectra for multiple preprocessing methods.

    spectra_dict: dict like {method_name: 2D array (samples x wavelengths)}
    """
    plt.figure(figsize=(8, 4))

    for method, X in spectra_dict.items():
        if X is None or len(X) == 0:
            continue
        avg = X.mean(axis=0)
        if wavelengths is None:
            wavelengths = np.arange(len(avg))
        plt.plot(wavelengths, avg, label=method, alpha=0.8)

    plt.xlabel("Wavelength index")
    plt.ylabel("Intensity")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=200)
        plt.close()
    else:
        plt.show()


def load_spectral_data(csv_path):
    """
    Load spectral data from CSV saved by Preprocessor.save_processed.
    Returns X (features), y (labels).
    """
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y
