#!/usr/bin/env python3
"""
Script 02: Visualize raw vs preprocessed spectra

For each dataset (barley, chickpea, sorghum):
- Load raw CSV
- Load each preprocessed CSV (S, SM, SA, SA0, SAM, SA0M, S0M, 0M)
- For a few samples, plot raw vs preprocessed
- Save figures to:
  outputs/visualizations/preprocessing/{dataset}/{technique}/sample_{i}.png

Also creates an average spectra comparison plot for each dataset.
"""

import os
import sys
import argparse
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.preprocessor import Preprocessor
from src.evaluation.visualizer import (
    plot_raw_vs_preprocessed,
    plot_average_spectra,
    load_spectral_data,
)


def visualize_dataset(
    dataset_name,
    raw_dir="data/raw",
    processed_dir="data/processed",
    output_dir="outputs/visualizations/preprocessing",
    num_samples=5,
    wavelength_start=1,
    wavelength_end=332,
):
    dataset_lower = dataset_name.lower()
    raw_path = os.path.join(raw_dir, f"{dataset_name.capitalize()}.data.csv")

    if not os.path.exists(raw_path):
        print(f"⚠ Raw file not found, skipping: {raw_path}")
        return

    print(f"\n{'='*70}")
    print(f"Visualizing dataset: {dataset_name.upper()}")
    print(f"{'='*70}")

    # Load raw data using Preprocessor to be consistent
    preprocessor = Preprocessor(
        wavelength_start=wavelength_start,
        wavelength_end=wavelength_end,
    )
    X_raw, y_raw = preprocessor.load_dataset(raw_path)

    techniques = ["S", "SM", "SA", "SA0", "SAM", "SA0M", "S0M", "0M"]

    # For average spectra plots
    spectra_for_avg = {}

    for tech in techniques:
        processed_path = os.path.join(
            processed_dir, dataset_lower, f"{dataset_lower}_{tech}.csv"
        )
        if not os.path.exists(processed_path):
            print(f"⚠ Processed file not found for {tech}, skipping: {processed_path}")
            spectra_for_avg[tech] = None
            continue

        X_proc, y_proc = load_spectral_data(processed_path)
        spectra_for_avg[tech] = X_proc

        # Pick some indices (first N samples)
        n = min(num_samples, X_raw.shape[0])
        indices = np.linspace(0, X_raw.shape[0] - 1, n, dtype=int)

        for idx in indices:
            raw_sample = X_raw[idx]
            proc_sample = X_proc[idx]

            save_path = os.path.join(
                output_dir,
                dataset_lower,
                tech,
                f"sample_{idx}_raw_vs_{tech}.png",
            )

            title = f"{dataset_name.capitalize()} - Sample {idx} - Raw vs {tech}"
            plot_raw_vs_preprocessed(
                raw_sample=raw_sample,
                preprocessed_sample=proc_sample,
                wavelengths=np.arange(len(raw_sample)),
                title=title,
                labels=("Raw", f"{tech}"),
                save_path=save_path,
            )

    # Average spectra comparison plot
    avg_save_path = os.path.join(
        "outputs",
        "visualizations",
        "analysis",
        f"{dataset_lower}_average_spectra.png",
    )
    plot_average_spectra(
        spectra_dict=spectra_for_avg,
        wavelengths=np.arange(X_raw.shape[1]),
        title=f"{dataset_name.capitalize()} - Average spectra per preprocessing",
        save_path=avg_save_path,
    )
    print(f"✓ Saved average spectra plot: {avg_save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize raw vs preprocessed spectra for all datasets."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Directory with raw *.data.csv files",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Directory with processed CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visualizations/preprocessing",
        help="Directory to save visualization images",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples per dataset/technique to visualize",
    )
    parser.add_argument(
        "--wavelength-start",
        type=int,
        default=1,
        help="Starting wavelength column index in raw CSV",
    )
    parser.add_argument(
        "--wavelength-end",
        type=int,
        default=332,
        help="Ending wavelength column index in raw CSV (exclusive)",
    )

    args = parser.parse_args()

    datasets = ["barley", "chickpea", "sorghum"]
    for ds in datasets:
        visualize_dataset(
            dataset_name=ds,
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            wavelength_start=args.wavelength_start,
            wavelength_end =args.wavelength_end ,
        )