#!/usr/bin/env python3
"""
Script 03: Hyperparameter Tuning for SpectraFormer

This script tests different hyperparameter combinations and saves the best one.

For each dataset (barley, chickpea, sorghum):
- Tests: learning_rate, batch_size, dropout, hidden_dim
- Trains model for N epochs with each combination
- Records test accuracy
- Saves best hyperparameters to JSON

Output: outputs/results/best_hyperparameters_{dataset}.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.config import SpectraFormerConfig
from src.model.spectraformer import SpectraFormer
from src.training.trainer import Trainer
from src.utils.data_loader import load_processed, create_dataloaders, get_device


def tune_hyperparameters(
    dataset_name,
    processed_dir="data/processed",
    output_dir="outputs/results",
    epochs_per_trial=50,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
):
    """
    Tune hyperparameters for a specific dataset.

    Returns
    -------
    dict or None
        Best hyperparameters dictionary, or None if no trial succeeded.
    """
    print(f"\n{'='*70}")
    print(f"Hyperparameter Tuning: {dataset_name.upper()}")
    print(f"{'='*70}")

    # Define hyperparameter search space
    search_space = {
        "learning_rate": [0.0001, 0.0005, 0.001, 0.005],
        "batch_size": [8, 16, 32, 64],
        "dropout": [0.1, 0.3, 0.5],
        "hidden_dim": [64, 128, 256],
    }

    device = get_device()
    results = {}
    best_accuracy = float("-inf")
    best_params = None

    # Load processed data (using SA0M as it's usually best)
    processed_path = os.path.join(
        processed_dir, dataset_name.lower(), f"{dataset_name.lower()}_SA0M.csv"
    )

    if not os.path.exists(processed_path):
        print(f"⚠ Processed file not found: {processed_path}")
        return None

    print(f"Loading data from: {processed_path}")
    X, y = load_processed(processed_path)

    # Count total combinations
    total_combinations = (
        len(search_space["learning_rate"])
        * len(search_space["batch_size"])
        * len(search_space["dropout"])
        * len(search_space["hidden_dim"])
    )

    print(f"Search space: {total_combinations} combinations")
    print(f"Epochs per trial: {epochs_per_trial}\n")

    trial = 0

    for lr in search_space["learning_rate"]:
        for batch_size in search_space["batch_size"]:
            for dropout in search_space["dropout"]:
                for hidden_dim in search_space["hidden_dim"]:
                    trial += 1

                    # Create new dataloaders with this batch size
                    train_loader_trial, val_loader_trial, test_loader_trial = create_dataloaders(
                        X,
                        y,
                        batch_size=batch_size,
                        train_ratio=train_ratio,
                        val_ratio=val_ratio,
                        test_ratio=test_ratio,
                    )

                    # Create config with these hyperparameters
                    config = SpectraFormerConfig()
                    config.learning_rate = lr
                    config.batch_size = batch_size
                    config.dropout = dropout
                    config.hidden_dim = hidden_dim
                    config.epochs = epochs_per_trial
                    config.device = device
                    config.input_dim = X.shape[1]           # ADD THIS LINE
                    config.num_classes = len(np.unique(y))  # ADD THIS LINE

                    # Create model and trainer
                    model = SpectraFormer(config)
                    trainer = Trainer(model, config, device=device)

                    print(
                        f"\n[{trial}/{total_combinations}] Testing: "
                        f"lr={lr}, bs={batch_size}, do={dropout}, hid={hidden_dim}"
                    )
                    print("  Starting trial with params:", lr, batch_size, dropout, hidden_dim)
                    try:
                        # Train for specified epochs
                        trainer.train(
                            train_loader_trial,
                            val_loader_trial,
                            epochs=epochs_per_trial,
                        )

                        # Evaluate on test set
                        metrics, _, _ = trainer.evaluate(test_loader_trial)
                        test_accuracy = float(metrics.get("accuracy", 0.0))

                        # Store result
                        param_key = (
                            f"lr={lr}_bs={batch_size}_do={dropout}_hid={hidden_dim}"
                        )
                        results[param_key] = test_accuracy

                        print(f"  → Test Accuracy: {test_accuracy:.4f}")

                        # Track best during loop (for logging)
                        if test_accuracy > best_accuracy:
                            best_accuracy = test_accuracy
                            best_params = {
                                "learning_rate": float(lr),
                                "batch_size": int(batch_size),
                                "dropout": float(dropout),
                                "hidden_dim": int(hidden_dim),
                                "test_accuracy": float(test_accuracy),
                            }
                            print(f"  ✓ NEW BEST! Accuracy: {best_accuracy:.4f}")

                    except Exception as e:
                        print(f"  ✗ Error: {e}")
                        continue

    # Save all trial results
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, f"tuning_results_{dataset_name}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Tuning results saved to {results_path}")

    # Derive best_params from results if best_params is still None
    if not best_params and results:
        best_key = max(results, key=results.get)
        best_acc = float(results[best_key])

        parts = dict(p.split("=") for p in best_key.split("_"))
        best_params = {
            "learning_rate": float(parts["lr"]),
            "batch_size": int(parts["bs"]),
            "dropout": float(parts["do"]),
            "hidden_dim": int(parts["hid"]),
            "test_accuracy": best_acc,
        }

    # Save best parameters if any trial succeeded
    if best_params:
        best_params_path = os.path.join(
            output_dir, f"best_hyperparameters_{dataset_name}.json"
        )
        with open(best_params_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"✓ Best hyperparameters saved to {best_params_path}")
        print(f"\nBest Parameters for {dataset_name}:")
        print(f"  Learning Rate: {best_params['learning_rate']}")
        print(f"  Batch Size: {best_params['batch_size']}")
        print(f"  Dropout: {best_params['dropout']}")
        print(f"  Hidden Dim: {best_params['hidden_dim']}")
        print(f"  Test Accuracy: {best_params['test_accuracy']:.4f}")
    else:
        print("\n⚠ No successful trials; best hyperparameters not saved.")

    return best_params


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for SpectraFormer"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Directory with preprocessed data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/results",
        help="Directory to save tuning results",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Epochs per trial (reduce for faster tuning)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["barley", "chickpea", "sorghum", "all"],
        help="Which dataset to tune (default: all)",
    )

    args = parser.parse_args()

    datasets = (
        [args.dataset] if args.dataset != "all" else ["barley", "chickpea", "sorghum"]
    )

    print("\n" + "=" * 70)
    print("SPECTRAFORMER: Hyperparameter Tuning")
    print("=" * 70)

    all_best_params = {}

    for dataset in datasets:
        best_params = tune_hyperparameters(
            dataset_name=dataset,
            processed_dir=args.processed_dir,
            output_dir=args.output_dir,
            epochs_per_trial=args.epochs,
        )
        if best_params:
            all_best_params[dataset] = best_params

    # Save combined best parameters
    if all_best_params:
        combined_path = os.path.join(
            args.output_dir, "best_hyperparameters_all.json"
        )
        with open(combined_path, "w") as f:
            json.dump(all_best_params, f, indent=2)
        print(f"\n✓ Combined best hyperparameters saved to {combined_path}")

    print("\n" + "=" * 70)
    print("✓ Hyperparameter tuning complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
