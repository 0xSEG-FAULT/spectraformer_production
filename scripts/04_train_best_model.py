#!/usr/bin/env python3
"""
Script 04: Train SpectraFormer with Best Hyperparameters

This script:
1. Loads the best hyperparameters from tuning results
2. Creates and trains the final model
3. Saves the trained model
4. Generates training plots

Output: 
- Model: outputs/models/{dataset}_best.pth
- Plot: outputs/visualizations/training_{dataset}.png
- Metrics: outputs/results/training_metrics_{dataset}.json
"""

import os
import sys
import json
import argparse
import torch
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.config import SpectraFormerConfig
from src.model.spectraformer import SpectraFormer
from src.training.trainer import Trainer
from src.utils.data_loader import load_processed, create_dataloaders, get_device


def train_best_model(
    dataset_name,
    processed_dir="data/processed",
    results_dir="outputs/results",
    model_dir="outputs/models",
    viz_dir="outputs/visualizations",
):
    """
    Train model with best hyperparameters
    """
    print(f"\n{'='*70}")
    print(f"Training Final Model: {dataset_name.upper()}")
    print(f"{'='*70}")
    
    # Load best hyperparameters
    best_params_path = os.path.join(results_dir, f"best_hyperparameters_{dataset_name}.json")
    
    if not os.path.exists(best_params_path):
        print(f"⚠ Best hyperparameters not found: {best_params_path}")
        print(f"  Run: python scripts/03_tune_hyperparameters.py --dataset {dataset_name}")
        return None
    
    with open(best_params_path, 'r') as f:
        best_params = json.load(f)
    
    print(f"Loading best hyperparameters from: {best_params_path}")
    print(f"  Learning Rate: {best_params['learning_rate']}")
    print(f"  Batch Size: {best_params['batch_size']}")
    print(f"  Dropout: {best_params['dropout']}")
    print(f"  Hidden Dim: {best_params['hidden_dim']}")
    
    # Load processed data (SA0M)
    processed_path = os.path.join(
        processed_dir, dataset_name.lower(), f"{dataset_name.lower()}_SA0M.csv"
    )
    
    if not os.path.exists(processed_path):
        print(f"⚠ Processed file not found: {processed_path}")
        return None
    
    print(f"\nLoading data from: {processed_path}")
    X, y = load_processed(processed_path)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X, y,
        batch_size=best_params['batch_size'],
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )
    
    # Create config with best parameters
    # Create config with best parameters
    config = SpectraFormerConfig()
    config.learning_rate = best_params['learning_rate']
    config.batch_size    = best_params['batch_size']
    config.dropout       = best_params['dropout']
    config.hidden_dim    = best_params['hidden_dim']
    config.epochs        = 200  # Full training

    device = get_device()
    config.device = device

     # Set from data to match shapes and labels
    config.input_dim   = X.shape[1]                # e.g., 331
    config.num_classes = len(np.unique(y))         # e.g., 1200

    print("\nModel Configuration:")
    print(f"\nSpectraFormer Config:")
    print(f"  Input Dim: {config.input_dim}")
    print(f"  Num Classes: {config.num_classes}")
    print(f"  Hidden Dim: {config.hidden_dim}")
    print(f"  Num Layers: {config.num_layers}")
    print(f"  Num Heads: {config.num_heads}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Dropout: {config.dropout}")

    # Create and train model
    model = SpectraFormer(config).to(device)
    print(f"Model created with {model.count_parameters():,} parameters")

    trainer = Trainer(model, config, device=device)

    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(model_dir, dataset_name)
    
    print(f"\nTraining for {config.epochs} epochs...")
    history = trainer.train(train_loader, val_loader, epochs=config.epochs, checkpoint_dir=checkpoint_dir)
    
    # Evaluate on test set
    print(f"\n{'='*70}")
    print("Evaluating on Test Set")
    print(f"{'='*70}")
    metrics, preds, labels = trainer.evaluate(test_loader)
    
    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    # Save final model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{dataset_name}_best.pth")
    trainer.save_checkpoint(model_path)
    
    # Save metrics
    metrics_path = os.path.join(results_dir, f"training_metrics_{dataset_name}.json")
    os.makedirs(results_dir, exist_ok=True)
    
    all_metrics = {
        'best_hyperparameters': best_params,
        'test_metrics': metrics,
        'model_parameters': model.count_parameters(),
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n✓ Metrics saved to {metrics_path}")
    
    # Save training plot
    plot_path = os.path.join(viz_dir, f"training_{dataset_name}.png")
    trainer.plot_history(save_path=plot_path)
    
    print(f"\n{'='*70}")
    print(f"✓ Training complete for {dataset_name}!")
    print(f"{'='*70}\n")
    
    return {
        'model_path': model_path,
        'metrics_path': metrics_path,
        'plot_path': plot_path,
        'metrics': metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train SpectraFormer with best hyperparameters"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Directory with preprocessed data",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="outputs/results",
        help="Directory with tuning results",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="outputs/models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--viz-dir",
        type=str,
        default="outputs/visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["barley", "chickpea", "sorghum", "all"],
        help="Which dataset to train (default: all)",
    )
    
    args = parser.parse_args()
    
    datasets = (
        [args.dataset] if args.dataset != "all"
        else ["barley", "chickpea", "sorghum"]
    )
    
    print("\n" + "="*70)
    print("SPECTRAFORMER: Training with Best Hyperparameters")
    print("="*70)
    
    all_results = {}
    
    for dataset in datasets:
        result = train_best_model(
            dataset_name=dataset,
            processed_dir=args.processed_dir,
            results_dir=args.results_dir,
            model_dir=args.model_dir,
            viz_dir=args.viz_dir,
        )
        if result:
            all_results[dataset] = result
    
    # Save combined results
    if all_results:
        results_summary_path = os.path.join(args.results_dir, "training_summary.json")
        summary = {dataset: result['metrics'] for dataset, result in all_results.items()}
        with open(results_summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Training summary saved to {results_summary_path}")
    
    print("\n" + "="*70)
    print("✓ All training complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()