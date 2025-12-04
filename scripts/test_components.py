#!/usr/bin/env python3
"""
Component verification script - test each module independently
"""
import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

print("="*70)
print("COMPONENT VERIFICATION")
print("="*70)

# ============================================================
# 1. Test data_loader.py
# ============================================================
print("\n[1/4] Testing data_loader...")
from src.utils.data_loader import load_processed, create_dataloaders, get_device

# Test get_device
device = get_device()
print(f"  ✓ get_device() -> {device}")

# Test load_processed
data_path = "data/processed/barley/barley_SA0M.csv"
X, y = load_processed(data_path)
print(f"  ✓ load_processed() -> X shape: {X.shape}, y shape: {y.shape}")
print(f"    - X dtype: {X.dtype}, y dtype: {y.dtype}")
print(f"    - Unique classes: {len(np.unique(y))}")
print(f"    - Class distribution sample: {np.bincount(y)[:10]}...")

# Test create_dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    X, y, batch_size=16, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
)
print(f"  ✓ create_dataloaders() -> 3 loaders created")
print(f"    - Train batches: {len(train_loader)}")
print(f"    - Val batches: {len(val_loader)}")
print(f"    - Test batches: {len(test_loader)}")

# Check one batch
for xb, yb in train_loader:
    print(f"    - Sample batch: X={xb.shape}, y={yb.shape}, X dtype={xb.dtype}, y dtype={yb.dtype}")
    print(f"    - X device: {xb.device}, y device: {yb.device}")
    break

# ============================================================
# 2. Test config.py
# ============================================================
print("\n[2/4] Testing config...")
from src.model.config import SpectraFormerConfig

config = SpectraFormerConfig()
print(f"  ✓ SpectraFormerConfig created")
print(f"    - input_dim: {config.input_dim}")
print(f"    - num_classes: {config.num_classes}")
print(f"    - hidden_dim: {config.hidden_dim}")
print(f"    - num_heads: {config.num_heads}")
print(f"    - num_layers: {config.num_layers}")
print(f"    - dropout: {config.dropout}")
print(f"    - learning_rate: {config.learning_rate}")
print(f"    - batch_size: {config.batch_size}")
print(f"    - epochs: {config.epochs}")
print(f"    - device: {config.device}")

# ============================================================
# 3. Test spectraformer.py (model)
# ============================================================
print("\n[3/4] Testing SpectraFormer model...")
from src.model.spectraformer import SpectraFormer

# Update config with actual dimensions from data
config.input_dim = X.shape[1]
config.num_classes = len(np.unique(y))
print(f"  ✓ Updated config: input_dim={config.input_dim}, num_classes={config.num_classes}")

model = SpectraFormer(config)
model = model.to(device)
print(f"  ✓ SpectraFormer model created and moved to {device}")
print(f"    - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"    - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Test forward pass
model.eval()
with torch.no_grad():
    xb_sample = xb[:4].to(device)  # Take 4 samples from earlier batch
    output = model(xb_sample)
    print(f"  ✓ Forward pass: input {xb_sample.shape} -> output {output.shape}")
    print(f"    - Output dtype: {output.dtype}, device: {output.device}")
    print(f"    - Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

# ============================================================
# 4. Test trainer.py
# ============================================================
print("\n[4/4] Testing Trainer...")
from src.training.trainer import Trainer

trainer = Trainer(model, config, device=device)
print(f"  ✓ Trainer created")
print(f"    - Optimizer: {type(trainer.optimizer).__name__}")
print(f"    - Scheduler: {type(trainer.scheduler).__name__ if hasattr(trainer, 'scheduler') and trainer.scheduler else 'None'}")
print(f"    - Loss function: {type(trainer.criterion).__name__}")

# Test one training epoch
print(f"\n  Testing 1 training epoch (this may take a minute)...")
model.train()
try:
    # Create small subset loaders for quick test
    small_train, small_val, _ = create_dataloaders(
        X[:100], y[:100], batch_size=16, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0
    )
    
    trainer.train(small_train, small_val, epochs=1)
    print(f"  ✓ Training completed 1 epoch without errors")
    
    # Test evaluation
    metrics, preds, labels = trainer.evaluate(small_val)
    print(f"  ✓ Evaluation completed")
    print(f"    - Metrics keys: {list(metrics.keys())}")
    print(f"    - Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"    - Predictions shape: {preds.shape if hasattr(preds, 'shape') else len(preds)}")
    print(f"    - Labels shape: {labels.shape if hasattr(labels, 'shape') else len(labels)}")
    
except Exception as e:
    print(f"  ✗ Error during training/evaluation:")
    import traceback
    traceback.print_exc()

# ============================================================
print("\n" + "="*70)
print("✓ COMPONENT VERIFICATION COMPLETE")
print("="*70)
