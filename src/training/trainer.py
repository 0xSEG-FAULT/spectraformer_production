"""
Trainer class for SpectraFormer model
Handles training, validation, and evaluation
"""

import os
import json
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


class Trainer:
    """Trainer for SpectraFormer model"""
    
    def __init__(self, model, config, device="cuda"):
        self.model = model
        self.config = config
        self.device = device
        self.model = self.model.to(device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc="Training")
        for X, y in pbar:
            X, y = X.to(self.device), y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(X)
            loss = self.criterion(logits, y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = accuracy_score(all_labels, all_preds)
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X, y in tqdm(val_loader, desc="Validating"):
                X, y = X.to(self.device), y.to(self.device)
                
                logits = self.model(X)
                loss = self.criterion(logits, y)
                
                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = accuracy_score(all_labels, all_preds)
        
        return avg_loss, avg_acc
    
    def evaluate(self, test_loader):
        """Evaluate on test set and return metrics"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X, y in tqdm(test_loader, desc="Evaluating"):
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        metrics = {
            'accuracy': float(accuracy_score(all_labels, all_preds)),
            'precision': float(precision_score(all_labels, all_preds, average='weighted')),
            'recall': float(recall_score(all_labels, all_preds, average='weighted')),
            'f1': float(f1_score(all_labels, all_preds, average='weighted')),
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        }
        
        return metrics, all_preds, all_labels
    
    def train(self, train_loader, val_loader, epochs=None, checkpoint_dir=None):
        """Full training loop"""
        if epochs is None:
            epochs = self.config.epochs
        
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"{'='*70}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                print("✓ New best model!")
                
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(best_model_state, os.path.join(checkpoint_dir, 'best_model.pth'))
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return self.history
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"✓ Model saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"✓ Model loaded from {path}")
    
    def plot_history(self, save_path=None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(self.history['train_loss'], label='Train')
        ax1.plot(self.history['val_loss'], label='Val')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training vs Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(self.history['train_acc'], label='Train')
        ax2.plot(self.history['val_acc'], label='Val')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training vs Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=200)
            print(f"✓ Plot saved to {save_path}")
        else:
            plt.show()