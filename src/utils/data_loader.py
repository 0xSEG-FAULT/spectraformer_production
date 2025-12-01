"""
Data loader utilities for loading and splitting preprocessed spectral data
"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


class SpectralDataset(Dataset):
    """
    PyTorch Dataset for spectral data
    """
    
    def __init__(self, X, y):
        """
        Parameters:
        -----------
        X : np.ndarray
            Spectral features (samples x wavelengths)
        y : np.ndarray
            Class labels
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_processed(data_path, test_size=0.2, val_size=0.1, random_state=42):
    """
    Load preprocessed data and split into train/val/test sets.
    
    Parameters:
    -----------
    data_path : str
        Path to preprocessed CSV file
    test_size : float
        Proportion of data for testing
    val_size : float
        Proportion of training data for validation
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    dict
        Dictionary containing X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Extract features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Encode labels as integers if necessary
    unique_labels = np.unique(y)
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_mapping[label] for label in y])
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Split train into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
    )
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'n_classes': len(unique_labels)
    }


def create_dataloaders(data_dict, batch_size=16, num_workers=0):
    """
    Create PyTorch DataLoaders from train/val/test splits.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary from load_processed()
    batch_size : int
        Batch size for training
    num_workers : int
        Number of workers for data loading
    
    Returns:
    --------
    dict
        Dictionary containing train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = SpectralDataset(data_dict['X_train'], data_dict['y_train'])
    val_dataset = SpectralDataset(data_dict['X_val'], data_dict['y_val'])
    test_dataset = SpectralDataset(data_dict['X_test'], data_dict['y_test'])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'n_classes': data_dict['n_classes']
    }


def get_device():
    """
    Get GPU device if available, else CPU.
    
    Returns:
    --------
    torch.device
        Device to use for training
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("⚠ GPU not available, using CPU")
    
    return device


def load_all_datasets(processed_data_dir, batch_size=16):
    """
    Load all preprocessed datasets for all preprocessing techniques.
    
    Parameters:
    -----------
    processed_data_dir : str
        Directory containing processed data
    batch_size : int
        Batch size for dataloaders
    
    Returns:
    --------
    dict
        Dictionary with structure: {dataset: {technique: loaders}}
    """
    datasets = ['barley', 'chickpea', 'sorghum']
    techniques = ['S', 'SM', 'SA', 'SA0', 'SAM', 'SA0M', 'S0M', '0M']
    
    all_data = {}
    
    for dataset in datasets:
        all_data[dataset] = {}
        dataset_dir = os.path.join(processed_data_dir, dataset)
        
        if not os.path.exists(dataset_dir):
            print(f"⚠ Dataset directory not found: {dataset_dir}")
            continue
        
        for technique in techniques:
            data_file = os.path.join(dataset_dir, f"{dataset}_{technique}.csv")
            
            if not os.path.exists(data_file):
                print(f"⚠ Data file not found: {data_file}")
                continue
            
            # Load data
            data_dict = load_processed(data_file)
            loaders = create_dataloaders(data_dict, batch_size=batch_size)
            
            all_data[dataset][technique] = loaders
            print(f"✓ Loaded {dataset} with {technique} preprocessing")
    
    return all_data