"""
Main preprocessor class that orchestrates all preprocessing techniques
"""

import numpy as np
import pandas as pd
import os
from .techniques import (
    savgol_smoothing,
    airpls_baseline,
    remove_negatives,
    minmax_normalize
)


class Preprocessor:
    """
    Handles all preprocessing operations for spectral data.
    
    Techniques:
    - S: Savitzky-Golay smoothing
    - A: airPLS baseline correction
    - 0: Remove negatives
    - M: Min-Max normalization
    
    Combined techniques:
    - SM, SA, SA0, SAM, SA0M, S0M, 0M
    """
    
    VALID_TECHNIQUES = ["S", "SM", "SA", "SA0", "SAM", "SA0M", "S0M", "0M"]
    DATASETS = ["barley", "chickpea", "sorghum"]
    
    def __init__(self, wavelength_start=1, wavelength_end=332):
        """
        Initialize preprocessor.
        
        Parameters:
        -----------
        wavelength_start : int
            Starting wavelength column index
        wavelength_end : int
            Ending wavelength column index
        """
        self.wavelength_start = wavelength_start
        self.wavelength_end = wavelength_end
    
    def load_dataset(self, dataset_path):
        """
        Load dataset from CSV file.
        
        Parameters:
        -----------
        dataset_path : str
            Path to CSV file
        
        Returns:
        --------
        tuple
            (X: spectral data, y: labels)
        """
        df = pd.read_csv(dataset_path)
        
        # Extract spectral features (wavelengths)
        X = df.iloc[:, self.wavelength_start:self.wavelength_end].values
        
        # Extract labels (last column)
        y = df.iloc[:, -1].values
        
        return X, y
    
    def apply_technique(self, X, technique):
        """
        Apply preprocessing technique(s) to spectral data.
        
        Parameters:
        -----------
        X : np.ndarray
            Input spectral data (samples x wavelengths)
        technique : str
            Technique code: S, SM, SA, SA0, SAM, SA0M, S0M, 0M
        
        Returns:
        --------
        np.ndarray
            Preprocessed spectral data
        """
        if technique not in self.VALID_TECHNIQUES:
            raise ValueError(f"Invalid technique: {technique}. Must be one of {self.VALID_TECHNIQUES}")
        
        result = X.copy()
        
        # Apply operations in order: S, A, 0, M
        if 'S' in technique:
            result = savgol_smoothing(result)
        
        if 'A' in technique:
            result = airpls_baseline(result)
        
        if '0' in technique:
            result = remove_negatives(result)
        
        if 'M' in technique:
            result = minmax_normalize(result)
        
        return result
    
    def save_processed(self, X, y, output_path):
        """
        Save preprocessed data to CSV.
        
        Parameters:
        -----------
        X : np.ndarray
            Preprocessed spectral data
        y : np.ndarray
            Labels
        output_path : str
            Path to save CSV file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Combine X and y
        data = np.hstack([X, y.reshape(-1, 1)])
        
        # Create DataFrame
        columns = [f"wavelength_{i}" for i in range(X.shape[1])] + ["label"]
        df = pd.DataFrame(data, columns=columns)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"✓ Saved: {output_path}")
    
    def preprocess_all(self, raw_data_dir, output_dir):
        """
        Preprocess all datasets with all techniques.
        
        Parameters:
        -----------
        raw_data_dir : str
            Directory containing raw CSV files
        output_dir : str
            Directory to save processed data
        """
        for dataset in self.DATASETS:
            # Load raw data
            dataset_path = os.path.join(raw_data_dir, f"{dataset.capitalize()}.data.csv")
            if not os.path.exists(dataset_path):
                print(f"⚠ Skipping {dataset}: file not found")
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing: {dataset.upper()}")
            print(f"{'='*60}")
            
            X, y = self.load_dataset(dataset_path)
            print(f"Loaded {X.shape[0]} samples with {X.shape[1]} wavelengths")
            
            # Apply each technique
            for technique in self.VALID_TECHNIQUES:
                print(f"  Applying {technique}...", end=" ")
                X_processed = self.apply_technique(X, technique)
                
                # Save processed data
                output_path = os.path.join(output_dir, dataset, f"{dataset}_{technique}.csv")
                self.save_processed(X_processed, y, output_path)
                print("✓")
        
        print(f"\n{'='*60}")
        print("All datasets processed successfully!")
        print(f"{'='*60}")
