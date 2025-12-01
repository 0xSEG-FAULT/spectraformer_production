#!/usr/bin/env python3
"""
Script 01: Preprocess all datasets with all preprocessing techniques

This script loads raw spectral data and applies all preprocessing methods:
- S, SM, SA, SA0, SAM, SA0M, S0M, 0M

Outputs: Preprocessed CSV files in data/processed/{dataset}/{dataset}_{technique}.csv
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing.preprocessor import Preprocessor


def main(args):
    """
    Main execution function
    """
    print("\n" + "="*70)
    print("SPECTRAFORMER: Preprocessing All Datasets")
    print("="*70)
    
    # Initialize preprocessor
    preprocessor = Preprocessor(
        wavelength_start=args.wavelength_start,
        wavelength_end=args.wavelength_end
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Preprocess all datasets
    try:
        preprocessor.preprocess_all(args.input_dir, args.output_dir)
        print("\n" + "="*70)
        print("✓ Preprocessing completed successfully!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess spectral data with all techniques"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw CSV files"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save preprocessed data"
    )
    
    parser.add_argument(
        "--wavelength-start",
        type=int,
        default=1,
        help="Starting wavelength column index"
    )
    
    parser.add_argument(
        "--wavelength-end",
        type=int,
        default=332,
        help="Ending wavelength column index"
    )
    
    args = parser.parse_args()
    
    exit_code = main(args)
    sys.exit(exit_code)