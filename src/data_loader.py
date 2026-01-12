"""Data loading module for NL House Price Prediction."""

import pandas as pd
import numpy as np
from typing import Tuple, Dict


class DataLoader:
    """Load and perform initial exploration of the dataset."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load CSV data with error handling."""
        print("=" * 60)
        print("STEP 1: DATA LOADING")
        print("=" * 60)
        
        self.df = pd.read_csv(
            self.data_path, 
            sep=',', 
            on_bad_lines='skip'
        )
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Total rows: {len(self.df)}")
        print(f"Total columns: {len(self.df.columns)}")
        
        return self.df
    
    def get_basic_info(self) -> Dict:
        """Get basic dataset information."""
        info = {
            'shape': self.df.shape,
            'dtypes': self.df.dtypes.value_counts().to_dict(),
            'numeric_cols': len(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_cols': len(self.df.select_dtypes(include=['object']).columns),
            'missing_cols': len(self.df.columns[self.df.isnull().any()].tolist())
        }
        
        print(f"\nData types summary: {info['dtypes']}")
        print(f"Numeric columns: {info['numeric_cols']}")
        print(f"Categorical columns: {info['categorical_cols']}")
        
        return info
    
    def get_target_stats(self, target_col: str = 'price') -> pd.Series:
        """Get statistics for target variable."""
        if target_col in self.df.columns:
            stats = self.df[target_col].describe()
            print(f"\n--- Target Variable ({target_col}) Statistics ---")
            print(stats)
            
            missing = self.df[target_col].isna().sum()
            zero_count = (self.df[target_col] == 0).sum()
            print(f"Missing values: {missing}")
            print(f"Zero values: {zero_count}")
            
            return stats
        return pd.Series()
    
    def preview_data(self, n: int = 5) -> pd.DataFrame:
        """Preview first n rows."""
        print(f"\n--- First {n} Rows Preview ---")
        print(self.df.head(n))
        return self.df.head(n)


if __name__ == "__main__":
    from config import DATA_PATH
    
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
    loader.get_basic_info()
    loader.get_target_stats()
