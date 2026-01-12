"""Data cleaning module with outlier detection and removal."""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from typing import Tuple, Dict, Optional


class DataCleaner:
    """Clean data and remove outliers using multiple methods."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_shape = df.shape
        self.outlier_info = {}
        
    def clean_basic(self) -> pd.DataFrame:
        """Basic cleaning: remove null/zero prices."""
        print("\n" + "=" * 60)
        print("STEP 2: BASIC DATA CLEANING")
        print("=" * 60)
        
        df_clean = self.df.copy()
        initial_count = len(df_clean)
        
        if 'price' in df_clean.columns:
            df_clean = df_clean[df_clean['price'].notna()]
            df_clean = df_clean[df_clean['price'] > 0]
            
        print(f"Original rows: {initial_count}")
        print(f"After removing null/zero prices: {len(df_clean)}")
        print(f"Rows removed: {initial_count - len(df_clean)}")
        
        self.df = df_clean
        return df_clean
    
    def remove_outliers_iqr(self, columns: list, multiplier: float = 1.5) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        print("\n" + "=" * 60)
        print("OUTLIER REMOVAL: IQR METHOD")
        print("=" * 60)
        
        df = self.df.copy()
        outlier_mask = pd.Series([False] * len(df))
        
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_mask = outlier_mask | col_outliers
                
                outlier_count = col_outliers.sum()
                print(f"{col}: {outlier_count} outliers removed")
                
                self.outlier_info[col] = {
                    'method': 'IQR',
                    'bounds': (lower_bound, upper_bound),
                    'count': outlier_count
                }
        
        df_clean = df[~outlier_mask]
        print(f"\nTotal rows after IQR cleaning: {len(df_clean)}")
        print(f"Total outliers removed: {outlier_mask.sum()}")
        
        self.df = df_clean
        return df_clean
    
    def remove_outliers_zscore(self, columns: list, threshold: float = 3.0) -> pd.DataFrame:
        """Remove outliers using Z-score method."""
        print("\n" + "=" * 60)
        print("OUTLIER REMOVAL: Z-SCORE METHOD")
        print("=" * 60)
        
        df = self.df.copy()
        outlier_mask = pd.Series([False] * len(df))
        
        for col in columns:
            if col in df.columns:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                
                if len(z_scores) > 0:
                    col_outliers = np.abs(stats.zscore(df[col])) > threshold
                    outlier_mask = outlier_mask | col_outliers
                    
                    outlier_count = col_outliers.sum()
                    print(f"{col}: {outlier_count} outliers removed")
                    
                    self.outlier_info[col] = {
                        'method': 'Z-score',
                        'threshold': threshold,
                        'count': outlier_count
                    }
        
        df_clean = df[~outlier_mask]
        print(f"\nTotal rows after Z-score cleaning: {len(df_clean)}")
        print(f"Total outliers removed: {outlier_mask.sum()}")
        
        self.df = df_clean
        return df_clean
    
    def remove_outliers_isolation_forest(self, columns: list, contamination: float = 0.1) -> pd.DataFrame:
        """Remove outliers using Isolation Forest."""
        print("\n" + "=" * 60)
        print("OUTLIER REMOVAL: ISOLATION FOREST")
        print("=" * 60)
        
        df = self.df.copy()
        
        available_cols = [col for col in columns if col in df.columns]
        
        if not available_cols:
            print("No valid columns found for Isolation Forest")
            return df
        
        X = df[available_cols].copy()
        X = X.fillna(X.median())
        
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        
        outliers = iso_forest.fit_predict(X)
        outlier_mask = outliers == -1
        
        outlier_count = outlier_mask.sum()
        print(f"Columns used: {available_cols}")
        print(f"Outliers detected: {outlier_count}")
        
        self.outlier_info['isolation_forest'] = {
            'columns': available_cols,
            'contamination': contamination,
            'count': outlier_count
        }
        
        df_clean = df[~outlier_mask]
        print(f"Total rows after Isolation Forest: {len(df_clean)}")
        
        self.df = df_clean
        return df_clean
    
    def remove_price_outliers(self, lower_percentile: float = 1, upper_percentile: float = 99) -> pd.DataFrame:
        """Remove price outliers using percentiles."""
        print("\n" + "=" * 60)
        print("PRICE OUTLIER REMOVAL")
        print("=" * 60)
        
        df = self.df.copy()
        
        lower_bound = df['price'].quantile(lower_percentile / 100)
        upper_bound = df['price'].quantile(upper_percentile / 100)
        
        print(f"Price range: ${lower_bound:,.2f} - ${upper_bound:,.2f}")
        
        before = len(df)
        df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
        
        print(f"Rows before: {before}")
        print(f"Rows after: {len(df)}")
        print(f"Removed: {before - len(df)}")
        
        self.df = df
        return df
    
    def clean_numeric_columns(self, columns: list) -> pd.DataFrame:
        """Clean and convert numeric columns."""
        print("\n" + "=" * 60)
        print("NUMERIC COLUMN CLEANING")
        print("=" * 60)
        
        df = self.df.copy()
        
        for col in columns:
            if col in df.columns:
                if col == 'property-sqft':
                    df[col] = df[col].astype(str).str.replace(',', '').replace('N/A', np.nan)
                
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                missing_before = self.df[col].isna().sum()
                missing_after = df[col].isna().sum()
                print(f"{col}: {missing_before} -> {missing_after} missing values")
        
        self.df = df
        return df
    
    def get_cleaning_report(self) -> Dict:
        """Generate cleaning report."""
        report = {
            'original_shape': self.original_shape,
            'final_shape': self.df.shape,
            'rows_removed': self.original_shape[0] - self.df.shape[0],
            'outlier_info': self.outlier_info
        }
        
        print("\n" + "=" * 60)
        print("CLEANING REPORT")
        print("=" * 60)
        print(f"Original rows: {report['original_shape'][0]}")
        print(f"Final rows: {report['final_shape'][0]}")
        print(f"Total rows removed: {report['rows_removed']}")
        print(f"Removal percentage: {report['rows_removed']/report['original_shape'][0]*100:.2f}%")
        
        return report


if __name__ == "__main__":
    from data_loader import DataLoader
    from config import DATA_PATH
    
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
    
    cleaner = DataCleaner(df)
    cleaner.clean_basic()
    cleaner.clean_numeric_columns(['property-beds', 'property-baths', 'property-sqft', 'Year Built'])
    cleaner.remove_price_outliers()
    cleaner.remove_outliers_iqr(['price', 'property-sqft'])
    cleaner.get_cleaning_report()
