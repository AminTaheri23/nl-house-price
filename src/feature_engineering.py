"""Feature engineering module for NL House Price Prediction."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict


class FeatureEngineer:
    """Create and engineer features for the ML model."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.feature_cols = []
        
    def create_price_features(self) -> pd.DataFrame:
        """Create price-related features."""
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING: PRICE FEATURES")
        print("=" * 60)
        
        df = self.df.copy()
        
        df['price_per_sqft'] = df['price'] / df['property-sqft'].replace(0, np.nan)
        
        df['price_log'] = np.log1p(df['price'])
        
        df['price_category'] = pd.cut(
            df['price'],
            bins=[0, 200000, 400000, 600000, 1000000, float('inf')],
            labels=['budget', 'moderate', 'mid_range', 'premium', 'luxury']
        )
        
        self.feature_cols.extend(['price_per_sqft', 'price_log', 'price_category'])
        
        print(f"Created price features: price_per_sqft, price_log, price_category")
        
        return df
    
    def create_property_features(self) -> pd.DataFrame:
        """Create property-related features."""
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING: PROPERTY FEATURES")
        print("=" * 60)
        
        df = self.df.copy()
        
        df['total_rooms'] = df['property-beds'].fillna(0) + df['property-baths'].fillna(0)
        
        df['bath_to_bed_ratio'] = df['property-baths'] / df['property-beds'].replace(0, np.nan)
        
        df['log_sqft'] = np.log1p(df['property-sqft'].clip(lower=1))
        
        df['sqft_per_room'] = df['property-sqft'] / df['total_rooms'].replace(0, np.nan)
        
        self.feature_cols.extend(['total_rooms', 'bath_to_bed_ratio', 'log_sqft', 'sqft_per_room'])
        
        print(f"Created property features: total_rooms, bath_to_bed_ratio, log_sqft, sqft_per_room")
        
        return df
    
    def create_age_features(self) -> pd.DataFrame:
        """Create age-related features."""
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING: AGE FEATURES")
        print("=" * 60)
        
        df = self.df.copy()
        current_year = datetime.now().year
        
        df['house_age'] = current_year - df['Year Built']
        
        df['house_age'] = df['house_age'].replace({np.nan: df['house_age'].median()})
        
        df['age_category'] = pd.cut(
            df['house_age'],
            bins=[-1, 5, 15, 30, 50, float('inf')],
            labels=['new', 'recent', 'established', 'old', 'historic']
        )
        
        df['is_recent'] = (df['house_age'] <= 5).astype(int)
        
        self.feature_cols.extend(['house_age', 'age_category', 'is_recent'])
        
        print(f"Created age features: house_age, age_category, is_recent")
        
        return df
    
    def create_location_features(self) -> pd.DataFrame:
        """Create location-related features."""
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING: LOCATION FEATURES")
        print("=" * 60)
        
        df = self.df.copy()
        
        if 'addressLocality' in df.columns:
            locality_counts = df['addressLocality'].value_counts()
            top_localities = locality_counts.head(20).index.tolist()
            
            df['locality_grouped'] = df['addressLocality'].apply(
                lambda x: x if x in top_localities else 'Other' if pd.notna(x) else 'Unknown'
            )
            
            locality_avg_price = df.groupby('locality_grouped')['price'].mean()
            df['locality_avg_price'] = df['locality_grouped'].map(locality_avg_price)
            
            self.feature_cols.extend(['locality_grouped', 'locality_avg_price'])
            
            print(f"Unique localities: {df['addressLocality'].nunique()}")
            print(f"Top 5 localities: {list(locality_counts.head(5).index)}")
        
        if 'addressRegion' in df.columns:
            print(f"Regions: {df['addressRegion'].unique().tolist()}")
        
        if 'postalCode' in df.columns:
            df['postal_prefix'] = df['postalCode'].str[:3] if df['postalCode'].dtype == 'object' else 'Unknown'
            self.feature_cols.append('postal_prefix')
        
        print(f"Created location features")
        
        return df
    
    def create_property_type_features(self) -> pd.DataFrame:
        """Create property type features."""
        print("\n" + "=" * 60)
        print("FEATURE ENGINEERING: PROPERTY TYPE FEATURES")
        print("=" * 60)
        
        df = self.df.copy()
        
        if 'Property Type' in df.columns:
            type_counts = df['Property Type'].value_counts()
            print(f"Property types distribution:")
            print(type_counts)
            
            type_avg_price = df.groupby('Property Type')['price'].mean()
            df['type_avg_price'] = df['Property Type'].map(type_avg_price)
            
            self.feature_cols.extend(['Property Type', 'type_avg_price'])
        
        return df
    
    def encode_categoricals(self, cat_columns: list) -> pd.DataFrame:
        """Encode categorical columns."""
        print("\n" + "=" * 60)
        print("CATEGORICAL ENCODING")
        print("=" * 60)
        
        df = self.df.copy()
        
        for col in cat_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
                print(f"Encoded {col}: {df[col].nunique()} unique values")
        
        self.feature_cols.extend(cat_columns)
        
        return df
    
    def select_features_for_ml(self, target_col: str = 'price') -> pd.DataFrame:
        """Select features for ML modeling."""
        print("\n" + "=" * 60)
        print("FEATURE SELECTION FOR ML")
        print("=" * 60)
        
        feature_cols = [
            'latitude', 'longitude',
            'property-beds', 'property-baths', 'property-sqft',
            'Year Built', 'house_age', 'total_rooms',
            'log_sqft', 'price_log'
        ]
        
        available_cols = [col for col in feature_cols if col in self.df.columns]
        
        print(f"Selected {len(available_cols)} features for ML:")
        for col in available_cols:
            print(f"  - {col}")
        
        self.feature_cols = available_cols
        self.target_col = target_col
        
        return self.df[available_cols + [target_col]]
    
    def run_pipeline(self) -> pd.DataFrame:
        """Run full feature engineering pipeline."""
        print("\n" + "=" * 60)
        print("STEP 3: FEATURE ENGINEERING PIPELINE")
        print("=" * 60)
        
        self.df = self.create_price_features()
        self.df = self.create_property_features()
        self.df = self.create_age_features()
        self.df = self.create_location_features()
        self.df = self.create_property_type_features()
        
        print(f"\nTotal engineered features: {len(self.feature_cols)}")
        print(f"Dataset shape: {self.df.shape}")
        
        return self.df
    
    def get_feature_summary(self) -> Dict:
        """Get summary of engineered features."""
        summary = {
            'total_features': len(self.feature_cols),
            'feature_list': self.feature_cols,
            'shape': self.df.shape
        }
        
        print("\n" + "=" * 60)
        print("FEATURE SUMMARY")
        print("=" * 60)
        print(f"Total features: {summary['total_features']}")
        print(f"Dataset shape: {summary['shape']}")
        
        return summary


if __name__ == "__main__":
    from data_loader import DataLoader
    from data_cleaner import DataCleaner
    from config import DATA_PATH
    
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
    
    cleaner = DataCleaner(df)
    df = cleaner.clean_basic()
    df = cleaner.clean_numeric_columns(['property-beds', 'property-baths', 'property-sqft', 'Year Built'])
    df = cleaner.remove_price_outliers()
    
    engineer = FeatureEngineer(df)
    df = engineer.run_pipeline()
    engineer.get_feature_summary()
