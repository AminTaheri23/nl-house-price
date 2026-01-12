"""Visualization module for EDA and model results."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
import os


class Visualizer:
    """Create visualizations for data analysis and model results."""
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
        
    def plot_price_distribution(self, df: pd.DataFrame, price_col: str = 'price'):
        """Plot price distribution."""
        print("\n" + "=" * 60)
        print("VISUALIZATION: PRICE DISTRIBUTION")
        print("=" * 60)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        prices = df[price_col].dropna() / 1000
        
        axes[0].hist(prices, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_xlabel('Price (CAD, thousands)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Price Distribution')
        
        axes[1].hist(np.log1p(prices), bins=50, edgecolor='black', alpha=0.7, color='coral')
        axes[1].set_xlabel('Log Price')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Log Price Distribution')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/price_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir}/price_distribution.png")
        
    def plot_price_vs_sqft(self, df: pd.DataFrame):
        """Plot price vs square footage."""
        print("\n" + "=" * 60)
        print("VISUALIZATION: PRICE VS SQUARE FOOTAGE")
        print("=" * 60)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        sqft = df['property-sqft'].fillna(0)
        price = df['price'].fillna(0) / 1000
        
        axes[0].scatter(sqft, price, alpha=0.3, s=10, c='steelblue')
        axes[0].set_xlabel('Square Footage')
        axes[0].set_ylabel('Price (CAD, thousands)')
        axes[0].set_title('Price vs Square Footage')
        
        axes[1].scatter(np.log1p(sqft.clip(lower=1)), np.log1p(price), alpha=0.3, s=10, c='coral')
        axes[1].set_xlabel('Log Square Footage')
        axes[1].set_ylabel('Log Price')
        axes[1].set_title('Log Price vs Log Square Footage')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/price_vs_sqft.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir}/price_vs_sqft.png")
        
    def plot_locality_prices(self, df: pd.DataFrame, n: int = 15):
        """Plot average prices by locality."""
        print("\n" + "=" * 60)
        print("VISUALIZATION: PRICES BY LOCALITY")
        print("=" * 60)
        
        locality_avg = df.groupby('addressLocality')['price'].mean().sort_values(ascending=False).head(n)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(locality_avg)), locality_avg.values / 1000, color='steelblue')
        plt.yticks(range(len(locality_avg)), locality_avg.index, fontsize=9)
        plt.xlabel('Average Price (CAD, thousands)')
        plt.title(f'Top {n} Localities by Average Price')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/locality_prices.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir}/locality_prices.png")
        
    def plot_bedroom_prices(self, df: pd.DataFrame):
        """Plot average prices by number of bedrooms."""
        print("\n" + "=" * 60)
        print("VISUALIZATION: PRICES BY BEDROOMS")
        print("=" * 60)
        
        beds_price = df.groupby('property-beds')['price'].mean()
        beds_price = beds_price[beds_price.index.notna()]
        beds_price = beds_price[beds_price.index <= 10]
        
        plt.figure(figsize=(10, 6))
        plt.bar(beds_price.index.astype(str), beds_price.values / 1000, color='coral', edgecolor='black')
        plt.xlabel('Number of Bedrooms')
        plt.ylabel('Average Price (CAD, thousands)')
        plt.title('Average Price by Number of Bedrooms')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/bedroom_prices.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir}/bedroom_prices.png")
        
    def plot_property_type_distribution(self, df: pd.DataFrame):
        """Plot property type distribution."""
        print("\n" + "=" * 60)
        print("VISUALIZATION: PROPERTY TYPE DISTRIBUTION")
        print("=" * 60)
        
        if 'Property Type' in df.columns:
            type_counts = df['Property Type'].value_counts()
            
            plt.figure(figsize=(10, 6))
            plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', 
                   colors=plt.cm.Set3.colors)
            plt.title('Property Type Distribution')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/property_type_distribution.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {self.output_dir}/property_type_distribution.png")
        
    def plot_feature_importance(self, importance_dict: Dict, feature_names: List[str]):
        """Plot feature importance for tree-based models."""
        print("\n" + "=" * 60)
        print("VISUALIZATION: FEATURE IMPORTANCE")
        print("=" * 60)
        
        plt.figure(figsize=(12, 8))
        
        if 'Random Forest' in importance_dict:
            importance = importance_dict['Random Forest']
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            features, importances = zip(*sorted_imp)
            
            plt.barh(range(len(features)), importances, align='center', color='steelblue')
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title('Random Forest Feature Importance')
            plt.gca().invert_yaxis()
            
        elif 'LightGBM' in importance_dict:
            importance = importance_dict['LightGBM']
            sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            features, importances = zip(*sorted_imp)
            
            plt.barh(range(len(features)), importances, align='center', color='coral')
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title('LightGBM Feature Importance')
            plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir}/feature_importance.png")
        
    def plot_actual_vs_predicted(self, y_test: pd.Series, y_pred: np.ndarray, model_name: str):
        """Plot actual vs predicted prices."""
        print("\n" + "=" * 60)
        print(f"VISUALIZATION: ACTUAL VS PREDICTED ({model_name})")
        print("=" * 60)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        y_test_thousands = y_test / 1000
        y_pred_thousands = y_pred / 1000
        
        axes[0].scatter(y_test_thousands, y_pred_thousands, alpha=0.3, s=10, c='steelblue')
        max_val = max(y_test_thousands.max(), y_pred_thousands.max())
        axes[0].plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Price (CAD, thousands)')
        axes[0].set_ylabel('Predicted Price (CAD, thousands)')
        axes[0].set_title(f'Actual vs Predicted - {model_name}')
        axes[0].legend()
        
        residuals = (y_test - y_pred) / 1000
        axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='coral')
        axes[1].set_xlabel('Residuals (CAD, thousands)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residuals Distribution')
        axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/actual_vs_predicted.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir}/actual_vs_predicted.png")
        
    def plot_model_comparison(self, evaluation_results: Dict):
        """Plot model comparison."""
        print("\n" + "=" * 60)
        print("VISUALIZATION: MODEL COMPARISON")
        print("=" * 60)
        
        valid_results = {k: v for k, v in evaluation_results.items() if 'error' not in v}
        
        if not valid_results:
            print("No valid results to plot")
            return
        
        models = list(valid_results.keys())
        r2_scores = [valid_results[m]['R2'] for m in models]
        rmse_scores = [valid_results[m]['RMSE'] / 1000 for m in models]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].barh(models, r2_scores, color='steelblue')
        axes[0].set_xlabel('R² Score')
        axes[0].set_title('Model Comparison by R² Score')
        axes[0].set_xlim(0, max(r2_scores) * 1.2)
        for i, v in enumerate(r2_scores):
            axes[0].text(v + 0.01, i, f'{v:.3f}', va='center')
        
        axes[1].barh(models, rmse_scores, color='coral')
        axes[1].set_xlabel('RMSE (CAD, thousands)')
        axes[1].set_title('Model Comparison by RMSE')
        for i, v in enumerate(rmse_scores):
            axes[1].text(v + 1, i, f'{v:.0f}', va='center')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir}/model_comparison.png")
        
    def plot_correlation_matrix(self, df: pd.DataFrame, columns: List[str]):
        """Plot correlation matrix for numerical features."""
        print("\n" + "=" * 60)
        print("VISUALIZATION: CORRELATION MATRIX")
        print("=" * 60)
        
        available_cols = [col for col in columns if col in df.columns]
        
        if len(available_cols) > 1:
            corr_matrix = df[available_cols].corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
            plt.title('Correlation Matrix')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/correlation_matrix.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {self.output_dir}/correlation_matrix.png")
        
    def create_eda_summary(self, df: pd.DataFrame):
        """Create comprehensive EDA summary plot."""
        print("\n" + "=" * 60)
        print("VISUALIZATION: EDA SUMMARY")
        print("=" * 60)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        prices = df['price'].dropna() / 1000
        axes[0, 0].hist(prices, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 0].set_xlabel('Price (CAD, thousands)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Price Distribution')
        
        sqft = df['property-sqft'].fillna(0)
        axes[0, 1].scatter(sqft, prices, alpha=0.3, s=10, c='coral')
        axes[0, 1].set_xlabel('Square Footage')
        axes[0, 1].set_ylabel('Price (CAD, thousands)')
        axes[0, 1].set_title('Price vs Square Footage')
        
        locality_avg = df.groupby('addressLocality')['price'].mean().sort_values(ascending=False).head(10)
        axes[1, 0].barh(range(len(locality_avg)), locality_avg.values / 1000, color='green')
        axes[1, 0].set_yticks(range(len(locality_avg)))
        axes[1, 0].set_yticklabels(locality_avg.index, fontsize=8)
        axes[1, 0].set_xlabel('Average Price (CAD, thousands)')
        axes[1, 0].set_title('Top 10 Localities by Price')
        axes[1, 0].invert_yaxis()
        
        beds_price = df.groupby('property-beds')['price'].mean()
        beds_price = beds_price[beds_price.index.notna() & (beds_price.index <= 10)]
        axes[1, 1].bar(beds_price.index.astype(str), beds_price.values / 1000, color='purple')
        axes[1, 1].set_xlabel('Bedrooms')
        axes[1, 1].set_ylabel('Average Price (CAD, thousands)')
        axes[1, 1].set_title('Average Price by Bedrooms')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/eda_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.output_dir}/eda_summary.png")


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
    
    viz = Visualizer()
    viz.create_eda_summary(df)
    viz.plot_price_distribution(df)
    viz.plot_price_vs_sqft(df)
    viz.plot_locality_prices(df)
    viz.plot_bedroom_prices(df)
    viz.plot_property_type_distribution(df)
