#!/usr/bin/env python3
"""
NL House Price Prediction Project
A comprehensive machine learning pipeline for predicting house prices in Newfoundland and Labrador.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class NLHousePriceML:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        self.preprocessing_pipeline = None
        
    def load_data(self):
        """Load and initial exploration of the dataset."""
        print("=" * 60)
        print("STEP 1: DATA LOADING")
        print("=" * 60)
        
        self.df = pd.read_csv(self.data_path, sep=',', on_bad_lines='skip')
        print(f"Dataset shape: {self.df.shape}")
        print(f"Total rows: {len(self.df)}")
        print(f"Total columns: {len(self.df.columns)}")
        print(f"\nColumn names (first 20):\n{list(self.df.columns[:20])}")
        
        print(f"\nData types summary:")
        print(self.df.dtypes.value_counts())
        
        print(f"\nFirst 5 rows:")
        print(self.df.head())
        
        return self.df
    
    def explore_data(self):
        """Comprehensive exploratory data analysis."""
        print("\n" + "=" * 60)
        print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
        print("=" * 60)
        
        if self.df is None:
            self.load_data()
        
        key_columns = [
            'streetAddress', 'addressLocality', 'addressRegion', 'postalCode',
            'latitude', 'longitude', 'description', 'price', 'property-beds',
            'property-baths', 'property-sqft', 'Year Built', 'Property Type'
        ]
        
        available_cols = [col for col in key_columns if col in self.df.columns]
        print(f"\nKey columns available: {available_cols}")
        
        print("\n--- Target Variable (price) Statistics ---")
        if 'price' in self.df.columns:
            print(self.df['price'].describe())
            print(f"\nMissing values in price: {self.df['price'].isna().sum()}")
            print(f"Zero prices: {(self.df['price'] == 0).sum()}")
        
        print("\n--- Numerical Features Statistics ---")
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Numerical columns count: {len(numerical_cols)}")
        print(f"\nNumerical columns: {numerical_cols[:15]}...")
        
        print("\n--- Categorical Features Statistics ---")
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        print(f"Categorical columns count: {len(categorical_cols)}")
        
        print("\n--- Missing Values Analysis ---")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({'Missing': missing, 'Percentage': missing_pct})
        missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Percentage', ascending=False)
        print(f"Columns with missing values: {len(missing_df)}")
        print(f"\nTop 10 columns with most missing values:")
        print(missing_df.head(10))
        
        return missing_df
    
    def clean_data(self):
        """Clean and preprocess the dataset."""
        print("\n" + "=" * 60)
        print("STEP 3: DATA CLEANING")
        print("=" * 60)
        
        df_clean = self.df.copy()
        
        print(f"Original shape: {df_clean.shape}")
        
        if 'price' in df_clean.columns:
            df_clean = df_clean[df_clean['price'].notna()]
            df_clean = df_clean[df_clean['price'] > 0]
            print(f"After removing null/zero prices: {len(df_clean)}")
        
        numeric_cols = ['property-beds', 'property-baths', 'property-sqft', 'Year Built', 
                       'latitude', 'longitude', 'Garage Spaces', 'Parking Spaces']
        
        for col in numeric_cols:
            if col in df_clean.columns:
                if col == 'property-sqft':
                    df_clean[col] = df_clean[col].astype(str).str.replace(',', '').replace('N/A', np.nan)
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        self.df_clean = df_clean
        print(f"Final cleaned shape: {df_clean.shape}")
        
        return df_clean
    
    def feature_engineering(self):
        """Create and select features for modeling."""
        print("\n" + "=" * 60)
        print("STEP 4: FEATURE ENGINEERING")
        print("=" * 60)
        
        df = self.df_clean.copy()
        
        selected_features = [
            'price',
            'latitude',
            'longitude',
            'property-beds',
            'property-baths',
            'property-sqft',
            'Year Built',
            'addressLocality',
            'addressRegion',
            'Property Type',
            'MLS® #'
        ]
        
        available_features = [col for col in selected_features if col in df.columns]
        print(f"Available features for modeling: {available_features}")
        
        df_model = df[available_features].copy()
        
        if 'addressLocality' in df_model.columns:
            locality_counts = df_model['addressLocality'].value_counts()
            top_localities = locality_counts.head(20).index.tolist()
            df_model['locality_encoded'] = df_model['addressLocality'].apply(
                lambda x: x if x in top_localities else 'Other' if pd.notna(x) else 'Unknown'
            )
            print(f"Unique localities: {df_model['addressLocality'].nunique()}")
            print(f"Top 5 localities by count:\n{locality_counts.head()}")
        
        if 'addressRegion' in df_model.columns:
            print(f"Unique regions: {df_model['addressRegion'].unique()}")
        
        if 'Property Type' in df_model.columns:
            print(f"Property types distribution:\n{df_model['Property Type'].value_counts()}")
        
        df_model['price_per_sqft'] = df_model['price'] / df_model['property-sqft'].replace(0, np.nan)
        
        df_model['house_age'] = datetime.now().year - df_model['Year Built']
        df_model['house_age'] = df_model['house_age'].replace({np.nan: df_model['house_age'].median()})
        
        df_model['total_rooms'] = df_model['property-beds'].fillna(0) + df_model['property-baths'].fillna(0)
        
        self.df_model = df_model
        print(f"\nFeature engineered dataset shape: {df_model.shape}")
        print(f"\nFeatures created:")
        print(f"  - locality_encoded: Locality encoding")
        print(f"  - price_per_sqft: Price per square foot")
        print(f"  - house_age: Age of the house")
        print(f"  - total_rooms: Total number of rooms")
        
        return df_model
    
    def prepare_features(self):
        """Prepare features for machine learning."""
        print("\n" + "=" * 60)
        print("STEP 5: FEATURE PREPARATION FOR ML")
        print("=" * 60)
        
        df = self.df_model.copy()
        
        feature_cols = ['latitude', 'longitude', 'property-beds', 'property-baths', 
                       'property-sqft', 'Year Built', 'house_age', 'total_rooms']
        
        if 'price' in df.columns:
            target_col = 'price'
            feature_cols_used = [col for col in feature_cols if col in df.columns]
            
            X = df[feature_cols_used].copy()
            y = df[target_col].copy()
            
            X = X.fillna(X.median())
            
            if 'property-sqft' in X.columns:
                X['log_sqft'] = np.log1p(X['property-sqft'].clip(lower=1))
            
            if 'price' in df.columns:
                y_log = np.log1p(y)
            
            print(f"Features used: {feature_cols_used}")
            print(f"Feature matrix shape: {X.shape}")
            print(f"Target shape: {y.shape}")
            
            self.X = X
            self.y = y
            self.y_log = y_log
            
            return X, y, y_log
        
        return None, None, None
    
    def train_test_split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets."""
        print("\n" + "=" * 60)
        print("STEP 6: TRAIN-TEST SPLIT")
        print("=" * 60)
        
        X = self.X
        y = self.y
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Testing set size: {len(X_test)}")
        print(f"Train/Test ratio: {1-test_size:.0%}/{test_size:.0%}")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self):
        """Scale numerical features."""
        print("\n" + "=" * 60)
        print("STEP 7: FEATURE SCALING")
        print("=" * 60)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        self.scaler = scaler
        
        print(f"Scaled training features shape: {X_train_scaled.shape}")
        print(f"Scaled testing features shape: {X_test_scaled.shape}")
        
        print(f"\nFeature means after scaling: {X_train_scaled.mean(axis=0).round(4)}")
        print(f"Feature stds after scaling: {X_train_scaled.std(axis=0).round(4)}")
        
        return X_train_scaled, X_test_scaled
    
    def train_models(self, X_train, y_train):
        """Train multiple regression models."""
        print("\n" + "=" * 60)
        print("STEP 8: MODEL TRAINING")
        print("=" * 60)
        
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'K-Neighbors': KNeighborsRegressor(n_neighbors=5)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            try:
                model.fit(X_train, y_train)
                self.models[name] = model
                
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                
                results[name] = {
                    'cv_r2_mean': cv_scores.mean(),
                    'cv_r2_std': cv_scores.std()
                }
                
                print(f"  CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
                
            except Exception as e:
                print(f"  Error training {name}: {str(e)}")
        
        self.model_results = results
        
        print("\n--- Model Comparison Summary ---")
        results_df = pd.DataFrame(results).T
        results_df = results_df.sort_values('cv_r2_mean', ascending=False)
        print(results_df)
        
        return results
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models on test set."""
        print("\n" + "=" * 60)
        print("STEP 9: MODEL EVALUATION")
        print("=" * 60)
        
        evaluation_results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            evaluation_results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
            
            print(f"\n{name}:")
            print(f"  RMSE: ${rmse:,.2f}")
            print(f"  MAE: ${mae:,.2f}")
            print(f"  R² Score: {r2:.4f}")
        
        self.evaluation_results = evaluation_results
        
        eval_df = pd.DataFrame(evaluation_results).T
        eval_df = eval_df.sort_values('R2', ascending=False)
        
        print("\n--- Final Model Ranking ---")
        print(eval_df)
        
        best_model_name = eval_df['R2'].idxmax()
        self.best_model = self.models[best_model_name]
        print(f"\nBest Model: {best_model_name} with R² = {eval_df.loc[best_model_name, 'R2']:.4f}")
        
        return evaluation_results
    
    def feature_importance_analysis(self):
        """Analyze feature importance for tree-based models."""
        print("\n" + "=" * 60)
        print("STEP 10: FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)
        
        feature_names = self.X.columns.tolist()
        
        importance_results = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                importance_results[name] = dict(zip(feature_names, importance))
                
                print(f"\n{name} Feature Importances:")
                sorted_importance = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
                for feat, imp in sorted_importance:
                    print(f"  {feat}: {imp:.4f}")
        
        self.feature_importance = importance_results
        
        if 'Random Forest' in importance_results:
            rf_importance = importance_results['Random Forest']
            sorted_imp = sorted(rf_importance.items(), key=lambda x: x[1], reverse=True)
            
            plt.figure(figsize=(10, 6))
            features, importances = zip(*sorted_imp)
            plt.barh(range(len(features)), importances, align='center')
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title('Random Forest Feature Importance')
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/feature_importance.png', dpi=150)
            plt.close()
            print(f"\nFeature importance plot saved to {OUTPUT_DIR}/feature_importance.png")
        
        return importance_results
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for the best model."""
        print("\n" + "=" * 60)
        print("STEP 11: HYPERPARAMETER TUNING")
        print("=" * 60)
        
        if 'Random Forest' in self.models:
            print("\nTuning Random Forest...")
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            rf = RandomForestRegressor(random_state=42, n_jobs=-1)
            
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            print(f"\nBest parameters: {grid_search.best_params_}")
            print(f"Best CV R² score: {grid_search.best_score_:.4f}")
            
            self.best_model = grid_search.best_estimator_
            
            return grid_search
        
        return None
    
    def create_visualizations(self):
        """Create data visualizations."""
        print("\n" + "=" * 60)
        print("STEP 12: VISUALIZATIONS")
        print("=" * 60)
        
        df = self.df_model
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].hist(df['price'].dropna() / 1000, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Price (CAD, thousands)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Price Distribution')
        
        axes[0, 1].scatter(df['property-sqft'].fillna(0), df['price'].fillna(0) / 1000, 
                          alpha=0.3, s=10)
        axes[0, 1].set_xlabel('Square Footage')
        axes[0, 1].set_ylabel('Price (CAD, thousands)')
        axes[0, 1].set_title('Price vs Square Footage')
        
        locality_avg = df.groupby('addressLocality')['price'].mean().sort_values(ascending=False).head(15)
        axes[1, 0].barh(range(len(locality_avg)), locality_avg.values / 1000)
        axes[1, 0].set_yticks(range(len(locality_avg)))
        axes[1, 0].set_yticklabels(locality_avg.index, fontsize=8)
        axes[1, 0].set_xlabel('Average Price (CAD, thousands)')
        axes[1, 0].set_title('Top 15 Localities by Average Price')
        
        beds_price = df.groupby('property-beds')['price'].mean()
        beds_price = beds_price[beds_price.index.notna()]
        axes[1, 1].bar(beds_price.index.astype(str), beds_price.values / 1000)
        axes[1, 1].set_xlabel('Number of Bedrooms')
        axes[1, 1].set_ylabel('Average Price (CAD, thousands)')
        axes[1, 1].set_title('Average Price by Bedrooms')
        
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/eda_visualizations.png', dpi=150)
        plt.close()
        print(f"EDA visualizations saved to {OUTPUT_DIR}/eda_visualizations.png")
        
        if self.best_model is not None and self.X_test is not None:
            y_pred = self.best_model.predict(self.X_test)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            axes[0].scatter(self.y_test / 1000, y_pred / 1000, alpha=0.3, s=10)
            axes[0].plot([self.y_test.min(), self.y_test.max()], 
                        [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            axes[0].set_xlabel('Actual Price (CAD, thousands)')
            axes[0].set_ylabel('Predicted Price (CAD, thousands)')
            axes[0].set_title('Actual vs Predicted Prices')
            
            residuals = (self.y_test - y_pred) / 1000
            axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
            axes[1].set_xlabel('Residuals (CAD, thousands)')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Residuals Distribution')
            
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/model_evaluation.png', dpi=150)
            plt.close()
            print(f"Model evaluation plots saved to {OUTPUT_DIR}/model_evaluation.png")
    
    def generate_report(self):
        """Generate a comprehensive report."""
        print("\n" + "=" * 60)
        print("STEP 13: GENERATING REPORT")
        print("=" * 60)
        
        report = {
            'project': 'NL House Price Prediction',
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_info': {
                'total_rows': len(self.df),
                'total_columns': len(self.df.columns),
                'cleaned_rows': len(self.df_clean),
                'training_rows': len(self.X_train),
                'testing_rows': len(self.X_test)
            },
            'model_results': self.model_results,
            'evaluation_results': self.evaluation_results,
            'best_model': max(self.evaluation_results.items(), key=lambda x: x[1]['R2'])[0],
            'feature_importance': self.feature_importance
        }
        
        report_path = f'{OUTPUT_DIR}/model_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {report_path}")
        
        summary = f"""
================================================================================
                    NL HOUSE PRICE PREDICTION - FINAL SUMMARY
================================================================================

Dataset Statistics:
  - Original rows: {len(self.df)}
  - Cleaned rows: {len(self.df_clean)}
  - Training samples: {len(self.X_train)}
  - Testing samples: {len(self.X_test)}

Best Performing Model:
  {max(self.evaluation_results.items(), key=lambda x: x[1]['R2'])[0]}
  - R² Score: {max(e[1]['R2'] for e in self.evaluation_results.items()):.4f}
  - RMSE: ${min(e[1]['RMSE'] for e in self.evaluation_results.items()):,.2f}
  - MAE: ${min(e[1]['MAE'] for e in self.evaluation_results.items()):,.2f}

Output Files:
  - {OUTPUT_DIR}/feature_importance.png
  - {OUTPUT_DIR}/eda_visualizations.png
  - {OUTPUT_DIR}/model_evaluation.png
  - {OUTPUT_DIR}/model_report.json

================================================================================
"""
        print(summary)
        
        with open(f'{OUTPUT_DIR}/summary.txt', 'w') as f:
            f.write(summary)
        print(f"Summary saved to {OUTPUT_DIR}/summary.txt")
        
        return report
    
    def run_full_pipeline(self):
        """Execute the complete ML pipeline."""
        print("\n" + "=" * 60)
        print("   NL HOUSE PRICE PREDICTION - FULL PIPELINE   ")
        print("=" * 60)
        
        self.load_data()
        self.explore_data()
        self.clean_data()
        self.feature_engineering()
        self.prepare_features()
        self.train_test_split_data()
        X_train_scaled, X_test_scaled = self.scale_features()
        self.train_models(X_train_scaled, self.y_train)
        self.evaluate_models(X_test_scaled, self.y_test)
        self.feature_importance_analysis()
        self.hyperparameter_tuning()
        self.create_visualizations()
        self.generate_report()
        
        print("\n" + "=" * 60)
        print("   PIPELINE COMPLETED SUCCESSFULLY!   ")
        print("=" * 60)


if __name__ == "__main__":
    DATA_PATH = "data_nl.csv"
    
    if os.path.exists(DATA_PATH):
        ml_project = NLHousePriceML(DATA_PATH)
        ml_project.run_full_pipeline()
    else:
        print(f"Error: {DATA_PATH} not found!")
        print("Please ensure the data file is in the current directory.")
