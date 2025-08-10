"""
Core Module
Contains all shared functions and utilities for the fuel forecasting application.
This module centralizes feature engineering, model training, forecasting, and evaluation functions.
"""

import logging
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from helpers import ModelConfig

# Configure logging
logger = logging.getLogger(__name__)


@st.cache_data
def enhanced_feature_engineering(df: pd.DataFrame, forecasting_mode: bool = False, random_state: Optional[int] = None) -> pd.DataFrame:
    """Create deterministic features for model training and forecasting.
    
    Focus on trend, seasonality, and lag features while minimizing noisy exogenous variables.
    
    Args:
        df: DataFrame with sales data
        forecasting_mode: If True, use minimal random factors for forecasting
        random_state: Random seed for reproducibility in training, set to None for forecasting
    
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Ensure required columns exist
    if 'week_start' not in df.columns:
        raise ValueError("'week_start' column is required")
    
    # Always create lag features (core predictive features)
    for lag in [1, 2, 4, 8]:  # Extended lag features for better trend capture
        lag_col = f'lag_{lag}'
        df[lag_col] = df.groupby(['Region', 'Product'])['sales_volume'].shift(lag)
    
    # Always create month feature
    if pd.api.types.is_datetime64_any_dtype(df['week_start']):
        df['month'] = df['week_start'].dt.month
    else:
        df['month'] = pd.to_datetime(df['week_start']).dt.month
    
    # Always create is_holiday_week feature (deterministic seasonality)
    if pd.api.types.is_datetime64_any_dtype(df['week_start']):
        df['is_holiday_week'] = df['week_start'].dt.month.isin([12, 1, 7, 8]).astype(int)
    else:
        df['is_holiday_week'] = pd.to_datetime(df['week_start']).dt.month.isin([12, 1, 7, 8]).astype(int)
    
    # Always create week_of_year
    if pd.api.types.is_datetime64_any_dtype(df['week_start']):
        df['week_of_year'] = df['week_start'].dt.isocalendar().week
    else:
        df['week_of_year'] = pd.to_datetime(df['week_start']).dt.isocalendar().week
    
    # Create price-related features only if avg_price exists and is reliable
    if 'avg_price' in df.columns and not df['avg_price'].isna().all():
        # Use price features but make them more stable
        df['price_elasticity'] = np.where(df['avg_price'] != 0,
                                          df['sales_volume'] / df['avg_price'],
                                          np.nan)
        df['price_volatility'] = df.groupby(['Region', 'Product'])['avg_price'].transform(
            lambda x: x.rolling(8, min_periods=1).std()  # Longer window for stability
        )
        df['price_change'] = df.groupby(['Region', 'Product'])['avg_price'].transform(
            lambda x: x.pct_change().rolling(4, min_periods=1).mean()  # Smoothed price change
        )
        df['price_volume_ratio'] = df['avg_price'] * df['sales_volume']
        df['price_over_volume'] = np.where(df['sales_volume'] != 0,
                                           df['avg_price'] / df['sales_volume'],
                                           np.nan)
    else:
        # Minimal price features based on sales volume patterns
        df['price_elasticity'] = 1.0
        df['price_volatility'] = 0.05  # Low constant volatility
        df['price_change'] = 0.0
        df['price_volume_ratio'] = df['sales_volume'] * 100
        df['price_over_volume'] = 0.01

    # Core volume-related features (deterministic trends)
    df['volume_trend'] = df.groupby(['Region', 'Product'])['sales_volume'].transform(
        lambda x: x.rolling(12, min_periods=1).mean()  # Longer trend window
    )
    df['volume_change'] = df.groupby(['Region', 'Product'])['sales_volume'].transform(
        lambda x: x.pct_change().rolling(4, min_periods=1).mean()  # Smoothed change
    )
    df['volume_volatility'] = df.groupby(['Region', 'Product'])['sales_volume'].transform(
        lambda x: x.rolling(8, min_periods=1).std()  # Stable volatility measure
    )

    # Strong seasonal features (deterministic)
    df['seasonal_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['seasonal_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
    df['quarterly_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 13)
    df['quarterly_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 13)
    df['monthly_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['monthly_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Deterministic trend features
    df['trend_factor'] = df['week_of_year'] / 52
    df['trend_squared'] = (df['week_of_year'] / 52) ** 2
    
    # Add year-over-year trend
    if pd.api.types.is_datetime64_any_dtype(df['week_start']):
        df['year'] = df['week_start'].dt.year
    else:
        df['year'] = pd.to_datetime(df['week_start']).dt.year
    
    # Normalize year for trend calculation
    min_year = df['year'].min()
    df['year_trend'] = (df['year'] - min_year) / max(1, df['year'].max() - min_year)

    # Regional and product trends (deterministic)
    df['regional_trend'] = df.groupby('Region')['sales_volume'].transform(
        lambda x: x.rolling(16, min_periods=1).mean()  # Longer regional trend
    )
    df['product_trend'] = df.groupby('Product')['sales_volume'].transform(
        lambda x: x.rolling(16, min_periods=1).mean()  # Longer product trend
    )

    # NO external factors - they are artificial and add only noise
    # Removed weather_factor and economic_factor as they are preset random values
    # with no connection to actual sales data

    # Rolling features with longer windows for stability
    for window in [4, 8, 12, 16]:
        df[f'roll{window}_mean'] = df.groupby(['Region', 'Product'])['sales_volume'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'roll{window}_std'] = df.groupby(['Region', 'Product'])['sales_volume'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )

    # Add moving average ratios for trend detection
    df['ma4_ma12_ratio'] = df['roll4_mean'] / (df['roll12_mean'] + 1e-10)
    df['ma8_ma16_ratio'] = df['roll8_mean'] / (df['roll16_mean'] + 1e-10)

    # Ensure all features are created and handle any missing ones
    expected_features = [
        'lag_1', 'lag_2', 'lag_4', 'lag_8', 'month', 'is_holiday_week', 'week_of_year', 'year_trend',
        'price_elasticity', 'price_volatility', 'price_change', 'price_volume_ratio', 'price_over_volume',
        'volume_trend', 'volume_change', 'volume_volatility',
        'seasonal_sin', 'seasonal_cos', 'quarterly_sin', 'quarterly_cos', 'monthly_sin', 'monthly_cos',
        'trend_factor', 'trend_squared', 'regional_trend', 'product_trend',
        'roll4_mean', 'roll4_std', 'roll8_mean', 'roll8_std', 'roll12_mean', 'roll12_std', 'roll16_mean', 'roll16_std',
        'ma4_ma12_ratio', 'ma8_ma16_ratio'
    ]
    
    # Create any missing features with default values
    for feature in expected_features:
        if feature not in df.columns:
            if feature.startswith('lag_'):
                df[feature] = np.nan
            elif feature in ['price_elasticity', 'price_volatility', 'price_change', 'price_volume_ratio', 'price_over_volume']:
                df[feature] = np.nan
            else:
                df[feature] = 0.0

    return df


@st.cache_data
def evaluate_preds(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluate model predictions with comprehensive metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary with evaluation metrics
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = r2_score(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = float(np.mean(np.abs(y_true - y_pred) / np.where(denom == 0, 1, denom)))
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'SMAPE': smape}


def create_missing_features(data: pd.DataFrame, required_features: list) -> pd.DataFrame:
    """Create any missing features with appropriate default values.
    
    Args:
        data: DataFrame to add features to
        required_features: List of required feature names
    
    Returns:
        DataFrame with all required features
    """
    data = data.copy()
    
    for col in required_features:
        if col not in data.columns:
            if col.startswith('lag_'):
                # For lag features, use the last available sales volume
                lag_num = int(col.split('_')[1])
                if 'sales_volume' in data.columns and len(data) > lag_num:
                    data[col] = data['sales_volume'].shift(lag_num)
                else:
                    # Use a reasonable default based on available sales volume
                    if 'sales_volume' in data.columns and len(data) > 0:
                        data[col] = data['sales_volume'].mean()
                    else:
                        data[col] = 1000.0  # Default sales volume
            elif col in ['price_elasticity', 'price_volatility', 'price_change', 'price_volume_ratio', 'price_over_volume']:
                # For price-related features, use sales volume as proxy
                if 'sales_volume' in data.columns:
                    if col == 'price_elasticity':
                        data[col] = data['sales_volume'] / data['sales_volume'].mean() if data['sales_volume'].mean() > 0 else 1.0
                    elif col == 'price_volatility':
                        data[col] = data['sales_volume'].rolling(4, min_periods=1).std() / (data['sales_volume'].mean() + 1e-10)
                    elif col == 'price_change':
                        data[col] = data['sales_volume'].pct_change() * 0.1
                    elif col == 'price_volume_ratio':
                        data[col] = data['sales_volume'] * 100
                    elif col == 'price_over_volume':
                        data[col] = np.where(data['sales_volume'] != 0, 100 / data['sales_volume'], 1.0)
                else:
                    # Default values when sales_volume is not available
                    if col == 'price_elasticity':
                        data[col] = 1.0
                    elif col == 'price_volatility':
                        data[col] = 0.1
                    elif col == 'price_change':
                        data[col] = 0.0
                    elif col == 'price_volume_ratio':
                        data[col] = 100000.0
                    elif col == 'price_over_volume':
                        data[col] = 0.1
            elif col in ['event_factor']:
                # For external factors, use realistic random values
                np.random.seed(42)  # For consistency
                if col == 'event_factor':
                    data[col] = np.random.normal(1, 0.15, len(data))
            elif col in ['avg_price']:
                # For avg_price, use a reasonable default
                data[col] = 100.0
            elif col in ['month']:
                # Extract month from week_start if available
                if 'week_start' in data.columns:
                    try:
                        # Ensure week_start is datetime
                        if not pd.api.types.is_datetime64_any_dtype(data['week_start']):
                            data['week_start'] = pd.to_datetime(data['week_start'])
                        data[col] = data['week_start'].dt.month
                    except Exception:
                        # If conversion fails, use a default month
                        data[col] = 6  # Default to June
                else:
                    # Default to June if no week_start column
                    data[col] = 6
            elif col in ['is_holiday_week']:
                # Extract holiday weeks from month if available
                if 'month' in data.columns:
                    data[col] = data['month'].isin([12, 1, 7, 8]).astype(int)
                elif 'week_start' in data.columns:
                    try:
                        # Ensure week_start is datetime
                        if not pd.api.types.is_datetime64_any_dtype(data['week_start']):
                            data['week_start'] = pd.to_datetime(data['week_start'])
                        month_col = data['week_start'].dt.month
                        data[col] = month_col.isin([12, 1, 7, 8]).astype(int)
                    except Exception:
                        # Default to non-holiday
                        data[col] = 0
                else:
                    # Default to non-holiday
                    data[col] = 0
            elif col.startswith('monthly_'):
                # Monthly cyclical features
                month_col = None
                if 'month' in data.columns:
                    month_col = data['month']
                elif 'week_start' in data.columns:
                    try:
                        # Ensure week_start is datetime
                        if not pd.api.types.is_datetime64_any_dtype(data['week_start']):
                            data['week_start'] = pd.to_datetime(data['week_start'])
                        month_col = data['week_start'].dt.month
                        # Also create the month column for future use
                        if 'month' not in data.columns:
                            data['month'] = month_col
                    except Exception:
                        month_col = pd.Series([6] * len(data), index=data.index)  # Default to June
                else:
                    month_col = pd.Series([6] * len(data), index=data.index)  # Default to June
                
                if col == 'monthly_sin':
                    data[col] = np.sin(2 * np.pi * month_col / 12)
                elif col == 'monthly_cos':
                    data[col] = np.cos(2 * np.pi * month_col / 12)
            else:
                # For any other missing features, use zero
                data[col] = 0.0
    
    return data


def _tree_based_feature_selection(X_train: pd.DataFrame, y_train: pd.Series, 
                                 feature_cols: list, config: ModelConfig) -> tuple:
    """Perform tree-based feature selection using LightGBM feature importance.
    
    Args:
        X_train: Training features
        y_train: Training target
        feature_cols: List of feature column names
        config: Model configuration
        
    Returns:
        Tuple of (selector, selected_features)
    """
    # Create a temporary LightGBM model for feature selection
    lgbm_selector = lgb.LGBMRegressor(
        learning_rate=0.1,  # Faster learning for feature selection
        n_estimators=100,   # Fewer trees for speed
        max_depth=config.lgbm_max_depth,
        random_state=42,
        verbose=-1  # Suppress output
    )
    
    # Use SelectFromModel with LightGBM
    selector = SelectFromModel(
        lgbm_selector, 
        threshold='median',  # Select features above median importance
        max_features=getattr(config, 'k_features', 20)  # Respect max features limit
    )
    
    # Fit the selector
    selector.fit(X_train, y_train)
    
    # Get selected features
    selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
    
    return selector, selected_features


def train_models(df: pd.DataFrame, feature_cols: list, config: ModelConfig) -> Dict[str, Any]:
    """Train and evaluate models with proper cross-validation, then retrain on full dataset for forecasting.
    
    This function follows a comprehensive process:
    1. Split data and evaluate model performance on test set
    2. Perform time series cross-validation for robust evaluation
    3. Retrain models on the full dataset to incorporate all available information
    
    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        config: Model configuration object
    
    Returns:
        Dictionary with trained models and evaluation metrics
    """
    from sklearn.model_selection import TimeSeriesSplit
    from helpers import time_series_cv_scores
    
    # Step 1: Split data and evaluate model performance
    split_method = (config.split_method or 'Time-based')
    train_ratio = config.train_ratio
    if split_method.lower() in ['time', 'time-based', 'time based']:
        split_point = int(len(df) * train_ratio)
        train = df.iloc[:split_point]
        test = df.iloc[split_point:]
    else:
        train, test = train_test_split(df, train_size=train_ratio, random_state=42, shuffle=True)

    X_train, y_train = train[feature_cols], train['sales_volume']
    X_test, y_test = test[feature_cols], test['sales_volume']

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_clean = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_clean = pd.DataFrame(imputer.transform(X_test), columns=feature_cols, index=X_test.index)

    # Feature selection if enabled - using tree-based feature importance
    selector = None
    if getattr(config, 'feature_selection', True):
        # Use tree-based feature selection with LightGBM feature importance
        selector, selected_features = _tree_based_feature_selection(
            X_train_clean, y_train, feature_cols, config
        )
        X_train_sel = selector.transform(X_train_clean)
        X_test_sel = selector.transform(X_test_clean)
    else:
        X_train_sel, X_test_sel = X_train_clean.values, X_test_clean.values
        selected_features = feature_cols

    # Initialize models
    lgbm = lgb.LGBMRegressor(
        learning_rate=config.lgbm_learning_rate,
        n_estimators=config.lgbm_n_estimators,
        max_depth=config.lgbm_max_depth,
        random_state=42,
        verbose=-1  # Suppress output
    )
    rf = RandomForestRegressor(
        n_estimators=config.rf_n_estimators,
        max_depth=config.rf_max_depth,
        min_samples_split=config.rf_min_samples_split,
        random_state=42
    )

    # Train models on training set
    lgbm.fit(X_train_sel, y_train)
    rf.fit(X_train_sel, y_train)

    # Make predictions and evaluate on test set
    pred_lgbm = lgbm.predict(X_test_sel)
    pred_rf = rf.predict(X_test_sel)
    
    # Individual model metrics
    lgbm_metrics = evaluate_preds(y_test, pred_lgbm)
    rf_metrics = evaluate_preds(y_test, pred_rf)
    
    # Ensemble prediction
    if config.ensemble_method == 'Weighted Average':
        weight = config.lgbm_weight
        ensemble_pred = weight * pred_lgbm + (1 - weight) * pred_rf
    else:
        ensemble_pred = (pred_lgbm + pred_rf) / 2

    # Calculate ensemble metrics and residual standard deviation
    metrics = evaluate_preds(y_test, ensemble_pred)
    residual_std = float(np.std(y_test - ensemble_pred, ddof=1)) if len(y_test) > 1 else 0.0
    
    # Step 2: Perform time series cross-validation for robust evaluation
    cv_metrics = {}
    try:
        # Define fit_predict function for cross-validation
        def fit_predict_fn(X_tr, y_tr, X_te):
            # Apply imputation within each fold to prevent data leakage
            fold_imputer = SimpleImputer(strategy='mean')
            X_tr_clean = pd.DataFrame(fold_imputer.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
            X_te_clean = pd.DataFrame(fold_imputer.transform(X_te), columns=X_te.columns, index=X_te.index)
            
            # Apply feature selection within each fold if enabled
            if getattr(config, 'feature_selection', True):
                # Use tree-based feature selection within each fold
                fold_selector, fold_selected_features = _tree_based_feature_selection(
                    X_tr_clean, y_tr, list(X_tr.columns), config
                )
                X_tr_sel = fold_selector.transform(X_tr_clean)
                X_te_sel = fold_selector.transform(X_te_clean)
            else:
                X_tr_sel = X_tr_clean.values
                X_te_sel = X_te_clean.values
            
            # Train models
            lgbm_cv = lgb.LGBMRegressor(
                learning_rate=config.lgbm_learning_rate,
                n_estimators=config.lgbm_n_estimators,
                max_depth=config.lgbm_max_depth,
                random_state=42,
                verbose=-1
            )
            rf_cv = RandomForestRegressor(
                n_estimators=config.rf_n_estimators,
                max_depth=config.rf_max_depth,
                min_samples_split=config.rf_min_samples_split,
                random_state=42
            )
            
            lgbm_cv.fit(X_tr_sel, y_tr)
            rf_cv.fit(X_tr_sel, y_tr)
            
            # Make predictions
            pred_lgbm_cv = lgbm_cv.predict(X_te_sel)
            pred_rf_cv = rf_cv.predict(X_te_sel)
            
            # Ensemble prediction
            if config.ensemble_method == 'Weighted Average':
                return config.lgbm_weight * pred_lgbm_cv + (1 - config.lgbm_weight) * pred_rf_cv
            else:
                return (pred_lgbm_cv + pred_rf_cv) / 2
        
        # Perform time series cross-validation using raw training data to prevent data leakage
        cv_metrics = time_series_cv_scores(
            X_train, y_train, fit_predict_fn, n_splits=5
        )
        
    except Exception as e:
        logger.warning(f"Cross-validation failed: {str(e)}")
        cv_metrics = {'SMAPE': 0.0, 'MAE': 0.0, 'RMSE': 0.0, 'R2': 0.0}
    
    # Store test set and predictions for later use
    test_data = {
        'y_test': y_test,
        'y_pred': ensemble_pred,
        'test_df': test
    }

    # Step 3: Retrain models on full dataset for forecasting
    imputer_full = SimpleImputer(strategy='mean')
    X_full = df[feature_cols]
    X_full_clean = pd.DataFrame(imputer_full.fit_transform(X_full), columns=feature_cols, index=df.index)
    
    # Apply feature selection on full dataset if needed
    if selector is not None:
        # Use the same tree-based feature selection approach for full dataset
        selector_full, final_features = _tree_based_feature_selection(
            X_full_clean, df['sales_volume'], feature_cols, config
        )
        X_full_sel = selector_full.transform(X_full_clean)
        # Ensure X_full_sel is a 2D array
        if isinstance(X_full_sel, np.ndarray):
            if X_full_sel.ndim == 1:
                logger.debug(f"Feature selection produced a 1D array with shape {X_full_sel.shape}. Reshaping to 2D.")
                X_full_sel = X_full_sel.reshape(1, -1)
            elif X_full_sel.ndim > 2:
                logger.debug(f"Feature selection produced a {X_full_sel.ndim}D array with shape {X_full_sel.shape}. Reshaping to 2D.")
                X_full_sel = X_full_sel.reshape(X_full_sel.shape[0], -1)
    else:
        selector_full = None
        X_full_sel = X_full_clean.values
        final_features = feature_cols

    # Initialize and train models on full dataset
    lgbm_full = lgb.LGBMRegressor(
        learning_rate=config.lgbm_learning_rate,
        n_estimators=config.lgbm_n_estimators,
        max_depth=config.lgbm_max_depth,
        random_state=42,
        verbose=-1
    )
    rf_full = RandomForestRegressor(
        n_estimators=config.rf_n_estimators,
        max_depth=config.rf_max_depth,
        min_samples_split=config.rf_min_samples_split,
        random_state=42
    )
    lgbm_full.fit(X_full_sel, df['sales_volume'])
    rf_full.fit(X_full_sel, df['sales_volume'])

    # Return model bundle with all necessary components
    model_bundle = {
        'models': {'lgbm': lgbm_full, 'rf': rf_full},
        'imputer': imputer_full,
        'selector': selector_full,
        'feature_cols': feature_cols,
        'selected_features': final_features,
        'ensemble_method': config.ensemble_method,
        'lgbm_weight': config.lgbm_weight,
        'residual_std': residual_std,
        'metrics': metrics,
        'lgbm_metrics': lgbm_metrics,
        'rf_metrics': rf_metrics,
        'cv_metrics': cv_metrics,
        'test_data': test_data
    }
    return model_bundle


def generate_forecast(model_bundle: Dict[str, Any], history: pd.DataFrame, steps: int, 
                     include_confidence: bool = True, confidence_level: float = 0.95) -> Dict[str, Any]:
    """Generate forecasts with proper feature recalculation at each step.
    
    Args:
        model_bundle: Dictionary containing trained models and preprocessing components
        history: Historical data to use as a basis for forecasting
        steps: Number of time steps to forecast
        include_confidence: Whether to include confidence intervals
        confidence_level: Confidence level for prediction intervals (0-1)
        
    Returns:
        Dictionary with forecast dates, values, and confidence intervals
    """
    # Extract model components - handle both direct and nested structures
    if 'results' in model_bundle:
        # This is the structure from region training
        models = model_bundle['results'].get('models', {})
        lgbm = models.get('lgbm')
        rf = models.get('rf')
        imputer = model_bundle['results'].get('imputer')
        selector = model_bundle['results'].get('selector')
        feature_cols = model_bundle['results'].get('feature_cols', [])
        ensemble_method = model_bundle['results'].get('ensemble_method', 'Average')
        lgbm_weight = model_bundle['results'].get('lgbm_weight', 0.5)
        residual_std = model_bundle['results'].get('residual_std', 0.0)
        test_data = model_bundle['results'].get('test_data', {})
    else:
        # This is the direct structure (for overall models)
        models = model_bundle.get('models', {})
        lgbm = models.get('lgbm')
        rf = models.get('rf')
        imputer = model_bundle.get('imputer')
        selector = model_bundle.get('selector')
        feature_cols = model_bundle.get('feature_cols', [])
        ensemble_method = model_bundle.get('ensemble_method', 'Average')
        lgbm_weight = model_bundle.get('lgbm_weight', 0.5)
        residual_std = model_bundle.get('residual_std', 0.0)
        test_data = model_bundle.get('test_data', {})
    
    # Check if we have all required components
    if not feature_cols or lgbm is None or rf is None or imputer is None:
        st.warning(f"Missing required model components. Models: {lgbm is not None}, RF: {rf is not None}, Imputer: {imputer is not None}, Features: {len(feature_cols) if feature_cols else 0}")
        return {
            'dates': [],
            'values': [],
            'lower_bounds': [],
            'upper_bounds': [],
        }
    
    # Get test data metrics for better confidence intervals
    y_test = test_data.get('y_test', pd.Series(dtype=float))
    y_pred = test_data.get('y_pred', pd.Series(dtype=float))
    
    # Calculate error metrics if available
    if len(y_test) > 0 and len(y_pred) > 0:
        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
        error_estimate = (0.7 * mae) + (0.3 * rmse)
    else:
        error_estimate = residual_std

    # Prepare history data
    history = history.copy().sort_values('week_start')
    if not pd.api.types.is_datetime64_any_dtype(history['week_start']):
        history['week_start'] = pd.to_datetime(history['week_start'])

    # Ensure all required features are created
    history = enhanced_feature_engineering(history, forecasting_mode=False, random_state=42)
    history = create_missing_features(history, feature_cols)

    # Initialize forecast containers
    forecasts, dates, lower, upper = [], [], [], []
    z_lookup = {0.99: 2.58, 0.95: 1.96, 0.9: 1.645, 0.8: 1.28}
    z = z_lookup.get(confidence_level, 1.96)

    # Generate forecasts iteratively
    for step in range(steps):
        # Create new row with next date
        next_date = history['week_start'].iloc[-1] + timedelta(days=7)
        new_row = history.iloc[-1:].copy()
        new_row['week_start'] = next_date
        
        # Add new row to history
        history = pd.concat([history, new_row], ignore_index=True)
        
        # Recalculate ALL features with forecasting_mode=True for realistic external factors
        eng = enhanced_feature_engineering(history, forecasting_mode=True)
        eng = create_missing_features(eng, feature_cols)
        
        # Get features for prediction
        features = eng[feature_cols].iloc[-1:]

        # Apply preprocessing
        features_clean = pd.DataFrame(imputer.transform(features), columns=feature_cols, index=features.index)
        if selector is not None:
            features_sel = selector.transform(features_clean)
            # Ensure we have a 2D array after feature selection
            if isinstance(features_sel, np.ndarray):
                if features_sel.ndim == 1:
                    features_sel = features_sel.reshape(1, -1)
                elif features_sel.ndim > 2:
                    features_sel = features_sel.reshape(features_sel.shape[0], -1)
        else:
            features_sel = features_clean.values
            
        # Ensure input is 2D for model prediction
        if isinstance(features_sel, np.ndarray):
            if features_sel.ndim == 1:
                features_sel = features_sel.reshape(1, -1)
            elif features_sel.ndim > 2:
                features_sel = features_sel.reshape(features_sel.shape[0], -1)

        # Make predictions with both models
        pred_lgbm = lgbm.predict(features_sel)
        pred_rf = rf.predict(features_sel)
        
        # Combine predictions according to ensemble method
        if ensemble_method == 'Weighted Average':
            forecast = lgbm_weight * pred_lgbm + (1 - lgbm_weight) * pred_rf
        else:
            forecast = (pred_lgbm + pred_rf) / 2
            
        # Store forecast and date
        forecasts.append(forecast[0])
        dates.append(next_date)

        # Calculate confidence intervals with increasing uncertainty over time
        if include_confidence:
            # Increase uncertainty for longer forecast horizons
            step_factor = 1 + (step * 0.1)  # 10% increase in uncertainty per step
            margin = z * error_estimate * step_factor
            
            # Ensure lower bound is non-negative
            lower.append(max(0, forecast[0] - margin))
            upper.append(forecast[0] + margin)

        # Update history with the forecasted value for next iteration
        history.loc[history.index[-1], 'sales_volume'] = forecast[0]

    # Return results as a dictionary
    result = {
        'dates': dates,
        'values': forecasts,
        'lower_bounds': lower if include_confidence else [],
        'upper_bounds': upper if include_confidence else [],
    }
    return result


def generate_detailed_forecast(res: Dict[str, Any], group_eng: pd.DataFrame, product: str, 
                              forecast_weeks: int, forecast_method: str,
                              include_confidence: bool, confidence_level: float) -> Dict[str, Any]:
    """Generate detailed forecasts with improved feature recalculation and confidence intervals.
    
    Args:
        res: Model bundle containing trained models and preprocessing components
        group_eng: Historical data for the specific region-product group
        product: Product name (e.g., 'HOBC')
        forecast_weeks: Number of weeks to forecast
        forecast_method: 'Direct' or 'Recursive' forecasting strategy
        include_confidence: Whether to include confidence intervals
        confidence_level: Confidence level for prediction intervals (0-1)
        
    Returns:
        Dictionary with forecast dates, values, and confidence intervals
    """
    # Extract model components - handle nested structure from region training
    if 'results' in res:
        # This is the structure from region training
        model_bundle = res['results']
        models = model_bundle.get('models', {})
        lgbm = models.get('lgbm')
        rf = models.get('rf')
        metrics = res.get('metrics', {})
        test_data = model_bundle.get('test_data', {})
        y_test = test_data.get('y_test', pd.Series(dtype=float)) if test_data else pd.Series(dtype=float)
        imputer = model_bundle.get('imputer')
        selector = model_bundle.get('selector')
        feature_cols = model_bundle.get('feature_cols', [])
        selected_features = model_bundle.get('selected_features', [])
        ensemble_method = model_bundle.get('ensemble_method', 'Average')
        lgbm_weight = model_bundle.get('lgbm_weight', 0.5)
    else:
        # This is the direct structure (for overall models)
        models = res.get('models', {})
        lgbm = models.get('lgbm')
        rf = models.get('rf')
        metrics = res.get('metrics', {})
        test_data = res.get('test_data', {})
        y_test = test_data.get('y_test', pd.Series(dtype=float)) if test_data else pd.Series(dtype=float)
        imputer = res.get('imputer')
        selector = res.get('selector')
        feature_cols = res.get('feature_cols', [])
        selected_features = res.get('selected_features', [])
        ensemble_method = res.get('ensemble_method', 'Average')
        lgbm_weight = res.get('lgbm_weight', 0.5)

    # Check if we have all required components
    if not feature_cols or lgbm is None or rf is None or imputer is None:
        # Try to get feature_cols from the group_eng data if not available
        if not feature_cols:
            feature_cols = [col for col in group_eng.columns 
                          if col not in ['Region', 'Product', 'week_start', 'sales_volume', 'sales_amount']]
        
        # Final check
        if not feature_cols or lgbm is None or rf is None or imputer is None:
            st.warning(f"Missing required model components for {product}. Models: {lgbm is not None}, RF: {rf is not None}, Imputer: {imputer is not None}, Features: {len(feature_cols) if feature_cols else 0}")
            return {
                'dates': [],
                'dates_str': [],
                'values': [],
                'lower_bounds': [],
                'upper_bounds': [],
                'method': forecast_method,
                'horizon': forecast_weeks,
                'confidence_level': confidence_level if include_confidence else None,
            }

    # Check if feature selection was enabled during training
    feature_selection_enabled = selector is not None

    group_eng = group_eng.copy()
    if not pd.api.types.is_datetime64_any_dtype(group_eng['week_start']):
        group_eng['week_start'] = pd.to_datetime(group_eng['week_start'])

    # CRITICAL: Create ALL features using the original feature_cols, just like in training
    group_eng = enhanced_feature_engineering(group_eng, forecasting_mode=True, random_state=None)
    
    # Create missing features for the original feature set (not the selected subset)
    group_eng = create_missing_features(group_eng, feature_cols)

    last_data = group_eng.iloc[-1:].copy()
    last_date = pd.to_datetime(last_data['week_start'].iloc[0])

    forecast_dates = [last_date + timedelta(days=7 * i) for i in range(1, forecast_weeks + 1)]
    forecast_values = []
    lower_bounds = []
    upper_bounds = []

    if forecast_method == 'Direct':
        # Create a copy of the historical data to use for feature engineering
        forecast_history = group_eng.copy()
        
        for i in range(forecast_weeks):
            # Create a new row for the forecast date
            future_data = last_data.copy()
            future_data['week_start'] = forecast_dates[i]
            future_data['week_start'] = pd.to_datetime(future_data['week_start'])
            
            # Add the future data row to the history for proper feature calculation
            temp_history = pd.concat([forecast_history, future_data], ignore_index=True)
            
            # Apply enhanced feature engineering with forecasting_mode=True for realistic external factors
            temp_history = enhanced_feature_engineering(temp_history, forecasting_mode=True)
            temp_history = create_missing_features(temp_history, feature_cols)
            
            # Get the last row which contains our forecast features
            future_data = temp_history.iloc[-1:].copy()
            
            # No additional product-specific adjustments - rely on deterministic features only
            # The enhanced_feature_engineering with forecasting_mode=True already provides
            # stable, deterministic features without noise

            # Use the same logic as the original: extract features using feature_cols, then apply preprocessing
            X_future = future_data[feature_cols]
            X_future_clean = pd.DataFrame(imputer.transform(X_future), columns=feature_cols, index=X_future.index)
            
            # Apply feature selection only if it was enabled during training
            if feature_selection_enabled and selector is not None:
                X_future_sel = selector.transform(X_future_clean)
                if isinstance(X_future_sel, np.ndarray):
                    if X_future_sel.ndim == 1:
                        X_future_sel = X_future_sel.reshape(1, -1)
                    elif X_future_sel.ndim > 2:
                        X_future_sel = X_future_sel.reshape(X_future_sel.shape[0], -1)
            else:
                X_future_sel = X_future_clean.values
            
            # Handle any remaining NaN values
            X_future_sel = np.nan_to_num(X_future_sel, nan=0.0, posinf=0.0, neginf=0.0)
                
            # Ensure input is 2D for model prediction
            if isinstance(X_future_sel, np.ndarray):
                if X_future_sel.ndim == 1:
                    X_future_sel = X_future_sel.reshape(1, -1)
                elif X_future_sel.ndim > 2:
                    X_future_sel = X_future_sel.reshape(X_future_sel.shape[0], -1)

            # Make predictions
            pred_lgbm = lgbm.predict(X_future_sel)
            pred_rf = rf.predict(X_future_sel)
            
            # Combine predictions
            if ensemble_method == 'Weighted Average':
                forecast = lgbm_weight * pred_lgbm + (1 - lgbm_weight) * pred_rf
            else:
                forecast = (pred_lgbm + pred_rf) / 2
                
            forecast_values.append(forecast[0])
            
            # Calculate confidence intervals
            if include_confidence:
                z_lookup = {0.99: 2.58, 0.95: 1.96, 0.9: 1.645, 0.8: 1.28}
                z = z_lookup.get(confidence_level, 1.96)
                
                # Use RMSE from metrics for error estimation
                error_estimate = metrics.get('RMSE', 0) if metrics.get('RMSE', 0) > 0 else historical_volatility * y_test.mean() if len(y_test) > 0 else 1000
                
                # Increase uncertainty for longer forecast horizons
                step_factor = 1 + (i * 0.1)
                margin = z * error_estimate * step_factor
                
                lower_bounds.append(max(0, forecast[0] - margin))
                upper_bounds.append(forecast[0] + margin)
    
    else:  # Recursive forecasting
        # Use the generate_forecast function for recursive forecasting
        forecast_result = generate_forecast(
            {'models': {'lgbm': lgbm, 'rf': rf}, 'imputer': imputer, 'selector': selector,
             'feature_cols': feature_cols, 'ensemble_method': ensemble_method, 'lgbm_weight': lgbm_weight,
             'residual_std': metrics.get('RMSE', 0), 'test_data': {'y_test': y_test, 'y_pred': pd.Series()}},
            group_eng, forecast_weeks, include_confidence, confidence_level
        )
        
        forecast_values = forecast_result.get('values', [])
        lower_bounds = forecast_result.get('lower_bounds', [])
        upper_bounds = forecast_result.get('upper_bounds', [])

    # Return results
    return {
        'dates': forecast_dates,
        'dates_str': [d.strftime('%Y-%m-%d') for d in forecast_dates],
        'values': forecast_values,
        'lower_bounds': lower_bounds if include_confidence else [],
        'upper_bounds': upper_bounds if include_confidence else [],
        'method': forecast_method,
        'horizon': forecast_weeks,
        'confidence_level': confidence_level if include_confidence else None,
    }


# Session State Management Functions
def initialize_session_state():
    """Initialize all session state variables with default values."""
    if "model_params" not in st.session_state:
        st.session_state.model_params = {
            "overall": {
                "train_ratio": 0.8,
                "split_method": "Time-based",
                "feature_selection": True,
                "k_features": 20,
                "lgbm_learning_rate": 0.01,
                "lgbm_n_estimators": 1000,
                "lgbm_max_depth": 7,
                "rf_n_estimators": 200,
                "rf_max_depth": 10,
                "rf_min_samples_split": 2,
                "ensemble_method": "Average",
                "lgbm_weight": 0.5
            },
            "region_fuel": {
                "train_ratio": 0.8,
                "split_method": "Time-based",
                "feature_selection": True,
                "k_features": 15,
                "lgbm_learning_rate": 0.05,
                "lgbm_n_estimators": 500,
                "lgbm_max_depth": 5,
                "rf_n_estimators": 100,
                "rf_max_depth": 8,
                "rf_min_samples_split": 5,
                "ensemble_method": "Weighted Average",
                "lgbm_weight": 0.6
            }
        }
    
    if "forecast_params" not in st.session_state:
        st.session_state.forecast_params = {
            "forecast_weeks": 12,
            "forecast_method": "Recursive",
            "include_confidence": True,
            "confidence_level": 0.95
        }
    
    # Initialize other session state variables
    session_vars = [
        "run_overall", "run_region_fuel", "data_loaded", "default_data_loaded",
        "overall_metrics", "overall_y_test", "overall_y_pred", "overall_test_df",
        "rp_results", "summary_df", "model_selection", "overall_results"
    ]
    
    for var in session_vars:
        if var not in st.session_state:
            if var in ["run_overall", "run_region_fuel", "data_loaded", "default_data_loaded"]:
                st.session_state[var] = False
            elif var in ["rp_results"]:
                st.session_state[var] = {}
            elif var in ["model_selection"]:
                st.session_state[var] = "overall"
            else:
                st.session_state[var] = None


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get feature columns from a DataFrame, excluding target and metadata columns.
    
    Args:
        df: DataFrame to extract feature columns from
    
    Returns:
        List of feature column names
    """
    excluded_cols = ['Region', 'Product', 'week_start', 'sales_volume', 'sales_amount']
    return [col for col in df.columns if col not in excluded_cols]