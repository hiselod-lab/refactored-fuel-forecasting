import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import enhanced_feature_engineering, train_models, generate_detailed_forecast, get_feature_columns
from helpers import ModelConfig


def generate_synthetic_data(periods=80, seed=1):
    rng = np.random.default_rng(seed)
    weeks = pd.date_range('2020-01-06', periods=periods, freq='W-MON')
    t = np.arange(periods)
    trend = 1000 + 5 * t
    seasonality = 100 * np.sin(2 * np.pi * t / 52)
    price = 50 + 5 * np.sin(2 * np.pi * t / 13) + rng.normal(0, 1, periods)
    sales = trend + seasonality - 0.1 * price + rng.normal(0, 5, periods)
    return pd.DataFrame({
        'week_start': weeks,
        'Region': 'R1',
        'Product': 'P1',
        'sales_volume': sales,
        'avg_price': price,
    })


def test_backtest_pipeline():
    df = generate_synthetic_data()
    df_eng = enhanced_feature_engineering(df, forecasting_mode=False)
    df_eng = df_eng.dropna(subset=['lag_8']).reset_index(drop=True)
    feature_cols = get_feature_columns(df_eng)
    config = ModelConfig(feature_selection=False, lgbm_n_estimators=50, rf_n_estimators=50)

    tscv = TimeSeriesSplit(n_splits=4)
    maes_direct, maes_rec = [], []
    for i, (train_idx, test_idx) in enumerate(tscv.split(df_eng), 1):
        train_df = df_eng.iloc[train_idx]
        test_df = df_eng.iloc[test_idx]
        model = train_models(train_df, feature_cols, config)
        direct = generate_detailed_forecast(model, train_df, 'P1', len(test_df), 'Direct', False, 0.95)
        recursive = generate_detailed_forecast(model, train_df, 'P1', len(test_df), 'Recursive', False, 0.95)
        actual = test_df['sales_volume'].values
        mae_d = np.mean(np.abs(np.array(direct['values']) - actual))
        mae_r = np.mean(np.abs(np.array(recursive['values']) - actual))
        maes_direct.append(mae_d)
        maes_rec.append(mae_r)
        print(f"fold {i} direct MAE {mae_d:.1f} recursive MAE {mae_r:.1f}")
    for d, r in zip(maes_direct, maes_rec):
        assert abs(d - r) / ((d + r) / 2) <= 0.15
