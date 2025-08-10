import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import enhanced_feature_engineering, train_models, generate_detailed_forecast, get_feature_columns
from helpers import ModelConfig


def generate_synthetic_data(periods=64, seed=0):
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


def test_synthetic_forecasts():
    full = generate_synthetic_data(64)
    hist = full.iloc[:52]
    future_actual = full['sales_volume'].iloc[52:64].values

    hist_eng = enhanced_feature_engineering(hist, forecasting_mode=False)
    hist_eng = hist_eng.dropna(subset=['lag_8']).reset_index(drop=True)
    feature_cols = get_feature_columns(hist_eng)

    config = ModelConfig(feature_selection=False, lgbm_n_estimators=50, rf_n_estimators=50)
    model = train_models(hist_eng, feature_cols, config)

    direct = generate_detailed_forecast(model, hist_eng, 'P1', 12, 'Direct', False, 0.95)
    recursive = generate_detailed_forecast(model, hist_eng, 'P1', 12, 'Recursive', False, 0.95)

    direct_vals = np.array(direct['values'])
    rec_vals = np.array(recursive['values'])

    assert np.var(direct_vals) > 0
    assert np.var(rec_vals) > 0

    corr_direct = np.corrcoef(direct_vals, future_actual)[0, 1]
    corr_rec = np.corrcoef(rec_vals, future_actual)[0, 1]
    assert abs(corr_direct) > 0.4
    assert abs(corr_rec) > 0.4

    mape_direct = np.mean(np.abs(direct_vals - future_actual) / np.abs(future_actual))
    mape_rec = np.mean(np.abs(rec_vals - future_actual) / np.abs(future_actual))
    assert mape_direct < 0.2
    assert mape_rec < 0.2
