import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from core import enhanced_feature_engineering, train_models, generate_detailed_forecast
from helpers import ModelConfig


def _make_series(n_weeks: int = 116) -> pd.DataFrame:
    rng = pd.date_range('2020-01-06', periods=n_weeks, freq='W-MON')
    trend = np.arange(n_weeks) * 5
    seasonal = 100 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
    price = 5 + 0.5 * np.sin(2 * np.pi * np.arange(n_weeks) / 26)
    noise = np.random.normal(0, 10, n_weeks)
    sales = 500 + trend + seasonal + noise
    return pd.DataFrame({
        'week_start': rng,
        'Region': 'A',
        'Product': 'X',
        'sales_volume': sales,
        'avg_price': price,
    })


def test_synthetic_forecast_variance_and_accuracy():
    df = _make_series()
    train_df = df.iloc[:104]
    test_df = df.iloc[104:116]

    eng = enhanced_feature_engineering(train_df, forecasting_mode=False, random_state=42)
    feature_cols = [c for c in eng.columns if c not in ['Region', 'Product', 'week_start', 'sales_volume', 'sales_amount']]

    config = ModelConfig(
        train_ratio=0.8,
        split_method='Time-based',
        feature_selection=False,
        lgbm_learning_rate=0.1,
        lgbm_n_estimators=50,
        lgbm_max_depth=5,
        rf_n_estimators=50,
        rf_max_depth=5,
    )

    res = train_models(eng, feature_cols, config)

    rec = generate_detailed_forecast(res, eng, 'X', 12, 'Recursive', False, 0.95)
    direct = generate_detailed_forecast(res, eng, 'X', 12, 'Direct', False, 0.95)

    rec_vals = np.array(rec['values'])
    dir_vals = np.array(direct['values'])

    assert rec_vals.var() > 1e-3
    assert dir_vals.var() > 1e-3

    actual = test_df['sales_volume'].values
    corr = np.corrcoef(actual, rec_vals)[0, 1]
    assert abs(corr) > 0.2
    mape_rec = np.mean(np.abs(actual - rec_vals) / actual)
    mape_dir = np.mean(np.abs(actual - dir_vals) / actual)
    assert mape_rec < 0.3
    assert mape_dir < 0.3
