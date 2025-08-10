import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from core import enhanced_feature_engineering, train_models, generate_detailed_forecast
from helpers import ModelConfig


def _make_series(n_weeks: int = 140) -> pd.DataFrame:
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


def test_backtest_parity():
    df = _make_series()
    eng = enhanced_feature_engineering(df, forecasting_mode=False, random_state=42)
    feature_cols = [c for c in eng.columns if c not in ['Region', 'Product', 'week_start', 'sales_volume', 'sales_amount']]

    horizon = 8
    start_train = 80
    n_folds = 4
    metrics = {'Direct': [], 'Recursive': []}

    for method in ['Direct', 'Recursive']:
        for i in range(n_folds):
            train_end = start_train + i * horizon
            train_df = eng.iloc[:train_end]
            test_df = eng.iloc[train_end: train_end + horizon]

            config = ModelConfig(
                train_ratio=0.8,
                split_method='Time-based',
                feature_selection=False,
                lgbm_learning_rate=0.1,
                lgbm_n_estimators=30,
                lgbm_max_depth=5,
                rf_n_estimators=30,
                rf_max_depth=5,
            )

            res = train_models(train_df, feature_cols, config)
            forecast = generate_detailed_forecast(res, train_df, 'X', horizon, method, False, 0.95)
            pred = np.array(forecast['values'])
            actual = test_df['sales_volume'].values

            mae = np.mean(np.abs(actual - pred))
            rmse = np.sqrt(np.mean((actual - pred) ** 2))
            mape = np.mean(np.abs(actual - pred) / actual)
            print(method, i, mae, rmse, mape)
            metrics[method].append(mape)

    mean_direct = np.mean(metrics['Direct'])
    mean_recursive = np.mean(metrics['Recursive'])
    assert abs(mean_direct - mean_recursive) / mean_direct < 0.9
