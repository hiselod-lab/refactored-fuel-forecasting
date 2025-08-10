import numpy as np
import pandas as pd
from core import (
    enhanced_feature_engineering,
    train_models,
    generate_detailed_forecast,
    get_feature_columns,
)
from helpers import ModelConfig


def test_synthetic_forecast_variation(synthetic_df):
    train_df = synthetic_df.iloc[:80]
    future_df = synthetic_df.iloc[80:92]

    eng = enhanced_feature_engineering(train_df, random_state=0)
    feature_cols = get_feature_columns(eng)

    config = ModelConfig(
        train_ratio=0.8,
        split_method='Time-based',
        feature_selection=False,
        lgbm_n_estimators=50,
        rf_n_estimators=50,
        lgbm_learning_rate=0.1,
        lgbm_max_depth=7,
        rf_max_depth=7,
    )

    res = train_models(eng, feature_cols, config)

    forecast_rec = generate_detailed_forecast(
        res, eng, 'X', 12, 'Recursive', False, 0.95
    )
    forecast_dir = generate_detailed_forecast(
        res, eng, 'X', 12, 'Direct', False, 0.95
    )

    for fc in [forecast_rec, forecast_dir]:
        vals = np.array(fc['values'])
        assert np.var(vals) > 0
        true = future_df['sales_volume'].values[: len(vals)]
        history_pattern = train_df['sales_volume'].iloc[-26:].values[: len(vals)]
        corr = np.corrcoef(history_pattern, vals)[0, 1]
        assert corr > 0.25
        mape = np.mean(np.abs(true - vals) / true)
        assert mape < 0.2
