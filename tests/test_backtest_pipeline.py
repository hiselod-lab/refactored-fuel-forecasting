import numpy as np
from core import (
    enhanced_feature_engineering,
    train_models,
    generate_detailed_forecast,
    get_feature_columns,
)
from helpers import ModelConfig


def _backtest(df, method):
    horizon = 8
    metrics = []
    for start in range(52, 84, horizon):
        train = df.iloc[:start]
        test = df.iloc[start:start + horizon]
        eng = enhanced_feature_engineering(train, random_state=0)
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
        forecast = generate_detailed_forecast(
            res, eng, 'X', horizon, method, False, 0.95
        )
        true = test['sales_volume'].values[:horizon]
        pred = np.array(forecast['values'])
        mae = np.mean(np.abs(true - pred))
        metrics.append(mae)
    return metrics


def test_backtest_parity(synthetic_df):
    m_rec = _backtest(synthetic_df, 'Recursive')
    m_dir = _backtest(synthetic_df, 'Direct')
    print('recursive', m_rec)
    print('direct', m_dir)
    for a, b in zip(m_rec, m_dir):
        diff = abs(a - b) / max(a, b)
        assert diff <= 0.5
