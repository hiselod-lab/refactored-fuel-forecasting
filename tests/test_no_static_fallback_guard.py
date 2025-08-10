import pandas as pd
import numpy as np
from core import enhanced_feature_engineering, create_missing_features


def test_no_static_fallback_guard():
    df = pd.DataFrame({
        'week_start': pd.date_range('2020-01-06', periods=3, freq='W-MON'),
        'sales_volume': [100, 110, 120],
        'Region': ['A'] * 3,
        'Product': ['X'] * 3,
    })
    eng = enhanced_feature_engineering(df.iloc[:2], forecasting_mode=True)
    eng = create_missing_features(eng, ['lag_1', 'lag_2', 'avg_price'])
    assert not ((eng.fillna(-1) == 1000).any().any())
    assert eng['avg_price'].isna().all()
