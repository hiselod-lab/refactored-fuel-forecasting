import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from core import enhanced_feature_engineering, create_missing_features


def test_no_static_fallbacks():
    df = pd.DataFrame({
        'week_start': pd.date_range('2024-01-01', periods=5, freq='W'),
        'Region': 'A',
        'Product': 'X',
        'sales_volume': np.linspace(100, 140, 5),
    })
    eng = enhanced_feature_engineering(df, forecasting_mode=True, random_state=None)
    eng = create_missing_features(eng[['week_start', 'Region', 'Product', 'sales_volume']],
                                  ['lag_1', 'avg_price', 'price_elasticity'])
    forbidden = {1000.0, 100000.0, 0.1, 0.05, 0.01}
    cols = ['lag_1', 'avg_price', 'price_elasticity']
    for col in cols:
        assert not eng[col].isin(forbidden).any()
