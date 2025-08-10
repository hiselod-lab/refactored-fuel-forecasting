import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import enhanced_feature_engineering


def test_no_static_fallback():
    df = pd.DataFrame({
        'week_start': pd.date_range('2020-01-06', periods=5, freq='W-MON'),
        'Region': 'R1',
        'Product': 'P1',
        'sales_volume': [10, 12, 11, 13, 12],
    })
    eng = enhanced_feature_engineering(df, forecasting_mode=True)
    numeric_df = eng.select_dtypes(include=['number'])
    assert not ((numeric_df == 100.0).any().any())
    assert not ((numeric_df == 1000.0).any().any())
