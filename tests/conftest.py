import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Ensure project root on path for imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

@pytest.fixture
def synthetic_df():
    rng = pd.date_range('2020-01-06', periods=92, freq='W-MON')
    trend = np.linspace(100, 200, len(rng))
    season = 20 * np.sin(2 * np.pi * np.arange(len(rng)) / 52)
    noise = np.random.RandomState(0).normal(0, 1, len(rng))
    price = 10 + 0.5 * np.cos(2 * np.pi * np.arange(len(rng)) / 26)
    sales = trend + season + noise
    df = pd.DataFrame({
        'week_start': rng,
        'sales_volume': sales,
        'avg_price': price,
        'Region': ['A'] * len(rng),
        'Product': ['X'] * len(rng)
    })
    return df
