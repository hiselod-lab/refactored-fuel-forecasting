import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import TimeSeriesSplit


# Configure module-level logger
logger = logging.getLogger(__name__)
def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce mixed/dirty numeric strings to floats safely.
    - Removes commas, currency symbols, spaces, and non-numeric chars except '.', '-', 'e', '+'.
    - Returns a float Series with NaNs where conversion fails.
    """
    if series is None:
        return series
    as_str = series.astype(str).str.replace(r"[^0-9eE+\-\.]", "", regex=True)
    return pd.to_numeric(as_str, errors='coerce')



# Global color palette used across charts
PRODUCT_COLOR_MAP: Dict[str, str] = {
    'HSD': '#1f77b4',   # Dark blue
    'PMG': '#d62728',   # Dark red
    'HOBC': '#2ca02c',  # Dark green
}


def parse_week_start(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the 'week_start' column is datetime64[ns].

    Converts in place for efficiency and returns the DataFrame for chaining.
    """
    if df is None or 'week_start' not in df.columns:
        return df
    try:
        if not pd.api.types.is_datetime64_any_dtype(df['week_start']):
            df.loc[:, 'week_start'] = pd.to_datetime(df['week_start'].astype(str), errors='coerce', infer_datetime_format=True)
    except Exception as exc:
        logger.warning("Failed to parse week_start: %s", exc)
        df.loc[:, 'week_start'] = pd.to_datetime(df['week_start'].astype(str), errors='coerce', infer_datetime_format=True)
    return df


def format_value_with_unit(value) -> str:
    """Format numeric values with K/M/B suffixes. Accepts array-like and returns string."""
    if isinstance(value, (np.ndarray, list, tuple, pd.Series)):
        arr = np.asarray(value)
        value = float(arr.ravel()[0]) if arr.size > 0 else 0.0
    else:
        value = float(value)
    if value >= 1_000_000_000:
        return f"{value/1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value/1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value/1_000:.2f}K"
    return f"{value:.2f}"


def format_percentage(value) -> str:
    """Format decimal values as percentages (e.g., 0.057 -> 5.7%)."""
    if isinstance(value, (np.ndarray, list, tuple, pd.Series)):
        arr = np.asarray(value)
        value = float(arr.ravel()[0]) if arr.size > 0 else 0.0
    else:
        value = float(value)
    return f"{value * 100:.1f}%"


def format_metric_value(value, metric_type):
    """Format metric values appropriately based on their type."""
    if metric_type in ['SMAPE', 'smape']:
        return f"{value * 100:.1f}%"
    elif metric_type in ['R2', 'r2', 'R²']:
        return f"{value * 100:.1f}%"
    elif metric_type in ['MAE', 'mae', 'RMSE', 'rmse']:
        return format_large_number(value)
    else:
        return f"{value:.3f}"


def get_metric_info(metric_type):
    """Get infographic information about each metric."""
    info_dict = {
        'SMAPE': {
            'icon': '🎯',
            'description': 'Symmetric Mean Absolute Percentage Error',
            'explanation': 'Lower is better. Measures prediction accuracy as percentage.',
            'range': '0-100%'
        },
        'R2': {
            'icon': '📊',
            'description': 'Coefficient of Determination (R-squared)',
            'explanation': 'Higher is better. Shows how well model explains variance.',
            'range': '0-100%'
        },
        'R²': {
            'icon': '📊',
            'description': 'Coefficient of Determination (R-squared)',
            'explanation': 'Higher is better. Shows how well model explains variance.',
            'range': '0-100%'
        },
        'MAE': {
            'icon': '📏',
            'description': 'Mean Absolute Error',
            'explanation': 'Lower is better. Average absolute difference between actual and predicted.',
            'range': 'Litres'
        },
        'RMSE': {
            'icon': '📐',
            'description': 'Root Mean Square Error',
            'explanation': 'Lower is better. Penalizes larger errors more than MAE.',
            'range': 'Litres'
        }
    }
    return info_dict.get(metric_type, {
        'icon': '📈',
        'description': 'Performance Metric',
        'explanation': 'Model performance indicator.',
        'range': 'Various'
    })


def format_large_number(value):
    """Format large numbers with appropriate suffixes (K, M, B)."""
    if abs(value) >= 1e9:
        return f"{value / 1e9:.1f}B"
    elif abs(value) >= 1e6:
        return f"{value / 1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"{value / 1e3:.1f}K"
    else:
        return f"{value:.1f}"


def format_tick(value: float) -> str:
    """Format tick labels without decimals for large numbers."""
    if value >= 1_000_000_000:
        return f"{int(value/1_000_000_000)}B"
    if value >= 1_000_000:
        return f"{int(value/1_000_000)}M"
    if value >= 1_000:
        return f"{int(value/1_000)}K"
    return str(int(value))


def generate_log_ticks(max_val: float) -> List[float]:
    """Generate tick positions for log-scaled axes with custom spacing."""
    max_val = max(max_val, 2e8)
    tick_vals = [9e7, 1e8, 2e8]
    val = 3e8
    while val <= min(max_val, 9e8):
        tick_vals.append(val)
        val += 1e8
    val = 1e9
    while val <= max_val:
        tick_vals.append(val)
        val += 1e9
    return tick_vals


@st.cache_data(ttl=3600)
def plot_actual_vs_predicted(test_data: pd.DataFrame, y_test: pd.Series, y_pred, region: str, product: str):
    """Plot Actual vs Predicted for a specific region-product pair."""
    mask = (test_data['Region'] == region) & (test_data['Product'] == product)
    combo_test = test_data[mask]
    combo_y_test = y_test[mask]
    combo_y_pred = y_pred[mask]
    if len(combo_test) == 0:
        return None
    x_values = combo_test['week_start']
    actual_formatted = [format_value_with_unit(val) for val in combo_y_test]
    pred_formatted = [format_value_with_unit(val) for val in combo_y_pred]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values,
        y=combo_y_test,
        mode='lines+markers',
        name='Actual',
        line=dict(color='blue', width=2),
        opacity=0.9,
        hovertemplate="Date: %{x}<br>Actual: %{y:.2f} (%{customdata})<extra></extra>",
        customdata=actual_formatted
    ))
    fig.add_trace(go.Scatter(
        x=x_values,
        y=combo_y_pred,
        mode='lines+markers',
        name='Predicted',
        line=dict(color='red', width=2, dash='dash'),
        opacity=0.9,
        hovertemplate="Date: %{x}<br>Predicted: %{y:.2f} (%{customdata})<extra></extra>",
        customdata=pred_formatted
    ))
    fig.update_layout(
        title=f"{region} - {product}: Actual vs Predicted Weekly Sales",
        xaxis_title="Date",
        yaxis_title="Weekly Sales Volume (Litres)",
        height=400,
        template="plotly_white",
        showlegend=True
    )
    return fig


@st.cache_data(ttl=3600)
def create_region_volume_chart(df: pd.DataFrame):
    df = parse_week_start(df.copy())
    # Ensure required columns and numeric types
    if 'Region' not in df.columns or 'sales_volume' not in df.columns:
        return go.Figure()
    df.loc[:, 'sales_volume'] = _coerce_numeric(df['sales_volume'])
    df = df[df['sales_volume'].notna()]
    if df.empty:
        return go.Figure()
    region_volume = df.groupby('Region')['sales_volume'].sum().sort_values(ascending=False)
    region_df = pd.DataFrame({
        'Region': region_volume.index,
        'sales_volume': region_volume.values,
        'formatted_volume': [format_value_with_unit(val) for val in region_volume.values]
    })
    
    # Professional color scheme
    region_colors = {
        'South': '#1f77b4',    # Professional blue
        'North': '#ff7f0e',    # Professional orange
        'Central': '#2ca02c',  # Professional green
        'East': '#d62728',     # Professional red
        'West': '#9467bd'      # Professional purple
    }
    
    fig_region = px.bar(
        data_frame=region_df,
        x='Region',
        y='sales_volume',
        orientation='v',
        title='<b>Sales Volume by Region</b>',
        labels={'sales_volume': 'Sales Volume (Litres)', 'Region': 'Region'},
        color='Region',
        color_discrete_map=region_colors,
        custom_data='formatted_volume'
    )
    fig_region.update_traces(
        hovertemplate='<b>%{x}</b><br>Sales Volume: %{customdata}<br>Exact: %{y:,.0f} Litres<extra></extra>',
        marker_line_color='rgba(0,0,0,0.1)',
        marker_line_width=1
    )
    fig_region.update_layout(
        height=500,
        template='plotly_white',
        font=dict(size=12, family="Arial"),
        title_font_size=16,
        title_x=0.0,  # Left align title
        yaxis_title='Sales Volume (Litres)',
        yaxis_title_font_size=12,
        showlegend=False,
        margin=dict(l=60, r=40, t=80, b=40),
        plot_bgcolor='rgba(248,249,250,0.8)'
    )
    fig_region.update_xaxes(
        categoryorder='total descending',
        title_font_size=12,
        tickfont_size=11
    )
    # Create custom tick values and labels for billions
    max_val = region_df['sales_volume'].max()
    if max_val >= 1e9:
        # For billion-scale values, create custom ticks
        tick_step = max(1e8, (max_val // 5e8) * 1e8)  # Step size based on max value
        tick_vals = list(range(0, int(max_val + tick_step), int(tick_step)))
        tick_text = [f"{val/1e9:.0f}B" if val >= 1e9 else f"{val/1e6:.0f}M" if val >= 1e6 else f"{val/1e3:.0f}K" if val >= 1e3 else str(int(val)) for val in tick_vals]
        
        fig_region.update_yaxes(
            tickmode='array',
            tickvals=tick_vals,
            ticktext=tick_text,
            tickfont_size=11,
            gridcolor='rgba(200,200,200,0.3)'
        )
    else:
        fig_region.update_yaxes(
            tickformat='~s',
            tickfont_size=11,
            gridcolor='rgba(200,200,200,0.3)'
        )
    return fig_region


@st.cache_data(ttl=3600)
def create_product_volume_chart(df: pd.DataFrame) -> pd.DataFrame:
    if 'Product' not in df.columns or 'sales_volume' not in df.columns:
        return pd.DataFrame(columns=['Product', 'sales_volume', 'formatted_volume'])
    df = df.copy()
    df.loc[:, 'sales_volume'] = _coerce_numeric(df['sales_volume'])
    df = df[df['sales_volume'].notna()]
    product_volume = df.groupby('Product')['sales_volume'].sum().sort_values(ascending=False)
    product_df = pd.DataFrame({
        'Product': product_volume.index,
        'sales_volume': product_volume.values,
        'formatted_volume': [format_value_with_unit(val) for val in product_volume.values]
    })
    return product_df


@st.cache_data(ttl=3600)
def create_product_chart(product_df: pd.DataFrame, log_y: bool = False):
    product_df = product_df.sort_values('sales_volume', ascending=False)
    if product_df.empty:
        return go.Figure()
    fig_product = px.bar(
        data_frame=product_df,
        x='Product',
        y='sales_volume',
        orientation='v',
        title='Sales Volume by Product',
        labels={'sales_volume': 'Sales Volume (Litres)', 'Product': 'Fuel Type'},
        color='Product',
        color_discrete_map=PRODUCT_COLOR_MAP,
        custom_data='formatted_volume',
        log_y=log_y
    )
    fig_product.update_traces(
        hovertemplate='Fuel Type: %{x}<br>Sales Volume: %{y:.2f} (%{customdata})<extra></extra>'
    )
    fig_product.update_layout(
        height=500,
        template='plotly_white',
        font=dict(size=12),
        title_font_size=16,
        title_x=0.0,  # Left align title
        yaxis_title='Sales Volume (Litres)',
        yaxis_title_font_size=12,
        showlegend=False,
        margin=dict(l=60, r=40, t=80, b=40),
        plot_bgcolor='rgba(248,249,250,0.8)'  # Match regional performance background
    )
    if log_y:
        min_vals = product_df.groupby('Product')['sales_volume'].min()
        hobc_min = min_vals.get('HOBC', min_vals.min())
        overall_min = min(hobc_min * 0.5, 1e7)
        max_val = product_df['sales_volume'].max()
        tick_vals = generate_log_ticks(max_val)
        tick_text = [format_tick(v) + '        ' for v in tick_vals]
        axis_max = tick_vals[-1]
        log_min = np.log10(overall_min)
        # Create custom tick values with better spacing
        custom_ticks = []
        custom_labels = []
        
        # Start with the minimum visible value
        current_val = overall_min
        while current_val <= axis_max:
            custom_ticks.append(current_val)
            custom_labels.append(format_tick(current_val))
            current_val *= 2  # Double each time for exponential spacing
        
        fig_product.update_yaxes(
            tickmode='array',
            tickvals=custom_ticks,
            ticktext=custom_labels,
            range=[log_min, np.log10(axis_max)],
            tickfont=dict(size=11),
            tickangle=0,
            gridcolor='rgba(200,200,200,0.3)'
        )
    return fig_product


@st.cache_data(ttl=3600)
def create_region_product_chart(df: pd.DataFrame, log_y: bool = False):
    if 'Region' not in df.columns or 'Product' not in df.columns or 'sales_volume' not in df.columns:
        return go.Figure()
    df = df.copy()
    df.loc[:, 'sales_volume'] = _coerce_numeric(df['sales_volume'])
    df = df[df['sales_volume'].notna()]
    if df.empty:
        return go.Figure()
    region_totals = df.groupby('Region')['sales_volume'].sum().sort_values(ascending=False)
    region_order = region_totals.index.tolist()
    rp_data = df.groupby(['Region', 'Product'])['sales_volume'].sum().reset_index()
    rp_data['formatted_volume'] = rp_data['sales_volume'].apply(format_value_with_unit)
    rp_data['Region'] = pd.Categorical(rp_data['Region'], categories=region_order, ordered=True)
    rp_data = rp_data.sort_values('Region')
    fig_rp = px.bar(
        rp_data,
        x='Region',
        y='sales_volume',
        color='Product',
        barmode='group',
        orientation='v',
        title='Sales Volume by Region and Product',
        labels={'sales_volume': 'Sales Volume (Litres)', 'Region': 'Region', 'Product': 'Fuel Type'},
        custom_data='formatted_volume',
        log_y=log_y,
        color_discrete_map=PRODUCT_COLOR_MAP
    )
    fig_rp.update_traces(
        hovertemplate='Region: %{x}<br>Fuel Type: %{fullData.name}<br>Sales Volume: %{y:.2f} (%{customdata})<extra></extra>'
    )
    fig_rp.update_layout(
        height=500,
        template='plotly_white',
        yaxis_title='Sales Volume (Litres)',
        font=dict(size=12),
        title_font_size=16,
        title_x=0.0,  # Left align title
        margin=dict(l=60, r=40, t=80, b=40),
        plot_bgcolor='rgba(248,249,250,0.8)'  # Match regional performance background
    )
    if log_y:
        product_mins = rp_data.groupby('Product')['sales_volume'].min()
        hobc_min = product_mins.get('HOBC', product_mins.min())
        overall_min = min(hobc_min * 0.5, 1e7)
        max_val = rp_data['sales_volume'].max()
        tick_vals = generate_log_ticks(max_val)
        tick_text = [format_tick(v) + '        ' for v in tick_vals]
        axis_max = tick_vals[-1]
        log_min = np.log10(overall_min)
        fig_rp.update_yaxes(
            tickmode='array',
            tickvals=tick_vals[::5],  # Show every 5th tick for maximum spacing
            ticktext=[tick_text[i].strip() for i in range(0, len(tick_text), 5)],  # Remove extra spaces
            range=[log_min, np.log10(axis_max)],
            tickfont=dict(size=11),
            tickangle=0,
            gridcolor='rgba(200,200,200,0.3)'
        )
    return fig_rp


@st.cache_data(ttl=3600)
def create_monthly_sales_chart(df: pd.DataFrame, log_y: bool = False):
    df = parse_week_start(df.copy())
    # Defensive: ensure week_start is datetimelike
    if 'week_start' not in df.columns:
        raise ValueError("Missing required column 'week_start' for monthly chart")
    if not pd.api.types.is_datetime64_any_dtype(df['week_start']):
        try:
            coerced = pd.to_datetime(df['week_start'], errors='coerce')
            if not pd.api.types.is_datetime64_any_dtype(coerced):
                coerced = pd.to_datetime(df['week_start'].astype(str), errors='coerce')
            df.loc[:, 'week_start'] = coerced
        except Exception as exc:
            logger.error("Failed to coerce week_start to datetime: %s", exc)
            df.loc[:, 'week_start'] = pd.to_datetime(df['week_start'].astype(str), errors='coerce')
    # Drop rows that still failed coercion
    df = df[df['week_start'].notna()].copy()
    if df.empty:
        # Return an empty placeholder figure rather than failing
        return go.Figure()
    if 'Product' not in df.columns:
        raise ValueError("Missing required column 'Product' for monthly chart")
    # Create a stable month column and group by it
    if 'sales_volume' not in df.columns or 'Product' not in df.columns:
        return go.Figure()
    df.loc[:, 'sales_volume'] = _coerce_numeric(df['sales_volume'])
    df = df[df['sales_volume'].notna()]
    if df.empty:
        return go.Figure()
    df.loc[:, 'month'] = df['week_start'].dt.to_period('M')
    # Aggregate monthly by product and also overall to allow a global trend line
    df_monthly = df.groupby(['month', 'Product'], as_index=False)['sales_volume'].sum()
    df_monthly['month'] = df_monthly['month'].astype(str)
    df_monthly['formatted_volume'] = df_monthly['sales_volume'].apply(format_value_with_unit)
    fig_monthly = px.line(
        df_monthly,
        x='month',
        y='sales_volume',
        color='Product',
        title='',  # Remove title as requested
        labels={'month': 'Month', 'sales_volume': 'Monthly Sales Volume (Litres)', 'Product': 'Fuel Type'},
        custom_data='formatted_volume',
        log_y=log_y,
        color_discrete_map=PRODUCT_COLOR_MAP
    )
    fig_monthly.update_traces(
        mode='lines',  # Remove markers for cleaner trend visualization
        line=dict(width=2, smoothing=1.2),  # Thinner lines as requested
        hovertemplate='Month: %{x}<br>Fuel Type: %{fullData.name}<br>Monthly Sales Volume: %{y:.2f} (%{customdata})<extra></extra>'
    )
    fig_monthly.update_layout(
        height=500,
        template='plotly_white',
        font=dict(size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12)
        ),
        margin=dict(l=60, r=40, t=40, b=60),
        plot_bgcolor='rgba(248,249,250,0.8)',  # Match other charts
        showlegend=True,
        yaxis_title='Monthly Sales Volume (Litres)',
        xaxis_title='Month'
    )

    # Add overall monthly total as a subtle grey line
    df_total = df.copy()
    df_total['month_period'] = df_total['week_start'].dt.to_period('M')
    monthly_total = df_total.groupby('month_period')['sales_volume'].sum().reset_index()
    monthly_total['month'] = monthly_total['month_period'].astype(str)
    fig_monthly.add_trace(
        go.Scatter(
            x=monthly_total['month'],
            y=monthly_total['sales_volume'],
            mode='lines',
            name='All Fuels (Total)',
            line=dict(color='rgba(100,100,100,0.6)', width=2, dash='dash'),
            hovertemplate='Month: %{x}<br>Total Volume: %{y:,.0f} L<extra></extra>'
        )
    )

    # Add a linear trendline for the total series to aid interpretation
    try:
        numeric_x = np.arange(len(monthly_total))
        z = np.polyfit(numeric_x, monthly_total['sales_volume'].values, 1)
        p = np.poly1d(z)
        fig_monthly.add_trace(
            go.Scatter(
                x=monthly_total['month'],
                y=p(numeric_x),
                mode='lines',
                name='Trend',
                line=dict(color='rgba(50,50,50,0.9)', width=2),
                hoverinfo='skip'
            )
        )
    except Exception:
        # If trendline computation fails, skip silently
        pass
    if log_y:
        min_vals = df_monthly.groupby('Product')['sales_volume'].min()
        hobc_min = min_vals.get('HOBC', min_vals.min())
        overall_min = min(hobc_min * 0.1, 1e6)
        max_val = df_monthly['sales_volume'].max()
        base_ticks = [1e6, 5e6, 1e7, 5e7] + generate_log_ticks(max_val)
        tick_vals = sorted(list(set(base_ticks)))
        tick_text = [format_tick(v) + '                          ' for v in tick_vals]
        axis_max = tick_vals[-1]
        log_min = np.log10(overall_min)
        fig_monthly.update_yaxes(
            tickmode='array',
            tickvals=tick_vals[::4],  # Show every 4th tick for cleaner look
            ticktext=[tick_text[i].strip() for i in range(0, len(tick_text), 4)],
            range=[log_min, np.log10(axis_max)],
            tickfont=dict(size=11),
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(200, 200, 200, 0.3)'
        )
    fig_monthly.update_xaxes(
        tickangle=45,
        tickfont=dict(size=11),
        showgrid=True,
        gridcolor='rgba(200, 200, 200, 0.2)'
    )
    return fig_monthly


@st.cache_data(ttl=3600)
def create_price_trend_chart(df: pd.DataFrame):
    df = parse_week_start(df.copy())
    if 'week_start' not in df.columns:
        raise ValueError("Missing required column 'week_start' for price trend chart")
    if not pd.api.types.is_datetime64_any_dtype(df['week_start']):
        try:
            coerced = pd.to_datetime(df['week_start'], errors='coerce')
            if not pd.api.types.is_datetime64_any_dtype(coerced):
                coerced = pd.to_datetime(df['week_start'].astype(str), errors='coerce')
            df.loc[:, 'week_start'] = coerced
        except Exception as exc:
            logger.error("Failed to coerce week_start to datetime: %s", exc)
            df.loc[:, 'week_start'] = pd.to_datetime(df['week_start'].astype(str), errors='coerce')
    df = df[df['week_start'].notna()].copy()
    if df.empty:
        return go.Figure()
    if 'Product' not in df.columns:
        raise ValueError("Missing required column 'Product' for price trend chart")
    if 'avg_price' not in df.columns or 'Product' not in df.columns:
        return go.Figure()
    df.loc[:, 'avg_price'] = _coerce_numeric(df['avg_price'])
    df = df[df['avg_price'].notna()]
    if df.empty:
        return go.Figure()
    df.loc[:, 'month'] = df['week_start'].dt.to_period('M')
    df_price_trend = df.groupby(['month', 'Product'], as_index=False)['avg_price'].mean()
    df_price_trend['month'] = df_price_trend['month'].astype(str)
    def format_price(value: float) -> str:
        return f"{value:.2f} PKR"
    df_price_trend['formatted_price'] = df_price_trend['avg_price'].apply(format_price)
    fig_price = px.line(
        df_price_trend,
        x='month',
        y='avg_price',
        color='Product',
        title='Price Trends Over Time',
        labels={'month': 'Month', 'avg_price': 'Monthly Average Price (PKR)'},
        custom_data='formatted_price',
        color_discrete_map=PRODUCT_COLOR_MAP
    )
    fig_price.update_traces(
        hovertemplate='Month: %{x}<br>Product: %{fullData.name}<br>Price: %{y:.2f} PKR (%{customdata})<extra></extra>'
    )
    fig_price.update_layout(height=450, width=800, template='plotly_white')
    fig_price.update_xaxes(tickangle=45)
    return fig_price


def compute_smape(y_true: pd.Series, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom))


def validate_input_df(df: pd.DataFrame, required_cols: Optional[List[str]] = None) -> None:
    """Validate that required columns exist in the DataFrame."""
    default_required = ['Region', 'Product', 'week_start', 'sales_volume', 'avg_price']
    required = required_cols or default_required
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def prepare_forecast_features(
    history: pd.DataFrame,
    feature_cols: List[str],
    imputer,
    selector,
    enhanced_feature_engineering: Callable[[pd.DataFrame], pd.DataFrame],
    forecasting_mode: bool = True,
) -> np.ndarray:
    """Recalculate features on the provided history and return last-row feature matrix (2D)."""
    eng = enhanced_feature_engineering(history, forecasting_mode=forecasting_mode)
    features = eng[feature_cols].iloc[-1:]
    features_clean = pd.DataFrame(imputer.transform(features), columns=feature_cols, index=features.index)
    if selector is not None:
        features_sel = selector.transform(features_clean)
    else:
        features_sel = features_clean.values
    features_sel = np.asarray(features_sel)
    if features_sel.ndim == 1:
        features_sel = features_sel.reshape(1, -1)
    elif features_sel.ndim > 2:
        features_sel = features_sel.reshape(features_sel.shape[0], -1)
    features_sel = np.nan_to_num(features_sel, nan=0.0, posinf=0.0, neginf=0.0)
    return features_sel


def apply_volatility_adjustments(
    future_data: pd.DataFrame,
    product: str,
    metrics: Dict[str, float],
    y_test: pd.Series,
    step_index: int,
) -> pd.DataFrame:
    """Apply product-specific volatility and trend adjustments."""
    is_hobc = product == 'HOBC'
    if len(y_test) > 1:
        historical_trend = 1 if y_test.iloc[-1] > y_test.iloc[0] else -1
        trend_strength = min(1.0, abs(y_test.iloc[-1] - y_test.iloc[0]) / (y_test.mean() + 1e-10))
    else:
        historical_trend = 1
        trend_strength = 0.5
    if is_hobc:
        volatility_factor = 0.02 * (step_index + 1) * 3.0
        trend_factor = 0.03 * (step_index + 1) * 3.0
    else:
        volatility_factor = 0.02 * (step_index + 1)
        trend_factor = 0.03 * (step_index + 1)
    trend_bias = historical_trend * trend_strength * 0.01 * (step_index + 1)
    if 'price_volatility' in future_data.columns:
        future_data.loc[:, 'price_volatility'] *= (1 + np.random.normal(trend_bias, volatility_factor))
    if 'volume_trend' in future_data.columns:
        future_data.loc[:, 'volume_trend'] *= (1 + np.random.normal(trend_bias * 1.5, trend_factor))
    if 'volume_volatility' in future_data.columns:
        future_data.loc[:, 'volume_volatility'] *= (1 + np.random.normal(trend_bias, volatility_factor))
    return future_data


def compute_prediction_interval(
    forecast_value: float,
    metrics: Dict[str, float],
    step_index: int,
    confidence_level: float,
    is_hobc: bool = False,
) -> Tuple[float, float, float]:
    """Compute lower, upper bounds and margin for a forecast value."""
    mae = metrics.get('MAE', 0.0)
    rmse = metrics.get('RMSE', 0.0)
    error_estimate = (0.7 * mae) + (0.3 * rmse)
    z_lookup = {0.99: 2.58, 0.95: 1.96, 0.9: 1.645, 0.85: 1.44, 0.8: 1.28}
    z_score = z_lookup.get(confidence_level, 1.96)
    if is_hobc:
        step_factor = 1 + (step_index * 0.2)
        min_margin = 2000 * (step_index + 1)
    else:
        step_factor = 1 + (step_index * 0.1)
        min_margin = 0
    margin = max(z_score * error_estimate * step_factor, min_margin)
    lower = max(0.0, forecast_value - margin)
    upper = forecast_value + margin
    return lower, upper, margin


@dataclass
class ModelConfig:
    train_ratio: float = 0.8
    split_method: str = 'Time-based'
    feature_selection: bool = True
    k_features: int = 20
    lgbm_learning_rate: float = 0.01
    lgbm_n_estimators: int = 1000
    lgbm_max_depth: int = 7
    rf_n_estimators: int = 200
    rf_max_depth: int = 10
    rf_min_samples_split: int = 2
    ensemble_method: str = 'Average'
    lgbm_weight: float = 0.5


def time_series_cv_scores(
    X: pd.DataFrame,
    y: pd.Series,
    fit_predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    n_splits: int = 5,
) -> Dict[str, float]:
    """Evaluate a generic fit/predict function with TimeSeriesSplit and return averaged metrics."""
    from sklearn.metrics import mean_absolute_error, r2_score
    smapes, maes, rmses, r2s = [], [], [], []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        y_hat = fit_predict_fn(X_tr, y_tr, X_te)
        smapes.append(compute_smape(y_te, y_hat))
        maes.append(mean_absolute_error(y_te, y_hat))
        rmses.append(float(np.sqrt(np.mean((y_te - y_hat) ** 2))))
        r2s.append(r2_score(y_te, y_hat))
    return {
        'SMAPE': float(np.mean(smapes)),
        'MAE': float(np.mean(maes)),
        'RMSE': float(np.mean(rmses)),
        'R2': float(np.mean(r2s)),
    }


def prepare_df_for_display(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Move display utility here so all display conversions are centralized."""
    if df is None or (hasattr(df, 'empty') and df.empty):
        return df
    df_display = df.copy()
    for col in df_display.columns:
        if pd.api.types.is_datetime64_any_dtype(df_display[col]):
            df_display[col] = df_display[col].dt.strftime('%Y-%m-%d')
        elif df_display[col].dtype == 'object' and len(df_display) > 0:
            try:
                sample_values = df_display[col].dropna().head(10).tolist()
                has_timestamp = any(isinstance(x, (pd.Timestamp,)) for x in sample_values)
                if has_timestamp:
                    df_display[col] = df_display[col].apply(
                        lambda x: x.strftime('%Y-%m-%d') if isinstance(x, (pd.Timestamp,)) else str(x) if x is not None else x
                    )
            except Exception:
                try:
                    df_display[col] = df_display[col].apply(
                        lambda x: x.strftime('%Y-%m-%d') if isinstance(x, (pd.Timestamp,)) else x
                    )
                except Exception:
                    df_display[col] = df_display[col].astype(str)
    return df_display


