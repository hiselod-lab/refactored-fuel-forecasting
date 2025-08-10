"""
Data Overview Module
Handles the complete Data Overview tab functionality for the fuel forecasting app.
"""

import streamlit as st
import pandas as pd
import numpy as np
from helpers import (
    parse_week_start,
    prepare_df_for_display,
    create_region_volume_chart,
    create_product_volume_chart,
    create_product_chart,
    create_region_product_chart,
    create_monthly_sales_chart,
    create_price_trend_chart,
    format_value_with_unit
)


def show_data_overview(weekly_feats: pd.DataFrame) -> None:
    """
    Render the complete Data Overview tab with all sections.
    
    Args:
        weekly_feats: The main dataframe with fuel sales data
    """
    st.markdown('<h2 class="section-header">📊 Data Overview & Market Analysis</h2>', unsafe_allow_html=True)
    
    # Prepare overview data
    df_overview = weekly_feats.copy()
    # Ensure datetime parsing
    if not pd.api.types.is_datetime64_any_dtype(df_overview['week_start']):
        df_overview['week_start'] = pd.to_datetime(df_overview['week_start'])

    # Business Performance Dashboard
    _render_business_dashboard(df_overview)
    
    # Price Analysis
    _render_price_analysis(df_overview)
    
    # Market Performance Analysis
    _render_market_analysis(df_overview)
    
    # Monthly Trends
    _render_monthly_trends(df_overview)
    
    # Price Trends
    _render_price_trends(df_overview)
    
    # Data Preview
    _render_data_preview(weekly_feats)


def _render_business_dashboard(df_overview: pd.DataFrame) -> None:
    """Render the Business Performance Dashboard section."""
    st.markdown("### 📊 Business Performance Dashboard")
    
    # Calculate meaningful business metrics
    region_performance = df_overview.groupby('Region')['sales_volume'].sum().sort_values(ascending=False)
    product_performance = df_overview.groupby('Product')['sales_volume'].sum().sort_values(ascending=False)
    
    # Price volatility and trends
    latest_month = df_overview['week_start'].max()
    recent_data = df_overview[df_overview['week_start'] >= (latest_month - pd.Timedelta(days=90))]
    
    # Regional dominance
    top_region = region_performance.index[0]
    top_region_share = (region_performance.iloc[0] / region_performance.sum() * 100).round(1)
    
    # Product dominance  
    top_product = product_performance.index[0]
    top_product_share = (product_performance.iloc[0] / product_performance.sum() * 100).round(1)
    
    # Market concentration
    regions_count = df_overview['Region'].nunique()
    products_count = df_overview['Product'].nunique()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Market Leader (Region)", 
            f"{top_region}",
            f"{top_region_share}% market share",
            help=f"Dominant region with {format_value_with_unit(region_performance.iloc[0])} total volume"
        )
    with col2:
        st.metric(
            "Top Product", 
            f"{top_product}",
            f"{top_product_share}% of total sales",
            help=f"Leading product with {format_value_with_unit(product_performance.iloc[0])} volume"
        )
    with col3:
        st.metric(
            "Market Coverage", 
            f"{regions_count} Regions",
            f"{products_count} Fuel Types",
            help="Geographic and product diversification"
        )
    with col4:
        # Calculate fiscal years (Pakistan fiscal year: July-June)
        min_date = df_overview['week_start'].min()
        max_date = df_overview['week_start'].max()
        
        # Fiscal year calculation: if month >= 7, FY = year+1, else FY = year (July-June cycle)
        min_fy = min_date.year + 1 if min_date.month >= 7 else min_date.year
        max_fy = max_date.year + 1 if max_date.month >= 7 else max_date.year
        
        st.metric(
            "Fiscal Period", 
            f"FY {min_fy}-FY {max_fy}",
            f"{len(df_overview)} weeks of data",
            help="Fiscal year coverage (July-June cycle)"
        )


def _render_price_analysis(df_overview: pd.DataFrame) -> None:
    """Render the Price & Profitability Analysis section."""
    st.markdown("### 💰 Price & Profitability Analysis")
    price_stats = df_overview.groupby('Product').agg({
        'avg_price': ['mean', 'min', 'max'],
        'sales_volume': 'sum',
        'sales_amount': 'sum'
    }).round(2)
    
    # Format volume and sales with M/B suffixes
    price_stats.columns = ['Avg Price (₨)', 'Min Price (₨)', 'Max Price (₨)', 'Total Volume (Raw)', 'Total Sales (Raw)']
    
    # Apply formatting
    price_stats['Total Volume (Litres)'] = price_stats['Total Volume (Raw)'].apply(
        lambda x: f"{x/1e9:.2f}B" if x >= 1e9 else f"{x/1e6:.2f}M"
    )
    price_stats['Total Sales'] = price_stats['Total Sales (Raw)'].apply(
        lambda x: f"₨{x/1e9:.2f}B" if x >= 1e9 else f"₨{x/1e6:.2f}M"
    )
    
    # Select final columns for display
    display_stats = price_stats[['Avg Price (₨)', 'Min Price (₨)', 'Max Price (₨)', 'Total Volume (Litres)', 'Total Sales']]
    st.dataframe(prepare_df_for_display(display_stats), use_container_width=True)


def _render_market_analysis(df_overview: pd.DataFrame) -> None:
    """Render the Market Performance Analysis section."""
    st.markdown("### 🏆 Market Performance Analysis")
    
    # Create a more comprehensive market analysis with visual focus
    col1, col2 = st.columns([2, 1])
    
    with col1:
        tab_region, tab_product, tab_region_product = st.tabs(["Regional Performance", "Product Performance", "Market Matrix"])
        
        with tab_region:
            fig_region = create_region_volume_chart(df_overview)
            st.plotly_chart(fig_region, use_container_width=True, key="region_chart_tab")
        
        with tab_product:
            show_lagged_product = st.toggle("Show logarithmic scale", key="product_lagged", help="Toggle to view data in logarithmic scale")
            product_df = create_product_volume_chart(df_overview)
            fig_product = create_product_chart(product_df, log_y=show_lagged_product)
            st.plotly_chart(fig_product, use_container_width=True, key="product_chart_tab")
        
        with tab_region_product:
            show_lagged_rp = st.toggle("Show logarithmic scale", key="rp_lagged", help="Toggle to view data in logarithmic scale")
            fig_rp = create_region_product_chart(df_overview, log_y=show_lagged_rp)
            st.plotly_chart(fig_rp, use_container_width=True, key="region_product_chart_tab")
    
    with col2:
        _render_strategic_intelligence(df_overview)


def _render_strategic_intelligence(df_overview: pd.DataFrame) -> None:
    """Render the Strategic Intelligence sidebar."""
    st.markdown("#### 🎯 Strategic Intelligence")
    
    # Advanced analytics for business insights
    df_overview = df_overview.copy()  # Avoid modifying original
    df_overview['year'] = df_overview['week_start'].dt.year
    df_overview['month'] = df_overview['week_start'].dt.month
    
    # Growth Leaders Analysis
    st.markdown("**🚀 Growth Champions**")
    latest_year = df_overview['year'].max()
    if latest_year > df_overview['year'].min():
        # Calculate year-over-year growth by region
        current_year = df_overview[df_overview['year'] == latest_year].groupby('Region')['sales_volume'].sum()
        prev_year = df_overview[df_overview['year'] == latest_year-1].groupby('Region')['sales_volume'].sum()
        
        region_growth = ((current_year - prev_year) / prev_year * 100).fillna(0).round(1)
        top_growth_region = region_growth.idxmax() if not region_growth.empty else "N/A"
        top_growth_rate = region_growth.max() if not region_growth.empty else 0
        
        st.write(f"📈 **{top_growth_region}**: {top_growth_rate:+.1f}% YoY growth")
        st.caption("Fastest growing region")
    
    # Seasonality Patterns
    st.markdown("**🌊 Seasonal Insights**")
    monthly_sales = df_overview.groupby('month')['sales_volume'].sum()
    peak_month = monthly_sales.idxmax()
    low_month = monthly_sales.idxmin()
    
    month_names = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                  7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
    
    st.write(f"🔥 **{month_names.get(peak_month, peak_month)}** is peak month")
    st.write(f"❄️ **{month_names.get(low_month, low_month)}** is lowest month")
    
    # Market Concentration
    st.markdown("**⚖️ Market Balance**")
    region_share = df_overview.groupby('Region')['sales_volume'].sum()
    region_pct = (region_share / region_share.sum() * 100).round(1)
    top2_share = region_pct.nlargest(2).sum()
    
    # HHI calculation
    hhi = sum((pct/100)**2 for pct in region_pct)
    if hhi > 0.25:
        concentration = "🔴 High"
    elif hhi > 0.15:
        concentration = "🟡 Moderate"  
    else:
        concentration = "🟢 Low"
        
    st.write(f"{concentration} concentration")
    st.caption(f"Top 2 regions: {top2_share:.0f}% share")
    
    # Product Mix Evolution
    st.markdown("**🔄 Mix Dynamics**")
    product_share = df_overview.groupby('Product')['sales_volume'].sum()
    product_pct = (product_share / product_share.sum() * 100).round(1)
    
    # Check if HSD (diesel) is significant
    hsd_share = product_pct.get('HSD', 0)
    if hsd_share > 30:
        st.write(f"🚛 **Diesel dominance**: {hsd_share:.0f}% share")
    else:
        st.write(f"⛽ **Petrol focus**: {product_pct.get('PMG', 0):.0f}% share")
    
    premium_share = product_pct.get('HOBC', 0)
    if premium_share > 0.5:
        st.write(f"💎 **Premium segment**: {premium_share:.1f}% niche")
    else:
        st.write(f"💎 **Ultra-niche**: HOBC <0.5% share")


def _render_monthly_trends(df_overview: pd.DataFrame) -> None:
    """Render the Monthly Sales Volume Trend section."""
    st.markdown("### 📈 Monthly Sales Volume Trend")
    show_lagged_month = st.toggle("Show logarithmic scale", key="monthly_lagged", help="Toggle to view data in logarithmic scale")
    fig_monthly = create_monthly_sales_chart(df_overview, log_y=show_lagged_month)
    st.plotly_chart(fig_monthly, use_container_width=True, key="monthly_chart")


def _render_price_trends(df_overview: pd.DataFrame) -> None:
    """Render the Price Trends over Time section."""
    st.markdown("### 💰 Price Trends over Time")
    fig_price = create_price_trend_chart(df_overview)
    st.plotly_chart(fig_price, use_container_width=True, key="price_chart")


def _render_data_preview(weekly_feats: pd.DataFrame) -> None:
    """Render the Data Preview section."""
    # Data preview
    with st.expander("📋 Detailed Data Preview", expanded=False):
        st.dataframe(prepare_df_for_display(weekly_feats.head(20)))
        
        # Data statistics
        st.markdown("### 📊 Statistical Summary")
        st.dataframe(prepare_df_for_display(weekly_feats.describe()))
        
        # Missing values
        st.markdown("### 🔍 Missing Values Analysis")
        missing_data = weekly_feats.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing Percentage': (missing_data.values / len(weekly_feats)) * 100
        })
        st.dataframe(prepare_df_for_display(missing_df))
