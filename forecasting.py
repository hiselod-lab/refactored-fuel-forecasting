"""
Forecasting Module
Handles the forecasting functionality for the fuel forecasting app.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from helpers import (
    prepare_df_for_display,
    plot_actual_vs_predicted
)

def _convert_forecast_to_df(forecast_result, include_confidence=True):
    """Convert forecast result dictionary to DataFrame format expected by display functions."""
    if not forecast_result:
        return pd.DataFrame()
    
    # Create DataFrame from forecast results
    forecast_df = pd.DataFrame({
        'week_start': forecast_result.get('dates', []),
        'forecast': forecast_result.get('values', [])
    })
    
    # Add confidence intervals if available
    if include_confidence and forecast_result.get('lower_bounds') and forecast_result.get('upper_bounds'):
        forecast_df['lower_bound'] = forecast_result.get('lower_bounds', [])
        forecast_df['upper_bound'] = forecast_result.get('upper_bounds', [])
    
    return forecast_df


def show_forecasting_tab(weekly_feats: pd.DataFrame, generate_forecast, generate_detailed_forecast) -> None:
    """
    Render the complete Forecasting tab.
    
    Args:
        weekly_feats: The main dataframe with fuel sales data
        generate_forecast: Forecast generation function
        generate_detailed_forecast: Detailed forecast function
    """
    st.markdown('<h2 class="section-header">🔮 Sales Forecasting</h2>', unsafe_allow_html=True)
    
    # Check if models are trained
    if not _check_trained_models():
        st.warning("⚠️ Please train models first in the Model Training & Analysis tab before generating forecasts.")
        return
    
    # Forecasting configuration
    forecast_config = _render_forecast_configuration()
    
    # Generate forecasts based on configuration
    if st.button("🚀 Generate Forecasts", type="primary", key="generate_forecasts"):
        _run_forecasting(weekly_feats, forecast_config, generate_forecast, generate_detailed_forecast)


def _check_trained_models() -> bool:
    """Check if any models have been trained."""
    return ('overall_results' in st.session_state or 
            'rp_results' in st.session_state)


def _render_forecast_configuration() -> dict:
    """Render the forecast configuration form."""
    st.markdown("### ⚙️ Forecasting Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📅 Forecast Period")
        forecast_weeks = st.slider(
            "Number of weeks to forecast",
            min_value=1,
            max_value=52,
            value=12,
            help="How many weeks into the future to predict"
        )
        
        forecast_method = st.radio(
            "Forecasting Method",
            ["Recursive", "Direct"],
            index=0,
            help="Recursive: Use previous predictions | Direct: Independent predictions"
        )
    
    with col2:
        st.markdown("#### 🎯 Forecast Scope")
        model_type = st.radio(
            "Model to use for forecasting",
            ["Overall Model", "Region-Fuel Models", "Both"],
            index=2,
            help="Which trained models to use for predictions"
        )
        
        include_confidence = st.checkbox(
            "Include confidence intervals",
            value=True,
            help="Show prediction uncertainty bands"
        )
        
        if include_confidence:
            confidence_level = st.select_slider(
                "Confidence Level",
                options=[0.8, 0.85, 0.9, 0.95, 0.99],
                value=0.95,
                help="Width of confidence intervals"
            )
        else:
            confidence_level = 0.95
    
    # Advanced options
    with st.expander("🔧 Advanced Forecasting Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            adjust_seasonality = st.checkbox(
                "Apply seasonal adjustments",
                value=True,
                help="Adjust forecasts based on historical seasonal patterns"
            )
            
            adjust_trends = st.checkbox(
                "Apply trend adjustments",
                value=True,
                help="Incorporate long-term trend patterns"
            )
        
        with col2:
            volatility_adjustment = st.slider(
                "Volatility adjustment factor",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Adjust forecast volatility (1.0 = no adjustment)"
            )
            
            external_factors = st.checkbox(
                "Include external factors",
                value=True,
                help="Consider external economic/market factors"
            )
    
    return {
        'forecast_weeks': forecast_weeks,
        'forecast_method': forecast_method,
        'model_type': model_type,
        'include_confidence': include_confidence,
        'confidence_level': confidence_level,
        'adjust_seasonality': adjust_seasonality,
        'adjust_trends': adjust_trends,
        'volatility_adjustment': volatility_adjustment,
        'external_factors': external_factors
    }


def _run_forecasting(weekly_feats, config, generate_forecast, generate_detailed_forecast):
    """Execute the forecasting process based on configuration."""
    
    with st.spinner("🔮 Generating forecasts..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            forecasts = {}
            
            # Overall model forecasts
            if config['model_type'] in ['Overall Model', 'Both'] and 'overall_results' in st.session_state:
                status_text.text("🔮 Generating overall forecasts...")
                progress_bar.progress(20)
                
                overall_forecast_raw = generate_forecast(
                    st.session_state.overall_results,
                    weekly_feats,
                    config['forecast_weeks'],
                    config['include_confidence'],
                    config['confidence_level']
                )
                
                # Convert to expected format for display
                overall_forecast_df = _convert_forecast_to_df(overall_forecast_raw, config['include_confidence'])
                
                forecasts['overall'] = {
                    'forecast_data': overall_forecast_df,
                    'raw_forecast': overall_forecast_raw
                }
            
            # Region-fuel model forecasts
            if config['model_type'] in ['Region-Fuel Models', 'Both'] and 'rp_results' in st.session_state:
                status_text.text("🔮 Generating region-fuel forecasts...")
                progress_bar.progress(50)
                
                rp_forecasts = {}
                rp_results = st.session_state.rp_results
                
                for model_key, model_data in rp_results.items():
                    region = model_data['region']
                    product = model_data['product']
                    
                    try:
                        # Filter data for this region-product
                        rp_data = weekly_feats[
                            (weekly_feats['Region'] == region) & 
                            (weekly_feats['Product'] == product)
                        ].copy()
                        
                        if len(rp_data) > 0:
                             rp_forecast = generate_detailed_forecast(
                                 model_data['results'],
                                 rp_data,
                                 product,
                                 config['forecast_weeks'],
                                 config['forecast_method'],
                                 config['include_confidence'],
                                 config['confidence_level']
                             )
                             
                             # Convert forecast result to expected format
                             forecast_df = _convert_forecast_to_df(rp_forecast, config['include_confidence'])
                             
                             rp_forecasts[f"{region}_{product}"] = {
                                 'region': region,
                                 'product': product,
                                 'forecast': {
                                     'forecast_df': forecast_df,
                                     'raw_forecast': rp_forecast
                                 }
                             }
                        else:
                            st.warning(f"No data available for {region} - {product}")
                    
                    except Exception as e:
                        st.error(f"Error processing {region} - {product}: {str(e)}")
                        # Continue with other forecasts even if one fails
                
                forecasts['region_fuel'] = rp_forecasts
                
                # Display completion summary
                total_rp_models = len(rp_results)
                successful_forecasts = len(rp_forecasts)
                
                if successful_forecasts == total_rp_models:
                    st.success(f"✅ Forecasting completed for all {total_rp_models} region-fuel combinations!")
                elif successful_forecasts > 0:
                    st.warning(f"⚠️ Forecasting completed with some errors. Successfully processed {successful_forecasts} out of {total_rp_models} combinations.")
                else:
                    st.error("❌ Forecasting failed for all region-fuel combinations. Please check the error messages above.")
            
            status_text.text("📊 Preparing forecast visualizations...")
            progress_bar.progress(80)
            
            # Store forecasts in session state
            st.session_state.forecasts = forecasts
            st.session_state.forecast_config = config
            
            progress_bar.progress(100)
            status_text.text("✅ Forecasts generated successfully!")
            
            # Display results
            _display_forecast_results(forecasts, config, weekly_feats)
            
        except Exception as e:
            st.error(f"❌ Forecasting failed: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()


def _display_forecast_results(forecasts, config, weekly_feats):
    """Display the forecasting results."""
    
    st.success("✅ Forecasts Generated Successfully!")
    
    # Forecast summary
    st.markdown("### 📊 Forecast Summary")
    
    forecast_summary = []
    total_forecasts = 0
    
    if 'overall' in forecasts:
        forecast_summary.append({
            'Model Type': 'Overall Model',
            'Forecasts Generated': 1,
            'Forecast Period': f"{config['forecast_weeks']} weeks",
            'Confidence Intervals': 'Yes' if config['include_confidence'] else 'No'
        })
        total_forecasts += 1
    
    if 'region_fuel' in forecasts:
        rp_count = len(forecasts['region_fuel'])
        forecast_summary.append({
            'Model Type': 'Region-Fuel Models',
            'Forecasts Generated': rp_count,
            'Forecast Period': f"{config['forecast_weeks']} weeks",
            'Confidence Intervals': 'Yes' if config['include_confidence'] else 'No'
        })
        total_forecasts += rp_count
    
    if forecast_summary:
        summary_df = pd.DataFrame(forecast_summary)
        st.dataframe(summary_df, use_container_width=True)
        
        st.info(f"📈 Total forecasts generated: **{total_forecasts}**")
    
    # Display forecast visualizations
    _display_forecast_visualizations(forecasts, config, weekly_feats)
    
    # Forecast export options
    _render_forecast_export(forecasts, config)


def _display_forecast_visualizations(forecasts, config, weekly_feats):
    """Display forecast visualizations and charts."""
    
    st.markdown("### 📈 Forecast Visualizations")
    
    # Create tabs for different forecast types
    tabs = []
    tab_names = []
    
    if 'overall' in forecasts:
        tab_names.append("🌐 Overall Forecast")
    
    if 'region_fuel' in forecasts:
        tab_names.append("🎯 Region-Fuel Forecasts")
    
    if tab_names:
        tabs = st.tabs(tab_names)
        tab_idx = 0
        
        # Overall forecast tab
        if 'overall' in forecasts:
            with tabs[tab_idx]:
                _display_overall_forecast(forecasts['overall'], config, weekly_feats)
            tab_idx += 1
        
        # Region-fuel forecast tab
        if 'region_fuel' in forecasts:
            with tabs[tab_idx]:
                _display_region_fuel_forecasts(forecasts['region_fuel'], config, weekly_feats)


def _display_overall_forecast(overall_forecast, config, weekly_feats):
    """Display overall model forecast results."""
    
    st.markdown("#### 🌐 Overall Market Forecast")
    
    if overall_forecast and 'forecast_data' in overall_forecast:
        # Forecast data table
        st.markdown("##### 📊 Forecast Data")
        forecast_df = overall_forecast['forecast_data'].copy()
        
        if config['include_confidence']:
            display_cols = ['week_start', 'forecast', 'lower_bound', 'upper_bound']
            if all(col in forecast_df.columns for col in display_cols):
                display_df = forecast_df[display_cols].copy()
                display_df.columns = ['Week', 'Forecast', 'Lower Bound', 'Upper Bound']
            else:
                display_df = forecast_df
        else:
            display_cols = ['week_start', 'forecast']
            if all(col in forecast_df.columns for col in display_cols):
                display_df = forecast_df[display_cols].copy()
                display_df.columns = ['Week', 'Forecast']
            else:
                display_df = forecast_df
        
        st.dataframe(prepare_df_for_display(display_df), use_container_width=True)
        
        # Summary statistics
        if 'forecast' in forecast_df.columns:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Average Weekly Forecast", f"{forecast_df['forecast'].mean():.0f}")
            with col2:
                st.metric("Total Period Forecast", f"{forecast_df['forecast'].sum():.0f}")
            with col3:
                st.metric("Min Weekly Forecast", f"{forecast_df['forecast'].min():.0f}")
            with col4:
                st.metric("Max Weekly Forecast", f"{forecast_df['forecast'].max():.0f}")


def _display_region_fuel_forecasts(rp_forecasts, config, weekly_feats):
    """Display region-fuel specific forecast results."""
    
    st.markdown("#### 🎯 Region-Fuel Specific Forecasts")
    
    if not rp_forecasts:
        st.warning("No region-fuel forecasts available.")
        return
    
    # Create region tabs
    regions = sorted(set(data['region'] for data in rp_forecasts.values()))
    region_tabs = st.tabs([f"📍 {region}" for region in regions])
    
    for tab_idx, region in enumerate(regions):
        with region_tabs[tab_idx]:
            region_forecasts = {k: v for k, v in rp_forecasts.items() if v['region'] == region}
            
            for model_key, model_data in region_forecasts.items():
                product = model_data['product']
                forecast_data = model_data['forecast']
                
                with st.expander(f"⛽ {product} Forecast", expanded=True):
                    if forecast_data and 'forecast_df' in forecast_data:
                        # Forecast data
                        forecast_df = forecast_data['forecast_df'].copy()
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown("**📊 Forecast Data**")
                            if config['include_confidence']:
                                display_cols = ['week_start', 'forecast', 'lower_bound', 'upper_bound']
                                if all(col in forecast_df.columns for col in display_cols):
                                    display_df = forecast_df[display_cols].copy()
                                    display_df.columns = ['Week', 'Forecast', 'Lower Bound', 'Upper Bound']
                                else:
                                    display_df = forecast_df
                            else:
                                display_cols = ['week_start', 'forecast']
                                if all(col in forecast_df.columns for col in display_cols):
                                    display_df = forecast_df[display_cols].copy()
                                    display_df.columns = ['Week', 'Forecast']
                                else:
                                    display_df = forecast_df
                            
                            st.dataframe(prepare_df_for_display(display_df), use_container_width=True)
                        
                        with col2:
                            st.markdown("**📈 Summary Statistics**")
                            if 'forecast' in forecast_df.columns:
                                stats_df = pd.DataFrame({
                                    'Metric': ['Average Weekly', 'Total Period', 'Min Weekly', 'Max Weekly'],
                                    'Value': [
                                        f"{forecast_df['forecast'].mean():.0f}",
                                        f"{forecast_df['forecast'].sum():.0f}",
                                        f"{forecast_df['forecast'].min():.0f}",
                                        f"{forecast_df['forecast'].max():.0f}"
                                    ]
                                })
                                st.dataframe(stats_df, use_container_width=True)


def _render_forecast_export(forecasts, config):
    """Render forecast export options."""
    
    st.markdown("### 💾 Export Forecasts")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export to CSV", key="export_csv"):
            # Create combined forecast data for export
            export_data = []
            
            if 'overall' in forecasts and forecasts['overall']:
                overall_data = forecasts['overall'].get('forecast_data', pd.DataFrame())
                if not overall_data.empty:
                    overall_data['Model_Type'] = 'Overall'
                    overall_data['Region'] = 'All'
                    overall_data['Product'] = 'All'
                    export_data.append(overall_data)
            
            if 'region_fuel' in forecasts:
                for model_key, model_data in forecasts['region_fuel'].items():
                    forecast_data = model_data['forecast'].get('forecast_df', pd.DataFrame())
                    if not forecast_data.empty:
                        forecast_data['Model_Type'] = 'Region-Fuel'
                        forecast_data['Region'] = model_data['region']
                        forecast_data['Product'] = model_data['product']
                        export_data.append(forecast_data)
            
            if export_data:
                combined_df = pd.concat(export_data, ignore_index=True)
                csv_data = combined_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Forecasts CSV",
                    data=csv_data,
                    file_name=f"fuel_forecasts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No forecast data available for export.")
    
    with col2:
        if st.button("📈 Export Summary Report", key="export_summary"):
            # Generate summary report
            summary_lines = []
            summary_lines.append("# Fuel Sales Forecast Report")
            summary_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            summary_lines.append(f"Forecast Period: {config['forecast_weeks']} weeks")
            summary_lines.append(f"Confidence Level: {config['confidence_level']:.0%}")
            summary_lines.append("")
            
            if 'overall' in forecasts:
                summary_lines.append("## Overall Market Forecast")
                # Add overall forecast summary
                summary_lines.append("Overall market forecast generated successfully.")
                summary_lines.append("")
            
            if 'region_fuel' in forecasts:
                summary_lines.append("## Region-Fuel Forecasts")
                summary_lines.append(f"Generated forecasts for {len(forecasts['region_fuel'])} region-product combinations:")
                
                for model_key, model_data in forecasts['region_fuel'].items():
                    summary_lines.append(f"- {model_data['region']} - {model_data['product']}")
            
            summary_text = "\n".join(summary_lines)
            
            st.download_button(
                label="📥 Download Summary Report",
                data=summary_text,
                file_name=f"forecast_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
    
    with col3:
        st.info("💡 **Export Options**\n\n📊 CSV: Raw forecast data\n📈 Summary: Executive report")
