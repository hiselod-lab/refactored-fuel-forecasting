import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# Import utility functions
from helpers import (
    parse_week_start,
    prepare_df_for_display,
    ModelConfig,
    validate_input_df,
    format_value_with_unit, 
    format_tick, 
    generate_log_ticks,
    create_region_volume_chart, 
    create_product_volume_chart, 
    create_product_chart,
    create_region_product_chart, 
    create_monthly_sales_chart, 
    create_price_trend_chart,
    plot_actual_vs_predicted, 
    compute_smape,
    prepare_forecast_features, 
    apply_volatility_adjustments, 
    compute_prediction_interval,
    time_series_cv_scores
)

# Import core functions
from core import (
    enhanced_feature_engineering, train_models, evaluate_preds, 
    generate_forecast, generate_detailed_forecast, create_missing_features,
    initialize_session_state, get_feature_columns
)

# Import modular components
from data_overview import show_data_overview
from model_training import show_overall_model_training
from region_training import show_region_fuel_training
from forecasting import show_forecasting_tab

# Page configuration
st.set_page_config(
    page_title="Fuel Forecast Analysis",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Configure logging early
logging.basicConfig(level=logging.INFO)

# Load CSS from file (wrap in <style> to avoid showing raw CSS)
_css = open('style.css', 'r', encoding='utf-8').read()
st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

# ---------- Forecasting utilities ----------



# Helper function to convert datetime columns to strings for Streamlit display
def prepare_df_for_display(df):
    """Convert any datetime columns to strings to avoid Arrow conversion issues"""
    if df is None or df.empty:
        return df
    
    df_display = df.copy()
    for col in df_display.columns:
        # Check if column is datetime type
        if pd.api.types.is_datetime64_any_dtype(df_display[col]):
            df_display[col] = df_display[col].dt.strftime('%Y-%m-%d')
        # Check if column is object type and might contain Timestamp objects
        elif df_display[col].dtype == 'object' and len(df_display) > 0:
            # Try to convert the entire column to datetime if it contains any Timestamp objects
            try:
                # Check if any values in the column are Timestamp objects
                sample_values = df_display[col].dropna().head(10).tolist()
                has_timestamp = any(isinstance(x, (pd.Timestamp, datetime)) for x in sample_values)
                
                if has_timestamp:
                    # Convert all values in the column to strings
                    df_display[col] = df_display[col].apply(
                        lambda x: x.strftime('%Y-%m-%d') if isinstance(x, (pd.Timestamp, datetime)) 
                        else str(x) if x is not None else x
                    )
            except Exception:
                # If there's any error, try to handle individual Timestamp objects
                try:
                    df_display[col] = df_display[col].apply(
                        lambda x: x.strftime('%Y-%m-%d') if isinstance(x, (pd.Timestamp, datetime))
                        else x
                    )
                except Exception:
                    # If all else fails, convert the column to strings
                    df_display[col] = df_display[col].astype(str)
    
    return df_display

@st.cache_data(ttl=3600)  # Cache for 1 hour to improve performance
def create_metrics_dashboard(metrics):
    """Create a metrics dashboard with cards."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MAE", f"{metrics['MAE']:.2f}")
    with col2:
        st.metric("RMSE", f"{metrics['RMSE']:.2f}")
    with col3:
        st.metric("R²", f"{metrics['R2']:.3f}")
    with col4:
        st.metric("SMAPE", f"{metrics['SMAPE']*100:.2f}%")

def create_comprehensive_data_overview(df):
    """Create comprehensive data overview with multiple visualizations."""
    
    # Pre-compute statistics to avoid recalculation
    stats = {
        "total_records": len(df),
        "unique_regions": df['Region'].nunique(),
        "unique_products": df['Product'].nunique(),
        "date_range": f"{df['week_start'].min().strftime('%Y-%m-%d') if not isinstance(df['week_start'].min(), str) else df['week_start'].min()} to {df['week_start'].max().strftime('%Y-%m-%d') if not isinstance(df['week_start'].max(), str) else df['week_start'].max()}"
    }

    # Basic statistics
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        st.metric("Total Records", stats["total_records"])
    with col2:
        st.metric("Regions", stats["unique_regions"])
    with col3:
        st.metric("Products", stats["unique_products"])
    with col4:
        st.metric("Date Range", stats["date_range"])
    
    # Price statistics by fuel type
    st.markdown("### 💰 Price Statistics by Fuel Type")
    price_stats = df.groupby('Product')['avg_price'].agg(['mean', 'min', 'max', 'std']).round(2)
    price_stats.columns = ['Average Price (PKR)', 'Min Price (PKR)', 'Max Price (PKR)', 'Std Dev (PKR)']
    st.dataframe(prepare_df_for_display(price_stats), use_container_width=True)



    # Visualizations
    st.markdown("### 📊 Sales Volume Analysis")
    
    # Create region and product charts with caching

    tab_region, tab_product, tab_region_product = st.tabs(["By Region", "By Product", "Region vs Product"])
    
    with tab_region:
        fig_region = create_region_volume_chart(df)
        st.plotly_chart(fig_region, use_container_width=True)
    
    with tab_product:
        # Create toggle widget inside the tab
        show_lagged_product = st.toggle("Show logarithmic scale", key="product_lagged", help="Toggle to view data in logarithmic scale")
        product_df = create_product_volume_chart(df)
        fig_product = create_product_chart(product_df, log_y=show_lagged_product)
        st.plotly_chart(fig_product, use_container_width=True)
    
    with tab_region_product:
        # Create toggle widget inside the tab
        show_lagged_rp = st.toggle("Show logarithmic scale", key="rp_lagged", help="Toggle to view data in logarithmic scale")
        fig_rp = create_region_product_chart(df, log_y=show_lagged_rp)
        st.plotly_chart(fig_rp, use_container_width=True)

    # Time series analysis
    st.markdown("### 📈 Monthly Sales Volume Trend")
    


    # Create toggle widget directly in the flow
    show_lagged_month = st.toggle("Show logarithmic scale", key="monthly_lagged", help="Toggle to view data in logarithmic scale")
    fig_monthly = create_monthly_sales_chart(df, log_y=show_lagged_month)
    st.plotly_chart(fig_monthly, use_container_width=True)

    st.markdown("### 💰 Monthly Average Price Trend by Fuel Type")
    

    
    # Generate and display the price trend chart
    fig_price = create_price_trend_chart(df)
    st.plotly_chart(fig_price, use_container_width=True)
    

def main():
    # Initialize session state variables
    if "model_params" not in st.session_state:
        st.session_state.model_params = {
            "overall": {
                "train_ratio": 0.8,
                "split_method": "Time-based",
                "feature_selection": True,
                "k_features": 20,
                "lgbm_learning_rate": 0.01,
                "lgbm_n_estimators": 1000,
                "lgbm_max_depth": 7,
                "rf_n_estimators": 200,
                "rf_max_depth": 10,
                "rf_min_samples_split": 2,
                "ensemble_method": "Average",
                "lgbm_weight": 0.5
            },
            "region_fuel": {
                "train_ratio": 0.8,
                "split_method": "Time-based",
                "feature_selection": True,
                "k_features": 15,
                "lgbm_learning_rate": 0.01,
                "lgbm_n_estimators": 500,
                "lgbm_max_depth": 7,
                "rf_n_estimators": 100,
                "rf_max_depth": 10,
                "rf_min_samples_split": 2,
                "ensemble_method": "Average",
                "lgbm_weight": 0.5
            }
        }
    
    if "forecast_params" not in st.session_state:
        st.session_state.forecast_params = {
            "forecast_weeks": 4,
            "forecast_method": "Direct",
            "include_confidence": True,
            "confidence_level": 0.9
        }
    
    if "run_overall" not in st.session_state:
        st.session_state.run_overall = False
    
    if "run_region_fuel" not in st.session_state:
        st.session_state.run_region_fuel = False
        
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
        
    if "default_data_loaded" not in st.session_state:
        st.session_state.default_data_loaded = False
        
    if "overall_metrics" not in st.session_state:
        st.session_state.overall_metrics = None
        
    if "overall_y_test" not in st.session_state:
        st.session_state.overall_y_test = None
        
    if "overall_y_pred" not in st.session_state:
        st.session_state.overall_y_pred = None
        
    if "overall_test_df" not in st.session_state:
        st.session_state.overall_test_df = None
        
    if "rp_results" not in st.session_state:
        st.session_state.rp_results = {}
        
    if "summary_df" not in st.session_state:
        st.session_state.summary_df = None
        
    if "model_selection" not in st.session_state:
        st.session_state.model_selection = "overall"
    
    # Header with animation
    st.markdown('<div class="animate-fade-in">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">⛽ Fuel Forecast Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced analytics and predictive modeling for fuel sales data</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Streamlined file upload in header
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        # Create a container for the upload button
        st.markdown('<div style="display: flex; justify-content: center; margin-bottom: 1rem;">', unsafe_allow_html=True)
        
        # File upload option with minimal footprint
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed", 
                                        help="Upload a CSV file with your fuel sales data")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle file upload and data loading
    if uploaded_file is not None:
        try:
            weekly_feats = pd.read_csv(uploaded_file)
            validate_input_df(weekly_feats)
            weekly_feats = parse_week_start(weekly_feats)
            st.toast("✅ File uploaded successfully", icon="✅")
            st.session_state.data_loaded = True
        except FileNotFoundError as e:
            st.error(f"File not found: {e}")
            st.stop()
        except pd.errors.ParserError as e:
            st.error(f"CSV parsing error: {e}")
            st.stop()
        except ValueError as e:
            st.error(f"Validation error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Upload failed: {e}")
            st.stop()
    else:
        # Load default data if no file is uploaded
        try:
            weekly_feats = pd.read_csv("weekly_features_no_ogra_for_11_years.csv")
            validate_input_df(weekly_feats)
            weekly_feats = parse_week_start(weekly_feats)
            if not st.session_state.get('default_data_loaded', False):
                st.info("Using default dataset. Upload your own CSV file for custom analysis.")
                st.session_state.default_data_loaded = True
        except FileNotFoundError:
            st.error("Default data file not found. Please upload a CSV file.")
            st.stop()
        except pd.errors.ParserError as e:
            st.error(f"Default CSV parsing error: {e}")
            st.stop()
        except ValueError as e:
            st.error(f"Default data validation error: {e}")
            st.stop()
    
    # Executive Dashboard removed as requested
    
    # Create tabs with centered alignment
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: linear-gradient(135deg, #ffffff, #f8f9fa);
        padding: 1rem;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border: none;
        justify-content: center; /* Center the tabs */
    }
    </style>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Data Overview", 
        "🤖 Model Training & Analysis", 
        "🌍 Region-Fuel Analysis", 
        "🔮 Forecasting",
        "📋 Results & Export"
    ])
    
    # Tab 1: Data Overview
    with tab1:
        show_data_overview(weekly_feats)
    
    # Tab 2: Model Training & Analysis (Combined)
    with tab2:
        st.markdown('<h2 class="section-header">🤖 Model Training & Analysis</h2>', unsafe_allow_html=True)
        
        # Add CSS for the model selection tabs - moved outside the model-section div
        st.markdown("""
        <style>
        .model-selection-container {
            background: #ffffff;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }
        
        .model-selection-title {
            font-size: 1.4rem;
            font-weight: 600;
            color: #262730;
            margin-bottom: 1rem;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Model selection title and container - moved outside the model-section div
        st.markdown('<div class="model-selection-title">Select Model Type to Run</div>', unsafe_allow_html=True)
        
        # Initialize session state for model selection if not exists
        if 'model_selection' not in st.session_state:
            st.session_state.model_selection = "overall"
        
        # Create tabs for model selection
        model_tab1, model_tab2 = st.tabs(["🚀 Overall Model", "🔍 Region-Fuel Models"])
        
        # Overall Model Training Tab
        with model_tab1:
            show_overall_model_training(weekly_feats, enhanced_feature_engineering, train_models, evaluate_preds, generate_forecast, generate_detailed_forecast, create_metrics_dashboard)
        
        with model_tab2:
            show_region_fuel_training(weekly_feats, enhanced_feature_engineering, train_models, evaluate_preds, generate_forecast, generate_detailed_forecast, create_metrics_dashboard)
        
        # Overall Model Training section has been moved inside the model_tab1
        
        # Region-Fuel Model Training section has been moved inside the model_tab2
    
    # Tab 3: Region-Fuel Analysis
    with tab3:
        st.markdown('<h2 class="section-header">🌍 Region-Fuel Specific Analysis</h2>', unsafe_allow_html=True)
        
        # Individual Model Results section (moved from region training)
        if 'rp_results' in st.session_state and st.session_state.rp_results is not None:
            st.markdown("### 🔍 Individual Model Results")
            
            if st.session_state.rp_results:
                # Create tabs for different regions
                regions = sorted(set(rp['region'] for rp in st.session_state.rp_results.values()))
                region_tabs = st.tabs([f"📍 {region}" for region in regions])
                
                for tab_idx, region in enumerate(regions):
                    with region_tabs[tab_idx]:
                        region_models = {k: v for k, v in st.session_state.rp_results.items() if v['region'] == region}
                        
                        # Sort models by product in consistent order: PMG, HSD, HOBC
                        product_order = ['PMG', 'HSD', 'HOBC']
                        sorted_models = sorted(region_models.items(), 
                                             key=lambda x: product_order.index(x[1]['product']) if x[1]['product'] in product_order else 999)
                        
                        for model_key, model_data in sorted_models:
                            with st.expander(f"🔍 {model_data['product']} Model Details", expanded=False):
                                from helpers import format_metric_value, format_large_number
                                
                                # Performance metrics and data information in vertical layout
                                st.markdown("### 📊 Performance Metrics")
                                metrics = model_data['metrics']
                                
                                # Create metrics display with larger font
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.markdown(f"<div style='font-size: 20px;'><b>MAE:</b><br>{format_large_number(metrics.get('MAE', 0))}</div>", unsafe_allow_html=True)
                                with col2:
                                    st.markdown(f"<div style='font-size: 20px;'><b>RMSE:</b><br>{format_large_number(metrics.get('RMSE', 0))}</div>", unsafe_allow_html=True)
                                with col3:
                                    st.markdown(f"<div style='font-size: 20px;'><b>R²:</b><br>{format_metric_value(metrics.get('R2', 0), 'R2')}</div>", unsafe_allow_html=True)
                                with col4:
                                    st.markdown(f"<div style='font-size: 20px;'><b>SMAPE:</b><br>{format_metric_value(metrics.get('SMAPE', 0), 'SMAPE')}</div>", unsafe_allow_html=True)
                                
                                st.markdown("### 📈 Data Information")
                                config = st.session_state.get('rp_config', None)
                                train_ratio = config.train_ratio if config else 0.8
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.markdown(f"<div style='font-size: 20px;'><b>Total Points:</b><br>{model_data['data_size']}</div>", unsafe_allow_html=True)
                                with col2:
                                    st.markdown(f"<div style='font-size: 20px;'><b>Training Points:</b><br>{int(model_data['data_size'] * train_ratio)}</div>", unsafe_allow_html=True)
                                with col3:
                                    st.markdown(f"<div style='font-size: 20px;'><b>Test Points:</b><br>{model_data['data_size'] - int(model_data['data_size'] * train_ratio)}</div>", unsafe_allow_html=True)
                                
                                # Actual vs Predicted plot
                                test_data = model_data['results'].get('test_data', {})
                                y_test = test_data.get('y_test', pd.Series())
                                y_pred = test_data.get('y_pred', pd.Series())
                                test_df = test_data.get('test_df', pd.DataFrame())
                                
                                if len(y_test) > 0 and len(y_pred) > 0 and len(test_df) > 0:
                                    fig = plot_actual_vs_predicted(
                                        test_df,
                                        y_test,
                                        y_pred,
                                        model_data['region'],
                                        model_data['product']
                                    )
                                    
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True, 
                                                      key=f"rp_pred_chart_{model_key}")
                                else:
                                    st.info("No test data available for visualization.")
            else:
                st.info("🔍 Please run the Region-Fuel Models first to see individual model results.")
            
            # Add forecasting info messages at the bottom
            st.markdown("---")
            st.info("ℹ️ Forecasting functionality has been moved to the dedicated '🔮 Forecasting' tab.")
            st.warning("⚠️ No forecasts available for this combination. Enable forecasting in the '🔮 Forecasting' tab.")
        else:
            st.info("🔍 Please run the Region-Fuel Models first to see detailed analysis.")
    
    # Tab 4: Forecasting
    with tab4:
        st.markdown('<h2 class="section-header">🔮 Forecasting</h2>', unsafe_allow_html=True)
        
        # Check if models are trained
        if 'rp_results' not in st.session_state or st.session_state.rp_results is None:
            st.warning("⚠️ Please train Region-Fuel models first before using the forecasting feature.")
            st.info("Go to the 'Model Training & Analysis' tab to train models.")
        else:
            # Display model training information
            st.markdown("### 🤖 Trained Model Information")
            
            # Show when models were trained
            if 'model_training_time' not in st.session_state and 'rp_results' in st.session_state and st.session_state.rp_results:
                # Set the timestamp when models are actually trained, not when the page is loaded
                st.session_state.model_training_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
            st.info(f"Models were last trained at: {st.session_state.model_training_time if 'model_training_time' in st.session_state else 'Not trained yet'}")
            
            # Option to discard trained models and retrain
            if st.button("Discard Trained Models and Re-train", type="secondary"):
                # Clear model results
                if 'rp_results' in st.session_state:
                    del st.session_state.rp_results
                if 'summary_df' in st.session_state:
                    del st.session_state.summary_df
                if 'model_training_time' in st.session_state:
                    del st.session_state.model_training_time
                st.session_state.run_region_fuel = True
                st.rerun()
            
            # Forecasting Configuration
            st.markdown("### ⚙️ Configure Forecasting Options")
            
            # Use a form to group forecasting options
            with st.form(key="forecasting_form"):
                col1, col2 = st.columns(2)
                with col1:
                    forecast_weeks = st.slider("Number of Weeks to Forecast", min_value=1, max_value=260, 
                                             value=st.session_state.get('forecast_params', {}).get('forecast_weeks', 4), step=1,
                                             help="Number of weeks to forecast into the future (maximum 5 years/260 weeks)")
                with col2:
                    forecast_method = st.radio("Forecast Method", ["Direct", "Recursive"], 
                                             index=0 if st.session_state.get('forecast_params', {}).get('forecast_method', "Direct") == "Direct" else 1,
                                             help="Direct: forecast all weeks at once, Recursive: use previous forecasts as inputs")
                
                # Advanced forecasting options
                st.markdown("#### Advanced Forecasting Options")
                col1, col2 = st.columns(2)
                with col1:
                    include_confidence = st.checkbox("Include Confidence Intervals", 
                                                  value=st.session_state.get('forecast_params', {}).get('include_confidence', True),
                                                  help="Show prediction uncertainty ranges")
                with col2:
                    if include_confidence:
                        confidence_level = st.select_slider("Confidence Level", options=[0.8, 0.85, 0.9, 0.95, 0.99], 
                                                          value=st.session_state.get('forecast_params', {}).get('confidence_level', 0.9),
                                                          help="Confidence level for prediction intervals")
                    else:
                        confidence_level = 0.9
                
                # Submit button for the form
                submit_forecast = st.form_submit_button("Apply Forecasting Settings", use_container_width=True)
                
            # Update session state when form is submitted
            if submit_forecast:
                if "forecast_params" not in st.session_state:
                    st.session_state.forecast_params = {}
                
                st.session_state.forecast_params = {
                    "forecast_weeks": forecast_weeks,
                    "forecast_method": forecast_method,
                    "include_confidence": include_confidence,
                    "confidence_level": confidence_level
                }
                
                # Display success message outside the form to prevent overlay
                st.success(f"Forecasting settings applied successfully.")
            
            # Display current forecasting settings
            if 'forecast_params' in st.session_state:
                st.info(f"Current settings: Forecasting {st.session_state.forecast_params.get('forecast_weeks', 4)} weeks using {st.session_state.forecast_params.get('forecast_method', 'Direct')} method.")
            
            # Run Forecasting button outside the form with container width to prevent overlay
            if st.button("Run Forecasting", type="primary", use_container_width=True):
                # Create a progress bar with a container to prevent UI issues
                with st.container():
                    # Create progress elements
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Get all region-product combinations
                    rp_combinations = list(st.session_state.rp_results.keys())
                    total_combinations = len(rp_combinations)
                    
                    # Flag to track if we've processed all combinations
                    all_processed = True
                    
                    # Run forecasting for all combinations
                    for i, rp_key in enumerate(rp_combinations):
                        # Parse the region and product from the key (format: "region_product")
                        if '_' in rp_key:
                            region, product = rp_key.split('_', 1)
                        else:
                            # Fallback if the key format is different
                            st.error(f"Invalid key format: {rp_key}")
                            continue
                        
                        # Update progress display
                        progress = int((i / total_combinations) * 100)
                        progress_bar.progress(progress)
                        progress_text.markdown(f"**Processing: {progress}%** | Running forecasts...", unsafe_allow_html=False)
                        status_text.markdown(f"Currently processing: **{region} - {product}**", unsafe_allow_html=False)
                        
                        # Get the result data for this combination
                        res = st.session_state.rp_results[rp_key]
                        
                        # Always generate forecasts to ensure latest parameters are applied
                        try:
                            forecast_params = st.session_state.forecast_params
                            forecast_weeks = forecast_params.get('forecast_weeks', 4)
                            forecast_method = forecast_params.get('forecast_method', 'Direct')
                            include_confidence = forecast_params.get('include_confidence', False)
                            confidence_level = forecast_params.get('confidence_level', 0.9)

                            group = weekly_feats[
                                (weekly_feats['Region'] == region) &
                                (weekly_feats['Product'] == product)
                            ].sort_values('week_start').reset_index(drop=True).copy()

                            # Ensure datetime type for feature engineering
                            if not pd.api.types.is_datetime64_any_dtype(group['week_start']):
                                group['week_start'] = pd.to_datetime(group['week_start'])

                            # Use deterministic feature engineering for historical data
                            group_eng = enhanced_feature_engineering(group, forecasting_mode=False)

                            forecast = generate_detailed_forecast(
                                res,
                                group_eng,
                                product,
                                forecast_weeks,
                                forecast_method,
                                include_confidence,
                                confidence_level,
                            )

                            res['forecast'] = forecast
                            st.session_state.rp_results[rp_key] = res
                        except Exception as e:
                            st.error(f"Error processing {region} - {product}: {str(e)}")
                            all_processed = False
                    
                    # Set progress to 100% when complete
                    progress_bar.progress(100)
                    progress_text.markdown("**Processing: 100%** | Complete!", unsafe_allow_html=False)
                    status_text.empty()
                    
                    # Success message
                    if all_processed:
                        st.success("✅ Forecasting completed for all region-fuel combinations!")
                    else:
                        st.warning("⚠️ Forecasting completed with some errors. Please check the messages above.")
            
            # Forecasting Visualization
            if 'rp_results' in st.session_state and any('forecast' in res for res in st.session_state.rp_results.values()):
                st.markdown("### 🎯 Region-Fuel Specific Forecasts")
                
                # Select region-product for detailed analysis
                rp_options = list(st.session_state.rp_results.keys())
                
                # Use a session state variable to track the previously selected combination
                if 'selected_forecast_rp' not in st.session_state:
                    st.session_state.selected_forecast_rp = rp_options[0] if rp_options else None
                
                # Create a container for the dropdown to prevent UI overlay
                with st.container():
                    # Use the selectbox with the current value from session state
                    selected_rp = st.selectbox(
                        "Select Region-Fuel Combination for Forecast", 
                        rp_options,
                        index=rp_options.index(st.session_state.selected_forecast_rp) if st.session_state.selected_forecast_rp in rp_options else 0,
                        key="forecast_rp_selector"
                    )
                
                # Update the session state with the new selection
                st.session_state.selected_forecast_rp = selected_rp
                
                if selected_rp in st.session_state.rp_results:
                    res = st.session_state.rp_results[selected_rp]
                    
                    # Parse the region and product from the selected_rp key
                    if '_' in selected_rp:
                        region, product = selected_rp.split('_', 1)
                    else:
                        st.error(f"Invalid key format: {selected_rp}")
                        return
                    
                    # Display forecasts if available
                    if 'forecast' in res:
                        if not res['forecast']['values']:
                            st.warning("No forecast generated for this combination. Please run the forecasting step again.")
                        else:
                            st.markdown(f"#### 🔮 Sales Volume Forecast: {region} - {product}")

                            # Normalize forecast results to plain floats
                            raw_vals = res['forecast']['values']
                            raw_lowers = res['forecast']['lower_bounds']
                            raw_uppers = res['forecast']['upper_bounds']
                            norm_vals = [float(np.asarray(v).ravel()[0]) for v in raw_vals]
                            norm_lowers = [float(np.asarray(v).ravel()[0]) if v is not None else None for v in raw_lowers]
                            norm_uppers = [float(np.asarray(v).ravel()[0]) if v is not None else None for v in raw_uppers]

                            # Replace original forecast entries with normalized scalars
                            res['forecast']['values'] = norm_vals
                            res['forecast']['lower_bounds'] = norm_lowers
                            res['forecast']['upper_bounds'] = norm_uppers

                            # Create a DataFrame for the forecast with string dates
                            forecast_data = []
                            for i in range(len(norm_vals)):
                                row = {
                                    'Date': res['forecast']['dates_str'][i],
                                    'Forecast': norm_vals[i]
                                }
                                if norm_lowers[i] is not None:
                                    row['Lower Bound'] = norm_lowers[i]
                                    row['Upper Bound'] = norm_uppers[i]
                                forecast_data.append(row)

                            forecast_df = pd.DataFrame(forecast_data)

                            # Display forecast table with formatted values for readability
                            forecast_display = forecast_df.copy()
                            for col in ['Forecast', 'Lower Bound', 'Upper Bound']:
                                if col in forecast_display:
                                    forecast_display[col] = forecast_display[col].map(format_value_with_unit)
                            st.dataframe(prepare_df_for_display(forecast_display), use_container_width=True)

                            # Create forecast visualization
                            fig_forecast = go.Figure()
                        
                            # Get historical data for this combination
                            combo_data = weekly_feats[
                                (weekly_feats['Region'] == region) & 
                                (weekly_feats['Product'] == product)
                            ].sort_values('week_start')
                            
                            # Ensure week_start is in the correct format for visualization
                            if pd.api.types.is_datetime64_any_dtype(combo_data['week_start']):
                                combo_data['week_start_str'] = combo_data['week_start'].dt.strftime('%Y-%m-%d')
                            else:
                                combo_data['week_start_str'] = combo_data['week_start'].astype(str)
                            
                            # Add historical data
                            historical_data = combo_data.sort_values('week_start')
                            if not historical_data.empty:
                                fig_forecast.add_trace(go.Scatter(
                                    x=historical_data['week_start_str'],
                                    y=historical_data['sales_volume'],
                                    mode='lines+markers',
                                    name='Historical',
                                    line=dict(color='blue'),
                                    hovertemplate='Date: %{x}<br>Sales Volume: %{y:.2f} (%{customdata})<extra></extra>',
                                    customdata=[format_value_with_unit(val) for val in historical_data['sales_volume']]
                                ))
                            
                            # Add forecast
                            fig_forecast.add_trace(go.Scatter(
                                x=res['forecast']['dates_str'],
                                y=norm_vals,
                                mode='lines+markers',
                                name='Forecast',
                                line=dict(color='red', dash='dash'),
                                hovertemplate='Date: %{x}<br>Forecast: %{y:.2f} (%{customdata})<extra></extra>',
                                customdata=[format_value_with_unit(val) for val in norm_vals]
                            ))
                            
                            # Only draw the confidence interval if bounds are available
                            if norm_lowers and norm_lowers[0] is not None:
                                # Create x values for the confidence interval (forward then backward)
                                x_conf = res['forecast']['dates_str'] + res['forecast']['dates_str'][::-1]
                                # Create y values for the confidence interval (upper bounds then lower bounds reversed)
                                y_conf = norm_uppers + norm_lowers[::-1]

                                fig_forecast.add_trace(go.Scatter(
                                    x=x_conf,
                                    y=y_conf,
                                    fill='toself',
                                    fillcolor='rgba(255,0,0,0.2)',
                                    line=dict(color='rgba(255,0,0,0)'),
                                    name=f"Confidence Interval ({int(res['forecast']['confidence_level']*100)}%)",
                                    hoverinfo='skip'
                                ))
                            
                            # Update layout
                            fig_forecast.update_layout(
                                title=f'Sales Volume Forecast: {region} - {product}',
                                xaxis_title='Date',
                                yaxis_title='Sales Volume',
                                height=500,
                                template='plotly_white',
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(
                                fig_forecast,
                                use_container_width=True,
                                key=f"forecast_chart_{region}_{product}"
                            )
                            
                            # Display forecast information
                            st.markdown("#### Forecast Information")
                            st.markdown(f"- **Forecast Method:** {res['forecast']['method']}")
                            st.markdown(f"- **Forecast Horizon:** {len(res['forecast']['dates'])} weeks")
                            if res['forecast']['confidence_level'] is not None:
                                st.markdown(f"- **Confidence Level:** {int(res['forecast']['confidence_level']*100)}%")
                            
                            # Download forecast as CSV
                            try:
                                csv = forecast_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Forecast as CSV",
                                    data=csv,
                                    file_name=f"{region}_{product}_forecast.csv",
                                    mime="text/csv"
                                )
                            except Exception as e:
                                st.error(f"Error generating CSV: {e}")
                                st.info("Please try again or contact support if the issue persists.")
                    else:
                        st.info("No forecast data available for this combination. Please run the forecasting step first.")
                else:
                    st.info("🔍 Please run the Region-Fuel Models first to see forecasts.")
    
    # Tab 5: Results & Export
    with tab5:
        st.markdown('<h2 class="section-header">📋 Model Results & Export</h2>', unsafe_allow_html=True)
        
        if 'summary_df' in st.session_state and st.session_state.summary_df is not None:
            st.markdown("### 📊 Complete Model Results")
            # Use helper function to prepare DataFrame for display
            display_df = prepare_df_for_display(st.session_state.summary_df)
            st.dataframe(display_df.sort_values('MAE'), use_container_width=True)
            
            # Download results
            try:
                # Use helper function to prepare DataFrame for download
                if st.session_state.summary_df is not None:
                    download_df = prepare_df_for_display(st.session_state.summary_df.copy())
                    
                    csv = download_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="region_fuel_analysis_results.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error generating CSV: {e}")
                st.info("Please try again or contact support if the issue persists.")
        
        if 'overall_metrics' in st.session_state and st.session_state.overall_metrics is not None:
            st.markdown("### 🤖 Overall Model Results")
            create_metrics_dashboard(st.session_state.overall_metrics)
            
            # Download overall results
            overall_results = pd.DataFrame([st.session_state.overall_metrics])
            csv_overall = overall_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Overall Results as CSV",
                data=csv_overall,
                file_name="overall_model_results.csv",
                mime="text/csv"
            )
    
    # Footer removed as requested

if __name__ == "__main__":
    main()