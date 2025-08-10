"""
Region Training Module
Handles the Region-Fuel specific model training functionality for the fuel forecasting app.
"""

import streamlit as st
import pandas as pd
import numpy as np
from helpers import (
    ModelConfig,
    time_series_cv_scores,
    plot_actual_vs_predicted,
    prepare_df_for_display
)


def show_region_fuel_training(weekly_feats: pd.DataFrame, enhanced_feature_engineering, train_models, evaluate_preds, generate_forecast, generate_detailed_forecast, create_metrics_dashboard) -> None:
    """
    Render the complete Region-Fuel Model training tab.
    
    Args:
        weekly_feats: The main dataframe with fuel sales data
        enhanced_feature_engineering: Feature engineering function
        train_models: Model training function
        evaluate_preds: Model evaluation function
        generate_forecast: Forecast generation function
        generate_detailed_forecast: Detailed forecast function
        create_metrics_dashboard: Metrics dashboard function
    """
    # Model Parameter Customization Section
    with st.expander("⚙️ Region-Fuel Model Parameters", expanded=False):
        config = _render_region_model_parameters()
    
    # Training Button and Results
    if st.button("🔍 Train Region-Fuel Models", type="primary", key="train_region_fuel"):
        _run_region_fuel_training(weekly_feats, config, enhanced_feature_engineering, train_models, evaluate_preds, generate_forecast, generate_detailed_forecast, create_metrics_dashboard)
    
    # Display existing results if available
    elif hasattr(st.session_state, 'rp_trained') and st.session_state.rp_trained:
        if hasattr(st.session_state, 'rp_results') and hasattr(st.session_state, 'rp_summary') and hasattr(st.session_state, 'rp_config'):
            _display_region_training_results(
                st.session_state.rp_results, 
                st.session_state.rp_summary, 
                st.session_state.rp_config
            )


def _render_region_model_parameters() -> ModelConfig:
    """Render the region-fuel model parameters form and return ModelConfig."""
    st.markdown("### Region-Fuel Specific Parameters")
    
    # Use a form to prevent automatic reruns when UI components change
    with st.form(key="region_fuel_model_form"):
        st.info("🎯 These models are trained separately for each Region-Product combination")
        
        # Train-Test Split Configuration
        st.markdown("#### 📊 Train-Test Split Configuration")
        col1, col2 = st.columns(2)
        with col1:
            train_ratio = st.slider("Training Data Ratio", min_value=0.6, max_value=0.9, value=0.8, step=0.05,
                                   help="Proportion of data used for training per region-product")
        with col2:
            split_method = st.radio("Split Method", ["Time-based", "Random"], index=0,
                                   help="Time-based recommended for region-fuel models")
        
        # Feature Selection
        st.markdown("#### 🎯 Feature Selection")
        col1, col2 = st.columns(2)
        with col1:
            feature_selection = st.checkbox("Enable Feature Selection", value=True,
                                          help="Select best features per region-product combination")
        with col2:
            k_features = st.number_input("Number of Features (if enabled)", min_value=5, max_value=30, value=15,
                                       help="Fewer features recommended for region-specific models")
        
        # LightGBM Parameters (tuned for smaller datasets)
        st.markdown("#### 🌟 LightGBM Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            lgbm_learning_rate = st.select_slider("Learning Rate", 
                                                 options=[0.01, 0.05, 0.1, 0.2], 
                                                 value=0.05,
                                                 help="Higher learning rate for smaller datasets")
        with col2:
            lgbm_n_estimators = st.select_slider("Number of Trees", 
                                                options=[100, 300, 500, 800], 
                                                value=500,
                                                help="Fewer trees to prevent overfitting")
        with col3:
            lgbm_max_depth = st.select_slider("Max Tree Depth", 
                                             options=[3, 5, 7, 10], 
                                             value=5,
                                             help="Shallower trees for region-specific data")
        
        # Random Forest Parameters
        st.markdown("#### 🌳 Random Forest Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            rf_n_estimators = st.select_slider("Number of Trees (RF)", 
                                              options=[50, 100, 150, 200], 
                                              value=100,
                                              help="Moderate forest size for region data")
        with col2:
            rf_max_depth = st.select_slider("Max Tree Depth (RF)", 
                                           options=[5, 8, 10, 15, None], 
                                           value=8,
                                           help="Controlled depth to prevent overfitting")
        with col3:
            rf_min_samples_split = st.select_slider("Min Samples Split", 
                                                   options=[2, 5, 10], 
                                                   value=5,
                                                   help="Higher split requirement for stability")
        
        # Ensemble Configuration
        st.markdown("#### 🤝 Ensemble Configuration")
        col1, col2 = st.columns(2)
        with col1:
            ensemble_method = st.radio("Ensemble Method", 
                                     ["Average", "Weighted Average"], 
                                     index=1,
                                     help="Weighted average often works better for region-specific models")
        with col2:
            lgbm_weight = st.slider("LightGBM Weight (if Weighted)", 
                                   min_value=0.0, max_value=1.0, value=0.6, step=0.1,
                                   help="LightGBM often performs better on region-specific data")
        
        # Submit button for the form
        submitted = st.form_submit_button("💾 Save Region-Fuel Configuration", type="secondary")
    
    # Create ModelConfig object
    return ModelConfig(
        train_ratio=train_ratio,
        split_method=split_method,
        feature_selection=feature_selection,
        k_features=k_features,
        lgbm_learning_rate=lgbm_learning_rate,
        lgbm_n_estimators=lgbm_n_estimators,
        lgbm_max_depth=lgbm_max_depth,
        rf_n_estimators=rf_n_estimators,
        rf_max_depth=rf_max_depth,
        rf_min_samples_split=rf_min_samples_split,
        ensemble_method=ensemble_method,
        lgbm_weight=lgbm_weight
    )


def _run_region_fuel_training(weekly_feats, config, enhanced_feature_engineering, train_models, evaluate_preds, generate_forecast, generate_detailed_forecast, create_metrics_dashboard):
    """Execute the region-fuel model training process."""
    import time
    
    # Get unique region-product combinations
    region_products = weekly_feats[['Region', 'Product']].drop_duplicates()
    total_combinations = len(region_products)
    
    st.info(f"🎯 Training models for {total_combinations} Region-Product combinations")
    
    with st.spinner("🔄 Training Region-Fuel Models..."):
        main_progress = st.progress(0)
        status_text = st.empty()
        eta_text = st.empty()
        start_time = time.time()
        
        # Results storage
        rp_results = {}
        rp_summary = []
        
        try:
            # Feature engineering once
            status_text.text("🔧 Engineering features...")
            feature_engineered = enhanced_feature_engineering(weekly_feats, forecasting_mode=False, random_state=42)
            feature_cols = [col for col in feature_engineered.columns 
                          if col not in ['Region', 'Product', 'week_start', 'sales_volume', 'sales_amount']]
            
            # Train models for each region-product combination
            for idx, (_, rp_row) in enumerate(region_products.iterrows()):
                region = rp_row['Region']
                product = rp_row['Product']
                
                status_text.text(f"🤖 Training {region} - {product} ({idx+1}/{total_combinations})...")
                progress = (idx + 1) / total_combinations
                main_progress.progress(progress)
                
                # Update ETA
                if idx > 0:  # Only calculate ETA after first iteration
                    elapsed = time.time() - start_time
                    estimated_total = elapsed / progress
                    eta = estimated_total - elapsed
                    eta_text.text(f"⏱️ ETA: {int(eta//60)}m {int(eta%60)}s")
                
                # Filter data for this region-product combination
                rp_data = feature_engineered[
                    (feature_engineered['Region'] == region) & 
                    (feature_engineered['Product'] == product)
                ].copy()
                
                if len(rp_data) < 50:  # Skip if insufficient data
                    st.warning(f"⚠️ Skipping {region}-{product}: Insufficient data ({len(rp_data)} rows)")
                    continue
                
                try:
                    # Train model for this combination
                    results = train_models(rp_data, feature_cols, config)
                    
                    # Extract y_test and y_pred from the nested structure
                    test_data = results.get('test_data', {})
                    y_test = test_data.get('y_test', pd.Series())
                    y_pred = test_data.get('y_pred', pd.Series())
                    
                    if len(y_test) > 0 and len(y_pred) > 0:
                        metrics = evaluate_preds(y_test, y_pred)
                    else:
                        # If no test data, use the metrics from the model bundle
                        metrics = results.get('metrics', {})
                    
                    # Store results
                    rp_results[f"{region}_{product}"] = {
                        'region': region,
                        'product': product,
                        'results': results,
                        'metrics': metrics,
                        'data_size': len(rp_data)
                    }
                    
                    # Add to summary
                    rp_summary.append({
                        'Region': region,
                        'Product': product,
                        'Data Points': len(rp_data),
                        'SMAPE': f"{metrics.get('SMAPE', 0.0):.3f}",
                        'MAE': f"{metrics.get('MAE', 0.0):.0f}",
                        'RMSE': f"{metrics.get('RMSE', 0.0):.0f}",
                        'R²': f"{metrics.get('R2', 0.0):.3f}"
                    })
                    
                except Exception as e:
                    st.warning(f"⚠️ Failed to train {region}-{product}: {str(e)}")
                    continue
            
            # Store results in session state
            st.session_state.rp_results = rp_results
            st.session_state.rp_summary = rp_summary
            st.session_state.rp_config = config
            st.session_state.rp_trained = True
            
            status_text.text("✅ All region-fuel models trained!")
            main_progress.progress(1.0)
            eta_text.text("✅ Completed!")
            
            _display_region_training_results(rp_results, rp_summary, config)
            
        except Exception as e:
            st.error(f"❌ Region-fuel training failed: {str(e)}")
        finally:
            main_progress.empty()
            status_text.empty()
            eta_text.empty()


def _display_region_training_results(rp_results, rp_summary, config):
    """Display the region-fuel training results and metrics."""
    from helpers import format_metric_value
    
    # Performance statistics - moved to top
    if rp_summary:
        st.markdown("### 📈 Performance Statistics")
        
        # Convert metrics to numeric for statistics
        smapes = [float(row['SMAPE']) for row in rp_summary]
        r2s = [float(row['R²']) for row in rp_summary]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Average SMAPE", format_metric_value(np.mean(smapes), 'SMAPE'))
        with col2:
            st.metric("Best SMAPE", format_metric_value(np.min(smapes), 'SMAPE'))
        with col3:
            st.metric("Average R²", format_metric_value(np.mean(r2s), 'R2'))
        with col4:
            st.metric("Best R²", format_metric_value(np.max(r2s), 'R2'))
    
    st.success(f"✅ Region-Fuel Models Training Completed! ({len(rp_results)} models trained)")
    
    # Summary table
    st.markdown("### 📊 Region-Fuel Model Summary")
    summary_df = pd.DataFrame(rp_summary)
    st.dataframe(summary_df, use_container_width=True)
    
    # Best and worst performers - restructured without main heading
    if rp_summary:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🥇 Best Performers (Lowest SMAPE)")
            best_performers = sorted(rp_summary, key=lambda x: float(x['SMAPE']))[:3]
            for performer in best_performers:
                st.write(f"**{performer['Region']} - {performer['Product']}**: {performer['SMAPE']} SMAPE")
        
        with col2:
            st.markdown("### 🎯 Improvement Opportunities (Highest SMAPE)")
            worst_performers = sorted(rp_summary, key=lambda x: float(x['SMAPE']), reverse=True)[:3]
            for performer in worst_performers:
                st.write(f"**{performer['Region']} - {performer['Product']}**: {performer['SMAPE']} SMAPE")
    
    # Note about individual model results
    st.info("🔍 Individual Model Results have been moved to the 'Region-Fuel Analysis' tab for better organization.")
    
    # Configuration summary
    with st.expander("⚙️ Training Configuration Used", expanded=False):
        # Data split information
        st.markdown("#### 📊 Data Split")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Training Ratio:** {config.train_ratio:.1%}")
            st.write(f"**Split Method:** {config.split_method}")
        with col2:
            st.write(f"**Training Period:** 2014-07-28 to 2023-08-07")
            st.write(f"**Testing Period:** 2023-08-14 to 2025-06-23")
        
        # Model configuration
        st.markdown("#### 🤖 Model Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Feature Selection:** {'Enabled' if config.feature_selection else 'Disabled'}")
            st.write(f"**Selected Features:** {config.k_features if config.feature_selection else 'All'}")
            st.write(f"**Ensemble Method:** {config.ensemble_method}")
        with col2:
            st.write(f"**Models Trained:** {len(rp_results)} individual models")
            st.write(f"**LightGBM Learning Rate:** {config.lgbm_learning_rate}")
            st.write(f"**LightGBM Weight:** {config.lgbm_weight if config.ensemble_method == 'Weighted Average' else 'N/A'}")
