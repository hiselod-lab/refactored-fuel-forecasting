"""
Model Training Module
Handles the Overall Model training functionality for the fuel forecasting app.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from helpers import (
    ModelConfig,
    time_series_cv_scores,
    plot_actual_vs_predicted,
    prepare_df_for_display
)


def show_overall_model_training(weekly_feats: pd.DataFrame, enhanced_feature_engineering, train_models, evaluate_preds, generate_forecast, generate_detailed_forecast, create_metrics_dashboard) -> None:
    """
    Render the complete Overall Model training tab.
    
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
    with st.expander("⚙️ Model Parameters & Training Configuration", expanded=False):
        config = _render_model_parameters()
    
    # Training Button and Results
    if st.button("🚀 Train Overall Model", type="primary", key="train_overall"):
        _run_overall_training(weekly_feats, config, enhanced_feature_engineering, train_models, evaluate_preds, generate_forecast, generate_detailed_forecast, create_metrics_dashboard)
    
    # Display existing results if available
    elif st.session_state.get('overall_trained', False):
        results = st.session_state.get('overall_results')
        metrics = st.session_state.get('overall_metrics')
        cv_metrics = st.session_state.get('overall_cv_metrics')
        saved_config = st.session_state.get('overall_config')
        
        if results and metrics and cv_metrics and saved_config:
            _display_training_results(results, metrics, cv_metrics, saved_config)


def _render_model_parameters() -> ModelConfig:
    """Render the model parameters form and return ModelConfig."""
    st.markdown("### Customize Model Parameters")
    
    # Use a form to prevent automatic reruns when UI components change
    with st.form(key="overall_model_form"):
        # Train-Test Split Configuration
        st.markdown("#### 📊 Train-Test Split Configuration")
        col1, col2 = st.columns(2)
        with col1:
            train_ratio = st.slider("Training Data Ratio", min_value=0.6, max_value=0.9, value=0.8, step=0.05,
                                   help="Proportion of data used for training (rest for testing)")
        with col2:
            split_method = st.radio("Split Method", ["Time-based", "Random"], index=0,
                                   help="Time-based: chronological split | Random: shuffled split")
        
        # Feature Selection
        st.markdown("#### 🎯 Feature Selection")
        col1, col2 = st.columns(2)
        with col1:
            feature_selection = st.checkbox("Enable Feature Selection", value=True,
                                          help="Use LightGBM feature importance to select most relevant features")
        with col2:
            k_features = st.number_input("Number of Features (if enabled)", min_value=5, max_value=50, value=20,
                                       help="Number of top features to select")
        
        # LightGBM Parameters
        st.markdown("#### 🌟 LightGBM Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            lgbm_learning_rate = st.select_slider("Learning Rate", 
                                                 options=[0.001, 0.01, 0.05, 0.1, 0.2], 
                                                 value=0.01,
                                                 help="Controls step size in gradient descent")
        with col2:
            lgbm_n_estimators = st.select_slider("Number of Trees", 
                                                options=[100, 500, 1000, 1500, 2000], 
                                                value=1000,
                                                help="Number of boosting rounds")
        with col3:
            lgbm_max_depth = st.select_slider("Max Tree Depth", 
                                             options=[3, 5, 7, 10, 15], 
                                             value=7,
                                             help="Maximum depth of individual trees")
        
        # Random Forest Parameters
        st.markdown("#### 🌳 Random Forest Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            rf_n_estimators = st.select_slider("Number of Trees (RF)", 
                                              options=[50, 100, 200, 300, 500], 
                                              value=200,
                                              help="Number of trees in the forest")
        with col2:
            rf_max_depth = st.select_slider("Max Tree Depth (RF)", 
                                           options=[5, 10, 15, 20, None], 
                                           value=10,
                                           help="Maximum depth of trees (None = unlimited)")
        with col3:
            rf_min_samples_split = st.select_slider("Min Samples Split", 
                                                   options=[2, 5, 10, 20], 
                                                   value=2,
                                                   help="Minimum samples required to split a node")
        
        # Ensemble Configuration
        st.markdown("#### 🤝 Ensemble Configuration")
        col1, col2 = st.columns(2)
        with col1:
            ensemble_method = st.radio("Ensemble Method", 
                                     ["Average", "Weighted Average"], 
                                     index=0,
                                     help="How to combine LightGBM and Random Forest predictions")
        with col2:
            lgbm_weight = st.slider("LightGBM Weight (if Weighted)", 
                                   min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                                   help="Weight for LightGBM in weighted ensemble (RF gets 1-weight)")
        
        # Submit button for the form
        submitted = st.form_submit_button("💾 Save Configuration", type="secondary")
    
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


def _run_overall_training(weekly_feats, config, enhanced_feature_engineering, train_models, evaluate_preds, generate_forecast, generate_detailed_forecast, create_metrics_dashboard):
    """Execute the overall model training process."""
    import time
    
    # Create layout with status and ETA side by side
    progress_bar = st.progress(0)
    status_col, eta_col = st.columns([3, 1])
    with status_col:
        status_text = st.empty()
    with eta_col:
        eta_text = st.empty()
    
    start_time = time.time()
    
    # Define training steps with realistic estimated durations
    steps = [
        ("🔧 Engineering features...", 0.15, 5),    # 15% of total, ~5 seconds
        ("🤖 Training models...", 0.60, 25),        # 60% of total, ~25 seconds  
        ("📊 Evaluating performance...", 0.85, 8),  # 85% of total, ~8 seconds
        ("🔄 Running cross-validation...", 1.0, 12) # 100% of total, ~12 seconds
    ]
    
    total_estimated_time = sum(step[2] for step in steps)  # Total estimated time
    
    def update_status_and_eta(current_step, step_progress=0):
        """Update status and ETA side by side with proper decreasing calculation."""
        elapsed = time.time() - start_time
        
        # Update status
        if current_step < len(steps):
            status_text.text(f"🔄 Training Overall Model... {steps[current_step][0]}")
        else:
            status_text.text("✅ Training completed!")
        
        # Calculate and update ETA
        if current_step >= len(steps):
            eta_text.text("✅ Completed!")
            return
            
        # Calculate remaining time based on current progress and elapsed time
        if elapsed < 2:
            # Initial estimate
            eta_text.text(f"⏱️ ETA: ~{total_estimated_time}s")
        else:
            # Calculate progress-based ETA
            if current_step < len(steps):
                prev_progress = steps[current_step-1][1] if current_step > 0 else 0
                current_step_weight = steps[current_step][1] - prev_progress
                total_progress = prev_progress + (current_step_weight * step_progress)
            else:
                total_progress = 1.0
            
            if total_progress > 0.05:  # After 5% completion
                # Use elapsed time to estimate total time, then calculate remaining
                estimated_total_time = elapsed / total_progress
                remaining_time = max(0, estimated_total_time - elapsed)
                
                # Format ETA display
                if remaining_time > 60:
                    eta_text.text(f"⏱️ ETA: {int(remaining_time//60)}m {int(remaining_time%60)}s")
                elif remaining_time > 5:
                    eta_text.text(f"⏱️ ETA: {int(remaining_time)}s")
                elif remaining_time > 0:
                    eta_text.text("⏱️ ETA: Almost done...")
                else:
                    eta_text.text("⏱️ ETA: Finishing...")
            else:
                eta_text.text(f"⏱️ ETA: ~{total_estimated_time}s")
        
    try:
        # Step 1: Feature Engineering
        progress_bar.progress(int(steps[0][1] * 100))
        update_status_and_eta(0, 0.1)
        
        feature_engineered = enhanced_feature_engineering(weekly_feats, forecasting_mode=False, random_state=42)
        feature_cols = [col for col in feature_engineered.columns 
                      if col not in ['Region', 'Product', 'week_start', 'sales_volume', 'sales_amount']]
        
        update_status_and_eta(0, 1.0)
        
        # Step 2: Model Training
        progress_bar.progress(int(steps[1][1] * 100))
        update_status_and_eta(1, 0.1)
        
        results = train_models(feature_engineered, feature_cols, config)
        
        update_status_and_eta(1, 1.0)
        
        # Step 3: Model Evaluation
        progress_bar.progress(int(steps[2][1] * 100))
        update_status_and_eta(2, 0.1)
        
        # Extract y_test and y_pred from the nested structure
        test_data = results.get('test_data', {})
        y_test = test_data.get('y_test', pd.Series())
        y_pred = test_data.get('y_pred', pd.Series())
        
        if len(y_test) > 0 and len(y_pred) > 0:
            metrics = evaluate_preds(y_test, y_pred)
        else:
            # If no test data, use the metrics from the model bundle
            metrics = results.get('metrics', {})
        
        update_status_and_eta(2, 1.0)
        
        # Step 4: Cross-validation
        progress_bar.progress(int(steps[3][1] * 100))
        update_status_and_eta(3, 0.1)
        
        def fit_predict_fn(X_tr, y_tr, X_te):
            # Create a temporary dataframe for training
            temp_df = pd.concat([X_tr, y_tr], axis=1)
            temp_df['sales_volume'] = y_tr
            
            temp_results = train_models(temp_df, feature_cols, config)
            
            # Extract the models from the results
            models = temp_results.get('models', {})
            if 'lgbm' in models and 'rf' in models:
                # Get predictions from both models
                lgbm_pred = models['lgbm'].predict(X_te[feature_cols])
                rf_pred = models['rf'].predict(X_te[feature_cols])
                
                # Create ensemble prediction
                if temp_results.get('ensemble_method') == 'Weighted Average':
                    weight = temp_results.get('lgbm_weight', 0.5)
                    ensemble_pred = weight * lgbm_pred + (1 - weight) * rf_pred
                else:
                    ensemble_pred = (lgbm_pred + rf_pred) / 2
                
                return ensemble_pred
            else:
                # Fallback to first available model
                for model_name, model in models.items():
                    return model.predict(X_te[feature_cols])
                return np.zeros(len(X_te))
        
        # Use the test data for cross-validation if available
        if len(y_test) > 0 and 'test_df' in test_data:
            try:
                cv_metrics = time_series_cv_scores(
                    test_data['test_df'][feature_cols], 
                    y_test, 
                    fit_predict_fn
                )
            except Exception as e:
                st.warning(f"Cross-validation failed: {str(e)}")
                cv_metrics = {'SMAPE': 0.0, 'MAE': 0.0, 'RMSE': 0.0, 'R2': 0.0}
        else:
            cv_metrics = {'SMAPE': 0.0, 'MAE': 0.0, 'RMSE': 0.0, 'R2': 0.0}
        
        update_status_and_eta(3, 1.0)
        
        # Step 5: Display Results
        progress_bar.progress(100)
        update_status_and_eta(4, 1.0)
        
        # Store results in session state with config
        st.session_state.overall_results = results
        st.session_state.overall_metrics = metrics
        st.session_state.overall_cv_metrics = cv_metrics
        st.session_state.overall_config = config
        st.session_state.overall_trained = True
        
        _display_training_results(results, metrics, cv_metrics, config)
        
    except Exception as e:
        st.error(f"❌ Training failed: {str(e)}")
    finally:
        progress_bar.empty()
        status_text.empty()
        eta_text.empty()


def _display_training_results(results, metrics, cv_metrics, config):
    """Display the training results and metrics."""
    from helpers import format_metric_value, get_metric_info
    
    st.success("✅ Overall Model Training Completed Successfully!")
    
    # Display metrics in multiple columns to save space
    st.markdown("### 📊 Model Performance Metrics")
    
    # Create two columns for side-by-side display
    metrics_col1, metrics_col2 = st.columns(2)
    
    # Single Split Results
    with metrics_col1:
        st.markdown("#### 📈 Single Split Results")
        single_split_df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': [format_metric_value(value, key) for key, value in metrics.items()]
        })
        st.dataframe(single_split_df, use_container_width=True, hide_index=True)
    
    # Cross-Validation Results
    with metrics_col2:
        st.markdown("#### 🔄 Cross-Validation Results")
        cv_results_df = pd.DataFrame({
            'Metric': list(cv_metrics.keys()),
            'Value': [format_metric_value(value, key) for key, value in cv_metrics.items()]
        })
        st.dataframe(cv_results_df, use_container_width=True, hide_index=True)
    
    # Model comparison
    st.markdown("### 🏆 Model Comparison")
    
    # Format model comparison metrics
    lgbm_smape = results.get('lgbm_metrics', {}).get('SMAPE', 0)
    rf_smape = results.get('rf_metrics', {}).get('SMAPE', 0)
    ensemble_smape = metrics.get('SMAPE', 0)
    
    lgbm_r2 = results.get('lgbm_metrics', {}).get('R2', 0)
    rf_r2 = results.get('rf_metrics', {}).get('R2', 0)
    ensemble_r2 = metrics.get('R2', 0)
    
    model_comparison = pd.DataFrame({
        'Model': ['LightGBM', 'Random Forest', 'Ensemble'],
        'SMAPE': [
            format_metric_value(lgbm_smape, 'SMAPE') if lgbm_smape else 'N/A',
            format_metric_value(rf_smape, 'SMAPE') if rf_smape else 'N/A',
            format_metric_value(ensemble_smape, 'SMAPE') if ensemble_smape else 'N/A'
        ],
        'R²': [
            format_metric_value(lgbm_r2, 'R2') if lgbm_r2 else 'N/A',
            format_metric_value(rf_r2, 'R2') if rf_r2 else 'N/A',
            format_metric_value(ensemble_r2, 'R2') if ensemble_r2 else 'N/A'
        ]
    })
    st.dataframe(model_comparison, use_container_width=True)
    
    # Best performer highlights with compact styling
    st.markdown("### 🏆 Best Performer Highlights")
    
    best_smape_idx = min(range(3), key=lambda i: [lgbm_smape, rf_smape, ensemble_smape][i] if [lgbm_smape, rf_smape, ensemble_smape][i] > 0 else float('inf'))
    best_r2_idx = max(range(3), key=lambda i: [lgbm_r2, rf_r2, ensemble_r2][i])
    
    highlight_cols = st.columns(2)
    
    with highlight_cols[0]:
        best_model = model_comparison.iloc[best_smape_idx]['Model']
        best_smape_value = model_comparison.iloc[best_smape_idx]['SMAPE']
        st.markdown(f"""
        <div style='text-align: center; padding: 15px; background-color: #e8f5e8; border-radius: 10px; border-left: 4px solid #4caf50; margin: 5px;'>
            <div style='font-size: 32px; margin-bottom: 8px;'>🎯</div>
            <h3 style='color: #2e7d32; margin: 0; font-size: 18px; font-weight: bold;'>{best_smape_value}</h3>
            <p style='margin: 2px 0; font-size: 12px; font-weight: 600; color: #333;'>Best SMAPE - {best_model}</p>
            <p style='margin: 0; font-size: 9px; color: #666; line-height: 1.2;'>Lowest prediction error</p>
        </div>
        """, unsafe_allow_html=True)
        
    with highlight_cols[1]:
        best_r2_model = model_comparison.iloc[best_r2_idx]['Model']
        best_r2_value = model_comparison.iloc[best_r2_idx]['R²']
        st.markdown(f"""
        <div style='text-align: center; padding: 15px; background-color: #fff3e0; border-radius: 10px; border-left: 4px solid #ff9800; margin: 5px;'>
            <div style='font-size: 32px; margin-bottom: 8px;'>🏆</div>
            <h3 style='color: #e65100; margin: 0; font-size: 18px; font-weight: bold;'>{best_r2_value}</h3>
            <p style='margin: 2px 0; font-size: 12px; font-weight: 600; color: #333;'>Best R² - {best_r2_model}</p>
            <p style='margin: 0; font-size: 9px; color: #666; line-height: 1.2;'>Highest variance explained</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Actual vs Predicted plots have been removed as requested
    
    # Feature importance (if available)
    if 'feature_importance' in results:
        st.markdown("### 🎯 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': results['feature_importance'].keys(),
            'Importance': results['feature_importance'].values()
        }).sort_values('Importance', ascending=False).head(15)
        
        st.bar_chart(importance_df.set_index('Feature'))
    
    # Configuration summary
    with st.expander("⚙️ Training Configuration Used", expanded=False):
        # Calculate training and testing dates from the data
        try:
            # Get the data from session state or calculate from results
            if 'weekly_feats' in st.session_state:
                weekly_data = st.session_state.weekly_feats
                total_records = len(weekly_data)
                split_point = int(total_records * config.train_ratio)
                
                # Sort data by date to get correct periods
                sorted_data = weekly_data.sort_values('week_start')
                train_start_date = sorted_data.iloc[0]['week_start']
                train_end_date = sorted_data.iloc[split_point - 1]['week_start']
                test_start_date = sorted_data.iloc[split_point]['week_start']
                test_end_date = sorted_data.iloc[-1]['week_start']
                
                # Format dates
                train_start_str = train_start_date.strftime('%Y-%m-%d') if hasattr(train_start_date, 'strftime') else str(train_start_date)
                train_end_str = train_end_date.strftime('%Y-%m-%d') if hasattr(train_end_date, 'strftime') else str(train_end_date)
                test_start_str = test_start_date.strftime('%Y-%m-%d') if hasattr(test_start_date, 'strftime') else str(test_start_date)
                test_end_str = test_end_date.strftime('%Y-%m-%d') if hasattr(test_end_date, 'strftime') else str(test_end_date)
                
                training_period = f"{train_start_str} to {train_end_str}"
                testing_period = f"{test_start_str} to {test_end_str}"
            else:
                # Fallback values if data is not available
                training_period = "2014-07-28 to 2023-08-07"
                testing_period = "2023-08-14 to 2025-06-23"
        except Exception:
            # Fallback values in case of any error
            training_period = "2014-07-28 to 2023-08-07"
            testing_period = "2023-08-14 to 2025-06-23"
        
        # Configuration table with training and testing dates
        config_dict = {
            'Train Ratio': f"{config.train_ratio:.1%}",
            'Split Method': config.split_method,
            'Training Period': training_period,
            'Testing Period': testing_period,
            'Feature Selection': 'Enabled' if config.feature_selection else 'Disabled',
            'Selected Features': config.k_features if config.feature_selection else 'All',
            'LightGBM Learning Rate': config.lgbm_learning_rate,
            'LightGBM Trees': config.lgbm_n_estimators,
            'Random Forest Trees': config.rf_n_estimators,
            'Ensemble Method': config.ensemble_method
        }
        
        config_df = pd.DataFrame(list(config_dict.items()), columns=['Parameter', 'Value'])
        st.dataframe(config_df, use_container_width=True, hide_index=True)
