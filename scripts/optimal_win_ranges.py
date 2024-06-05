import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm
import os

def get_file_path(base_dir, relative_path):
    return os.path.join(base_dir, relative_path)

def calculate_optimal_win_ranges(data, target='result', features=None):
    optimal_ranges = []

    if features is None:
        features = data.columns.drop([target])

    for feature in tqdm(features, desc="Calculating Optimal Win Ranges"):
        data[feature] = pd.to_numeric(data[feature], errors='coerce')
        
        win_values = data[data[target] == 0][feature].dropna().values.astype(float)
        loss_values = data[data[target] == 1][feature].dropna().values.astype(float)

        if len(win_values) == 0 or len(loss_values) == 0:
            continue

        win_kde = gaussian_kde(win_values)
        loss_kde = gaussian_kde(loss_values)

        x_grid = np.linspace(min(data[feature].dropna()), max(data[feature].dropna()), 1000)
        win_density = win_kde(x_grid)
        loss_density = loss_kde(x_grid)

        is_winning = win_density > loss_density
        ranges = []
        start_idx = None

        for i in range(len(is_winning)):
            if is_winning[i] and start_idx is None:
                start_idx = i
            elif not is_winning[i] and start_idx is not None:
                ranges.append((x_grid[start_idx], x_grid[i - 1]))
                start_idx = None
        if start_idx is not None:
            ranges.append((x_grid[start_idx], x_grid[-1]))

        optimal_ranges.append({
            'feature': feature,
            'optimal_win_ranges': ranges
        })

    return optimal_ranges

def plot_optimal_win_ranges(data, optimal_ranges, target='result', trade_type='', model_name=''):
    for item in optimal_ranges:
        feature = item['feature']
        ranges = item['optimal_win_ranges']
        
        win_values = data[data[target] == 0][feature].dropna()
        loss_values = data[data[target] == 1][feature].dropna()
        
        plt.figure(figsize=(12, 6))
        sns.histplot(win_values, color='blue', label='Win', kde=True, stat='density', element='step', fill=True)
        sns.histplot(loss_values, color='red', label='Loss', kde=True, stat='density', element='step', fill=True)
        
        for range_start, range_end in ranges:
            plt.axvspan(range_start, range_end, color='blue', alpha=0.3)

        plt.title(f'Optimal Win Ranges for {feature} ({trade_type}, {model_name})')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()

def summarize_optimal_win_ranges(optimal_ranges):
    summary = []
    for item in optimal_ranges:
        feature = item['feature']
        for range_start, range_end in item['optimal_win_ranges']:
            summary.append({
                'feature': feature,
                'optimal_win_range_start': range_start,
                'optimal_win_range_end': range_end
            })
    return pd.DataFrame(summary)

def load_data(data_path):
    data = pd.read_csv(data_path)
    data['result'] = data['result'].map({'win': 0, 'loss': 1})
    
    for col in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            data[col] = data[col].apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
    
    return data

def run_optimal_win_ranges():
    st.subheader("Calculate Optimal Win Ranges")

    base_dir = st.text_input("Base Directory", "C:/Users/Administrator/Desktop/nocodeML/streamlit_app")
    data_path = get_file_path(base_dir, "data/processed/merged_trade_indicator_event.csv")

    if 'loaded_data' not in st.session_state:
        st.session_state.loaded_data = False

    if st.button("Load Data"):
        data = load_data(data_path)
        filtered_indicators = [col for col in data.columns if 'qty' not in col and 'price' not in col and '_percent_away' not in col and '_binary' not in col and col not in ['result', 'time', 'event', 'event_event', 'amount']]
        
        feature_importances = {
            'Random Forest': pd.DataFrame({'feature': filtered_indicators, 'importance': np.random.rand(len(filtered_indicators))}),
            'Gradient Boosting': pd.DataFrame({'feature': filtered_indicators, 'importance': np.random.rand(len(filtered_indicators))}),
            'XGBoost': pd.DataFrame({'feature': filtered_indicators, 'importance': np.random.rand(len(filtered_indicators))}),
            'LightGBM': pd.DataFrame({'feature': filtered_indicators, 'importance': np.random.rand(len(filtered_indicators))})
        }

        st.session_state.data = data
        st.session_state.filtered_indicators = filtered_indicators
        st.session_state.feature_importances = feature_importances
        st.session_state.loaded_data = True
        st.success("Data Loaded Successfully")

    if st.session_state.loaded_data:
        trade_type = st.selectbox("Select Trade Type", ["Long Only", "Short Only", "Long & Short"])
        model_name = st.selectbox("Select Model", ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"])
        top_n = st.selectbox("Select Top N Indicators", [None, 10, 5, 3, "ALL"])
        individual_indicator = st.selectbox("Select Individual Indicator", st.session_state.filtered_indicators)

        if st.button("Calculate and Plot Optimal Win Ranges"):
            if top_n:
                selected_features = st.session_state.feature_importances[model_name]['feature'] if top_n == "ALL" else st.session_state.feature_importances[model_name]['feature'].head(top_n)
            else:
                selected_features = [individual_indicator]

            optimal_ranges = calculate_optimal_win_ranges(st.session_state.data, features=selected_features)
            plot_optimal_win_ranges(st.session_state.data, optimal_ranges, trade_type=trade_type, model_name=model_name)
            
            optimal_win_ranges_summary = summarize_optimal_win_ranges(optimal_ranges)
            st.write(optimal_win_ranges_summary)
            
            output_path = get_file_path(base_dir, f'docs/ml_analysis/win_ranges_summary/optimal_win_ranges_summary_{model_name}.csv')
            optimal_win_ranges_summary.to_csv(output_path, index=False)
            st.write(f"Saved optimal win ranges summary to {output_path}")
