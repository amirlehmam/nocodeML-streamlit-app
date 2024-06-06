import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st

def get_file_path(base_dir, relative_path):
    return os.path.join(base_dir, relative_path)

def deep_dive_feature_analysis(data, feature, target='result', save_path=None):
    if feature not in data.columns or target not in data.columns:
        st.write(f"Feature '{feature}' or target '{target}' not found in data columns")
        return
    
    data[feature] = data[feature].fillna(data[feature].mean())
    
    win_values = data[data[target] == 0][feature]
    loss_values = data[data[target] == 1][feature]
    
    st.write(f"Feature: {feature}")
    st.write(f"Number of win values: {len(win_values)}")
    st.write(f"Number of loss values: {len(loss_values)}")
    
    if len(win_values) == 0:
        st.write(f"No win values found for feature: {feature}")
    if len(loss_values) == 0:
        st.write(f"No loss values found for feature: {feature}")
    
    if len(win_values) < 2 or len(loss_values) < 2:
        st.write(f"Not enough data points for KDE for feature: {feature}")
        return
    
    t_stat, p_value = stats.ttest_ind(win_values, loss_values, equal_var=False)
    st.write(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.3f}")
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        f'Histogram and KDE of {feature}',
        f'Box Plot of {feature} by {target}',
        f'KDE Plot of {feature} for Win and Loss',
        f'Violin Plot of {feature} by {target}'
    ))
    
    hist_data = data[[feature, target]]
    fig.add_trace(
        go.Histogram(x=hist_data[feature], nbinsx=50, histnorm='density', name=f'{feature} Histogram'),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=hist_data[feature], nbinsx=50, histnorm='density', cumulative_enabled=True, name=f'{feature} Cumulative'),
        row=1, col=1
    )
    
    kde_win = go.Scatter(x=win_values, y=stats.gaussian_kde(win_values)(win_values), mode='lines', name='Win KDE')
    kde_loss = go.Scatter(x=loss_values, y=stats.gaussian_kde(loss_values)(loss_values), mode='lines', name='Loss KDE')
    fig.add_trace(kde_win, row=1, col=1)
    fig.add_trace(kde_loss, row=1, col=1)
    
    fig.add_trace(
        go.Box(y=win_values, name='Win'),
        row=1, col=2
    )
    fig.add_trace(
        go.Box(y=loss_values, name='Loss'),
        row=1, col=2
    )
    
    fig.add_trace(kde_win, row=2, col=1)
    fig.add_trace(kde_loss, row=2, col=1)
    
    fig.add_trace(
        go.Violin(y=win_values, name='Win', box_visible=True, meanline_visible=True),
        row=2, col=2
    )
    fig.add_trace(
        go.Violin(y=loss_values, name='Loss', box_visible=True, meanline_visible=True),
        row=2, col=2
    )
    
    
    fig.update_layout(
        title_text=f'Deep Dive Analysis of {feature}',
        height=800
    )
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        fig.write_html(os.path.join(save_path, f'{feature}_deep_dive.html'))
    st.plotly_chart(fig)

def run_advanced_eda_specific_model():
    st.subheader("Advanced EDA on Specific Model")

    if "base_dir" not in st.session_state:
        st.session_state.base_dir = "."

    base_dir = st.text_input("Base Directory", value=st.session_state.base_dir)
    data_path = get_file_path(base_dir, "data/processed/merged_trade_indicator_event.csv")

    if 'loaded_data' not in st.session_state:
        st.session_state.loaded_data = False

    if st.button("Load Data"):
        if not os.path.exists(data_path):
            st.write("Data file not found. Please check the file path.")
            return

        merged_data = pd.read_csv(data_path)

        # Ensure the result column is properly mapped
        merged_data['result'] = merged_data['result'].map({'win': 0, 'loss': 1})

        percent_away_features = [feat for feat in merged_data.columns if "_percent_away" in feat]

        st.session_state.merged_data = merged_data
        st.session_state.percent_away_features = percent_away_features
        st.session_state.loaded_data = True
        st.success("Data Loaded Successfully")

    if st.session_state.loaded_data:
        feature = st.selectbox("Select Feature", st.session_state.percent_away_features)

        if st.button("Run Deep Dive Analysis"):
            analysis_save_path = get_file_path(base_dir, "data/feature_deep_dive_analysis")
            deep_dive_feature_analysis(st.session_state.merged_data, feature, save_path=analysis_save_path)

# Make sure to call the function to run the app
if __name__ == "__main__":
    run_advanced_eda_specific_model()
