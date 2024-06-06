import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import streamlit as st
import pandas as pd
import os

def get_file_path(base_dir, relative_path):
    return os.path.join(base_dir, relative_path)

def identify_winning_ranges(data, feature, target='result'):
    win_values = data[data[target] == 0][feature].dropna()
    loss_values = data[data[target] == 1][feature].dropna()
    
    if len(win_values) < 2 or len(loss_values) < 2:
        st.write(f"Not enough data points for KDE for feature: {feature}")
        return None
    
    kde_win = gaussian_kde(win_values)
    kde_loss = gaussian_kde(loss_values)
    
    min_value = min(win_values.min(), loss_values.min())
    max_value = max(win_values.max(), loss_values.max())
    x = np.linspace(min_value, max_value, 1000)
    
    kde_win_values = kde_win(x)
    kde_loss_values = kde_loss(x)
    
    winning_ranges = []
    in_range = False
    range_start = None
    
    for i in range(len(x)):
        if kde_win_values[i] > kde_loss_values[i]:
            if not in_range:
                range_start = x[i]
                in_range = True
        else:
            if in_range:
                winning_ranges.append((range_start, x[i]))
                in_range = False
    
    if in_range:
        winning_ranges.append((range_start, x[-1]))
    
    plt.figure(figsize=(12, 6))
    sns.histplot(win_values, color='blue', label='Win', kde=True, stat='density', element='step', fill=True)
    sns.histplot(loss_values, color='red', label='Loss', kde=True, stat='density', element='step', fill=True)
    
    for (start, end) in winning_ranges:
        plt.axvspan(start, end, color='green', alpha=0.3)
    
    plt.title(f'Optimal Win Ranges for {feature}')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend()
    st.pyplot(plt.gcf())
    plt.close()
    
    return winning_ranges

def run_win_ranges_specific_model():
    st.subheader("Win Ranges for Specific Model")
    
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
        percent_away_features = [feat for feat in merged_data.columns if "_percent_away" in feat]

        # Debugging: Check the first few rows and the result distribution
        st.write("First few rows of the loaded data:")
        st.write(merged_data.head())

        st.write("Distribution of 'result' column:")
        st.write(merged_data['result'].value_counts())

        # Ensure result column is correctly mapped
        merged_data['result'] = merged_data['result'].map({'win': 0, 'loss': 1})

        st.session_state.merged_data = merged_data
        st.session_state.percent_away_features = percent_away_features
        st.session_state.loaded_data = True
        st.success("Data Loaded Successfully")

    if st.session_state.loaded_data:
        feature = st.selectbox("Select Feature", st.session_state.percent_away_features)

        if st.button("Identify Winning Ranges"):
            ranges = identify_winning_ranges(st.session_state.merged_data, feature)
            if ranges:
                st.write(f"{feature}: {ranges}")

