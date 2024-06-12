import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

def load_data(data_dir):
    st.write(f"Loading data from {data_dir}...")
    data = pd.read_csv(os.path.join(data_dir, "merged_trade_indicator_event.csv"))
    st.write(f"Data loaded with shape: {data.shape}")
    return data

def calculate_moving_averages(data):
    # Identify columns that are true moving averages
    ma_columns = [
        '144_SMA_percent_away', '500P_BOLL_LOWER_percent_away', '500P_BOLL_UPPER_percent_away',
        '618_EMA_percent_away', 'MA_ENVELOPES_LOWER_percent_away', 'MA_ENVELOPES_MID_percent_away',
        'MA_ENVELOPES_UPPER_percent_away', 'ZLEMA_13_percent_away', 'ZLEMA_233_percent_away'
    ]
    st.write(f"Filtered MA Columns: {ma_columns}")  # Debug: Print filtered MA Columns
    ma_data = data[ma_columns]
    return ma_data

def clean_data(ma_data):
    # Fill or drop NaN values
    ma_data = ma_data.fillna(method='ffill').fillna(method='bfill')
    st.write("Cleaned MA Data Sample:")
    st.write(ma_data.head())  # Debug: Show a sample of the cleaned data
    return ma_data

def check_convergence(row, ma_columns, threshold):
    for i in range(len(ma_columns) - 1):
        for j in range(i + 1, len(ma_columns)):
            ma1 = row[ma_columns[i]]
            ma2 = row[ma_columns[j]]
            percentage_diff = abs(ma1 - ma2) / ((ma1 + ma2) / 2) * 100
            if percentage_diff > threshold:
                return False
    return True

def check_divergence(row, ma_columns, threshold):
    for i in range(len(ma_columns) - 1):
        for j in range(i + 1, len(ma_columns)):
            ma1 = row[ma_columns[i]]
            ma2 = row[ma_columns[j]]
            percentage_diff = abs(ma1 - ma2) / ((ma1 + ma2) / 2) * 100
            if percentage_diff < threshold:
                return False
    return True

def run_moving_average_convergence():
    if "base_dir" not in st.session_state:
        st.session_state.base_dir = "./data/processed"
    
    data = load_data(st.session_state.base_dir)
    ma_data = calculate_moving_averages(data)
    ma_data = clean_data(ma_data)
    threshold = st.slider("Set Convergence/Divergence Threshold (%)", 0.1, 100.0, 5.0)  # Adjust the range for better exploration

    # List of Moving Average columns
    ma_columns = ma_data.columns.tolist()
    
    # Check for convergence and divergence points
    convergence_points = []
    divergence_points = []

    # Debug: Print percentage differences for the first 10 rows
    for index, row in ma_data.head(10).iterrows():
        for i in range(len(ma_columns) - 1):
            for j in range(i + 1, len(ma_columns)):
                ma1 = row[ma_columns[i]]
                ma2 = row[ma_columns[j]]
                percentage_diff = abs(ma1 - ma2) / ((ma1 + ma2) / 2) * 100
                st.write(f"Row {index}: Comparing {ma_columns[i]} and {ma_columns[j]}: {percentage_diff}%")

    for index, row in ma_data.iterrows():
        if check_convergence(row, ma_columns, threshold):
            convergence_points.append((index, data.loc[index, 'price']))
        if check_divergence(row, ma_columns, threshold):
            divergence_points.append((index, data.loc[index, 'price']))

    st.write(f"Found {len(convergence_points)} convergence points.")
    st.write(f"Found {len(divergence_points)} divergence points.")

    if convergence_points:
        convergence_df = pd.DataFrame(convergence_points, columns=['Index', 'Price'])
        st.write("Convergence Points:")
        st.write(convergence_df.head())  # Display the first few convergence points

    if divergence_points:
        divergence_df = pd.DataFrame(divergence_points, columns=['Index', 'Price'])
        st.write("Divergence Points:")
        st.write(divergence_df.head())  # Display the first few divergence points

    # Visualize the convergence and divergence points
    fig = px.scatter(data, x='time', y='price', title='Price vs Time with Convergence and Divergence Points')
    if convergence_points:
        fig.add_scatter(x=data.loc[convergence_df['Index'], 'time'], y=convergence_df['Price'], mode='markers', name='Convergence Points', marker=dict(color='green'))
    if divergence_points:
        fig.add_scatter(x=data.loc[divergence_df['Index'], 'time'], y=divergence_df['Price'], mode='markers', name='Divergence Points', marker=dict(color='red'))
    st.plotly_chart(fig)

if __name__ == "__main__":
    run_moving_average_convergence()
