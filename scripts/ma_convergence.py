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

def check_convergence(ma_data, ma_columns, threshold):
    convergence_points = []
    for index, row in ma_data.iterrows():
        convergence = True
        for i in range(len(ma_columns) - 1):
            for j in range(i + 1, len(ma_columns)):
                ma1 = row[ma_columns[i]]
                ma2 = row[ma_columns[j]]
                percentage_diff = abs(ma1 - ma2) / ((ma1 + ma2) / 2) * 100
                if percentage_diff > threshold:
                    convergence = False
                    break
            if not convergence:
                break
        if convergence:
            convergence_points.append(index)
    return convergence_points

def run_moving_average_convergence():
    if "base_dir" not in st.session_state:
        st.session_state.base_dir = "./data/processed"
    
    data = load_data(st.session_state.base_dir)
    ma_data = calculate_moving_averages(data)
    ma_data = clean_data(ma_data)
    threshold = st.slider("Set Convergence Threshold (%)", 0.1, 20.0, 5.0)  # Adjust the range for better exploration

    # List of Moving Average columns
    ma_columns = ma_data.columns.tolist()
    
    # Check for convergence points
    convergence_points = check_convergence(ma_data, ma_columns, threshold)

    st.write(f"Found {len(convergence_points)} convergence points.")
    if convergence_points:
        convergence_df = data.loc[convergence_points, ['time', 'price']]
        st.write(convergence_df.head())  # Display the first few convergence points

        # Visualize the convergence points
        fig = px.scatter(data, x='time', y='price', title='Price vs Time with Convergence Points')
        fig.add_scatter(x=convergence_df['time'], y=convergence_df['price'], mode='markers', name='Convergence Points')
        st.plotly_chart(fig)
    else:
        st.write("No convergence points found.")

if __name__ == "__main__":
    run_moving_average_convergence()
