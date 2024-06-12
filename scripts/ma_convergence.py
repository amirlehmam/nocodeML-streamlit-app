import os
import pandas as pd
import streamlit as st
import plotly.express as px

def load_data(data_dir):
    st.write(f"Loading data from {data_dir}...")
    data = pd.read_csv(os.path.join(data_dir, "merged_trade_indicator_event.csv"))
    st.write(f"Data loaded with shape: {data.shape}")
    return data

def calculate_moving_averages(data):
    ma_columns = [
        '144_SMA_percent_away', '500P_BOLL_LOWER_percent_away', '500P_BOLL_UPPER_percent_away',
        '618_EMA_percent_away', 'MA_ENVELOPES_LOWER_percent_away', 'MA_ENVELOPES_MID_percent_away',
        'MA_ENVELOPES_UPPER_percent_away', 'ZLEMA_13_percent_away', 'ZLEMA_233_percent_away'
    ]
    st.write(f"Filtered MA Columns: {ma_columns}")
    ma_data = data[ma_columns]
    return ma_data

def clean_data(ma_data):
    ma_data = ma_data.fillna(method='ffill').fillna(method='bfill')
    st.write("Cleaned MA Data Sample:")
    st.write(ma_data.head())
    return ma_data

def find_convergence_divergence(ma_data, data, threshold):
    convergence_points = []
    divergence_points = []
    
    for index, row in ma_data.iterrows():
        min_val = row.min()
        max_val = row.max()
        if (max_val - min_val) / max_val * 100 <= threshold:
            convergence_points.append((index, data.loc[index, 'price']))
        elif (max_val - min_val) / max_val * 100 >= threshold:
            divergence_points.append((index, data.loc[index, 'price']))

    return convergence_points, divergence_points

def run_moving_average_convergence():
    if "base_dir" not in st.session_state:
        st.session_state.base_dir = "./data/processed"
    
    data = load_data(st.session_state.base_dir)
    ma_data = calculate_moving_averages(data)
    ma_data = clean_data(ma_data)
    threshold = st.slider("Set Convergence/Divergence Threshold (%)", 0.1, 100.0, 5.0)

    convergence_points, divergence_points = find_convergence_divergence(ma_data, data, threshold)

    st.write(f"Found {len(convergence_points)} convergence points.")
    st.write(f"Found {len(divergence_points)} divergence points.")

    if convergence_points:
        convergence_df = pd.DataFrame(convergence_points, columns=['Index', 'Price'])
        st.write("Convergence Points:")
        st.write(convergence_df.head())

    if divergence_points:
        divergence_df = pd.DataFrame(divergence_points, columns=['Index', 'Price'])
        st.write("Divergence Points:")
        st.write(divergence_df.head())

    fig = px.scatter(data, x='time', y='price', title='Price vs Time with Convergence and Divergence Points')
    if convergence_points:
        fig.add_scatter(x=data.loc[convergence_df['Index'], 'time'], y=convergence_df['Price'], mode='markers', name='Convergence Points', marker=dict(color='green'))
    if divergence_points:
        fig.add_scatter(x=data.loc[divergence_df['Index'], 'time'], y=divergence_df['Price'], mode='markers', name='Divergence Points', marker=dict(color='red'))
    st.plotly_chart(fig)

if __name__ == "__main__":
    run_moving_average_convergence()
