import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

def run_moving_average_convergence():
    def load_data(data_dir):
        st.write(f"Loading data from {data_dir}...")
        data = pd.read_csv(os.path.join(data_dir, "merged_trade_indicator_event.csv"))
        st.write(f"Data loaded with shape: {data.shape}")
        return data

    def calculate_moving_averages(data):
        ma_columns = [col for col in data.columns if 'percent_away' in col]
        ma_data = data[ma_columns]
        return ma_data

    def check_convergence(ma_data, threshold=0.01):
        convergence_points = []
        for index, row in ma_data.iterrows():
            ma_values = row.values
            if np.max(ma_values) - np.min(ma_values) <= threshold:
                convergence_points.append((index, ma_values))
        return convergence_points

    def visualize_convergence(data, convergence_points):
        fig = px.scatter(data, x='time', y='price', title='Price vs Time with Convergence Points')
        convergence_indices = [point[0] for point in convergence_points]
        convergence_prices = data.loc[convergence_indices, 'price']
        fig.add_scatter(x=data.loc[convergence_indices, 'time'], y=convergence_prices, mode='markers', name='Convergence Points')
        st.plotly_chart(fig)

    def main():
        if "base_dir" not in st.session_state:
            st.session_state.base_dir = "./data/processed"
        
        data = load_data(st.session_state.base_dir)
        ma_data = calculate_moving_averages(data)
        threshold = st.slider("Set Convergence Threshold", 0.01, 0.05, 0.01)
        
        convergence_points = check_convergence(ma_data, threshold)
        st.write(f"Found {len(convergence_points)} convergence points.")
        
        visualize_convergence(data, convergence_points)

    if __name__ == "__main__":
        main()
