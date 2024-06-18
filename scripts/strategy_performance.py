# strategy_perfomance.py

import os
import pandas as pd
import streamlit as st
import quantstats as qs

# Function to load and preprocess data
def load_and_preprocess_data(data_dir):
    st.write(f"Loading data from {data_dir}...")
    data_path = os.path.join(data_dir, "merged_trade_indicator_event.csv")
    if not os.path.exists(data_path):
        st.error(f"File not found: {data_path}")
        return None

    data = pd.read_csv(data_path)
    st.write(f"Data loaded with shape: {data.shape}")

    # Keep only the first 7 columns which are essential for performance analysis
    data = data.iloc[:, :7]
    
    # Convert 'time' to datetime
    data['time'] = pd.to_datetime(data['time'])

    # Ensure correct data types
    data['price'] = data['price'].astype(float)
    data['amount'] = data['amount'].replace('[\$,]', '', regex=True).astype(float)
    data['result'] = data['result'].apply(lambda x: 1 if x == 'win' else 0)
    
    return data

# Function to calculate and display performance metrics
def calculate_performance_metrics(data, output_path="tearsheet.html"):
    # Ensure 'time' is the index and sorted
    data.set_index('time', inplace=True)
    data.sort_index(inplace=True)
    
    # Assume 'price' represents the trade price and 'result' as win/loss
    returns = data['price'].pct_change().dropna()

    st.write("### Performance Metrics")
    st.write(qs.reports.metrics(returns, mode='full'))

    st.write("### Generating Full Report")
    qs.reports.html(returns, output=output_path, title="Strategy Performance Tearsheet")
    
    # Read the generated HTML file and embed it in the Streamlit app
    with open(output_path, 'r') as f:
        html_content = f.read()
        
    st.components.v1.html(html_content, height=800, scrolling=True)

# Main function for Streamlit app
def run_strategy_performance():
    st.title("Strategy Performance Analysis")

    if "base_dir" not in st.session_state:
        st.session_state.base_dir = "./data/processed"

    base_dir = st.text_input("Base Directory", value=st.session_state.base_dir)

    if st.button("Load and Analyze Data"):
        data = load_and_preprocess_data(base_dir)
        if data is not None:
            calculate_performance_metrics(data)

if __name__ == "__main__":
    run_strategy_performance()
