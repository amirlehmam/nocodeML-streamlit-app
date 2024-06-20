import os
import pandas as pd
import streamlit as st
import quantstats as qs
import numpy as np
from sqlalchemy import create_engine

# Database connection
DATABASE_URL = "postgresql+psycopg2://doadmin:AVNS_hnzmIdBmiO7aj5nylWW@nocodemldb-do-user-16993120-0.c.db.ondigitalocean.com:25060/defaultdb?sslmode=require"
engine = create_engine(DATABASE_URL)

# Function to load and preprocess data from the database
def load_and_preprocess_data():
    st.write("Loading data from the database...")
    query = "SELECT * FROM merged_trade_indicator_event"
    data = pd.read_sql(query, engine)
    st.write(f"Data loaded with shape: {data.shape}")

    # Convert 'time' to datetime
    data['time'] = pd.to_datetime(data['time'])

    # Ensure correct data types
    data['price'] = data['price'].astype(float)
    data['amount'] = data['amount'].replace(r'[\$,]', '', regex=True).astype(float)
    data['result'] = data['result'].apply(lambda x: 1 if x == 'win' else 0)

    # Debug: Check the data
    st.write("Sample Data:")
    st.write(data.head())

    return data

# Function to calculate additional metrics
def calculate_additional_metrics(data):
    metrics = {}
    metrics['Winning Trades'] = len(data[data['result'] == 1])
    metrics['Losing Trades'] = len(data[data['result'] == 0])
    metrics['Number of Exits'] = data['event'].str.contains('exit', case=False).sum()
    metrics['Times SL Hit'] = data['event'].str.contains('sl', case=False).sum()
    metrics['Total Gross Profit'] = data[data['result'] == 1]['amount'].sum()
    metrics['Total Gross Loss'] = data[data['result'] == 0]['amount'].sum()
    metrics['Net Profit/Loss'] = metrics['Total Gross Profit'] + metrics['Total Gross Loss']
    return metrics

# Function to calculate maximum drawdown in dollars
def calculate_max_drawdown(data):
    data['cum_return'] = data['amount'].cumsum()
    data['cum_max'] = data['cum_return'].cummax()
    data['drawdown'] = data['cum_max'] - data['cum_return']
    max_drawdown = data['drawdown'].max()
    return max_drawdown

# Function to calculate performance metrics
def calculate_performance_metrics(data, output_path="tearsheet.html"):
    # Ensure 'time' is the index and sorted
    data.set_index('time', inplace=True)
    data.sort_index(inplace=True)

    # Assume 'price' represents the trade price and 'result' as win/loss
    returns = data['price'].pct_change().dropna()

    # Calculate additional metrics
    additional_metrics = calculate_additional_metrics(data)
    max_drawdown_amount = calculate_max_drawdown(data)

    # Debug: Check calculated metrics
    st.write(f"Total Gross Profit ($): {additional_metrics['Total Gross Profit']}")
    st.write(f"Total Gross Loss ($): {additional_metrics['Total Gross Loss']}")
    st.write(f"Net Profit/Loss ($): {additional_metrics['Net Profit/Loss']}")
    st.write(f"Max Drawdown ($): {max_drawdown_amount}")

    # Create a DataFrame to display the metrics
    metrics_df = pd.DataFrame({
        "Metric": [
            "Winning Trades", 
            "Losing Trades", 
            "Total Gross Profit ($)", 
            "Total Gross Loss ($)", 
            "Net Profit/Loss ($)", 
            "Max Drawdown ($)"
        ],
        "Value": [
            additional_metrics['Winning Trades'],
            additional_metrics['Losing Trades'],
            f"${additional_metrics['Total Gross Profit']:.2f}",
            f"${additional_metrics['Total Gross Loss']:.2f}",
            f"${additional_metrics['Net Profit/Loss']:.2f}",
            f"${max_drawdown_amount:.2f}"
        ]
    })

    st.write("### Key Performance Metrics")
    st.dataframe(metrics_df)

    # Plotting with QuantStats
    st.write("### Performance Report")
    returns = data['price'].pct_change().dropna()
    qs.reports.html(returns, output=output_path, title="Strategy Performance Tearsheet")

    # Read the generated HTML file and embed it in the Streamlit app
    with open(output_path, 'r') as f:
        html_content = f.read()

    # Get the absolute path of the custom CSS file
    custom_css_path = os.path.join(os.path.dirname(__file__), 'custom_style.css')

    # Insert custom CSS into the HTML content
    with open(custom_css_path, 'r') as css_file:
        css_content = css_file.read()

    style_tag = f'<style>{css_content}</style>'
    html_content = html_content.replace('<head>', f'<head>{style_tag}')

    st.components.v1.html(html_content, height=800, scrolling=True)

# Main function for Streamlit app
def run_strategy_performance():
    st.title("Strategy Performance Analysis")

    if st.button("Load and Analyze Data"):
        data = load_and_preprocess_data()
        if data is not None:
            calculate_performance_metrics(data)

if __name__ == "__main__":
    run_strategy_performance()
