import os
import pandas as pd
import streamlit as st
import quantstats as qs
from sqlalchemy import create_engine

# Database connection
DATABASE_URL = "postgresql+psycopg2://doadmin:AVNS_hnzmIdBmiO7aj5nylWW@nocodemldb-do-user-16993120-0.c.db.ondigitalocean.com:25060/defaultdb?sslmode=require"
engine = create_engine(DATABASE_URL)

# Function to load and preprocess data from the database
def load_and_preprocess_data():
    st.write("Loading data from the database...")
    
    # Load data from merged_trade_indicator_event
    query = "SELECT * FROM merged_trade_indicator_event"
    data = pd.read_sql(query, engine)
    st.write(f"Data loaded with shape: {data.shape}")

    # Convert 'time' to datetime
    data['time'] = pd.to_datetime(data['time'])

    # Ensure correct data types
    data['price'] = data['price'].astype(float)
    data['amount'] = data['amount'].replace(r'[\$,]', '', regex=True).astype(float)

    # Filter out rows with missing results
    data = data.dropna(subset=['result'])

    # Map result to numeric values
    data['result'] = data['result'].apply(lambda x: 1 if x == 'win' else 0)

    # Remove duplicate 'time' entries
    data = data.drop_duplicates(subset=['time'])

    # Debug: Check the data
    st.write("Sample Data:")
    st.write(data.head(50))

    return data

# Function to calculate additional metrics
def calculate_additional_metrics(data):
    metrics = {}
    metrics['Winning Trades'] = len(data[data['result'] == 1])
    metrics['Losing Trades'] = len(data[data['result'] == 0])
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

    # Calculate returns
    data['returns'] = data['price'].pct_change().dropna()

    # Calculate additional metrics
    additional_metrics = calculate_additional_metrics(data)
    max_drawdown_amount = calculate_max_drawdown(data)
    total_gross_profit = data[data['result'] == 1]['amount'].sum()
    total_gross_loss = data[data['result'] == 0]['amount'].sum()
    net_profit_loss = total_gross_profit + total_gross_loss  # Adjusted calculation

    # Debug: Check calculated metrics
    st.write("Total Gross Profit ($):", total_gross_profit)
    st.write("Total Gross Loss ($):", total_gross_loss)
    st.write("Net Profit/Loss ($):", net_profit_loss)
    st.write("Max Drawdown ($):", max_drawdown_amount)

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
            f"${total_gross_profit:.2f}",
            f"${total_gross_loss:.2f}",
            f"${net_profit_loss:.2f}",
            f"${max_drawdown_amount:.2f}"
        ]
    })

    st.write("### Key Performance Metrics")
    st.dataframe(metrics_df)

    # Generate the full report with QuantStats
    st.write("### Performance Report")
    returns = data['returns'].dropna()
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
