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

    # Keep only the first 7 columns
    data = data.iloc[:, :7]

    # Convert 'time' to datetime
    data['time'] = pd.to_datetime(data['time'])

    # Ensure correct data types
    data['price'] = data['price'].astype(float)
    data['amount'] = data['amount'].replace(r'[\$,]', '', regex=True).astype(float)

    # Check unique values in 'event_event'
    st.write("Unique values in 'event_event' column:")
    st.write(data['event_event'].unique())

    # Process 'result' column based on 'event_event'
    data['result'] = data['event_event'].apply(lambda x: 1 if x.lower() == 'profit' else 0)

    # Sort data by time
    data = data.sort_values(by='time')

    # Ensure unique timestamps by keeping the first occurrence
    data = data.drop_duplicates(subset=['time'], keep='first')

    # Debugging output
    st.write("Data preview after preprocessing:")
    st.dataframe(data.head())

    return data

# Function to calculate performance metrics
def calculate_performance_metrics(data):
    st.write("Calculating performance metrics...")

    # Calculate additional metrics
    winning_trades = len(data[data['result'] > 0])
    losing_trades = len(data[data['result'] == 0])
    total_gross_profit = data[data['result'] > 0]['amount'].sum()
    total_gross_loss = data[data['result'] == 0]['amount'].sum()
    net_profit_loss = total_gross_profit + total_gross_loss

    # Calculate maximum drawdown
    data['cum_return'] = data['amount'].cumsum()
    data['cum_max'] = data['cum_return'].cummax()
    data['drawdown'] = data['cum_max'] - data['cum_return']
    max_drawdown = data['drawdown'].max()

    # Debugging output
    st.write("Intermediate calculations:")
    st.write(f"Winning trades: {winning_trades}")
    st.write(f"Losing trades: {losing_trades}")
    st.write(f"Total gross profit: {total_gross_profit}")
    st.write(f"Total gross loss: {total_gross_loss}")
    st.write(f"Net profit/loss: {net_profit_loss}")
    st.write(f"Max drawdown: {max_drawdown}")

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
            winning_trades,
            losing_trades,
            f"${total_gross_profit:.2f}",
            f"${total_gross_loss:.2f}",
            f"${net_profit_loss:.2f}",
            f"${max_drawdown:.2f}"
        ]
    })

    st.write("### Key Performance Metrics")
    st.dataframe(metrics_df)

    # Display cumulative return over time
    st.write("### Cumulative Return Over Time")
    st.line_chart(data.set_index('time')['cum_return'])
    
    return data, metrics_df

# Function to generate quantstats tearsheet
def generate_quantstats_tearsheet(data, output_path="tearsheet.html"):
    data.set_index('time', inplace=True)
    returns = data['amount'].pct_change().dropna()

    # Generate the full report
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
            data, metrics_df = calculate_performance_metrics(data)
            st.write("Performance analysis completed.")
            generate_quantstats_tearsheet(data)

if __name__ == "__main__":
    run_strategy_performance()
