import os
import pandas as pd
import streamlit as st
import quantstats as qs

# Function to load event_data from the processed directory
def load_event_data(base_dir):
    try:
        data_path = os.path.join(base_dir, "event_data.csv")
        data = pd.read_csv(data_path)
        st.write("Loaded data columns:", data.columns.tolist())  # Print column names for inspection
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to process initial data
def process_initial_data(initial_data):
    st.write("Initial data columns:", initial_data.columns.tolist())

    # Use the actual timestamp column name
    timestamp_col = 'time'
    initial_data[timestamp_col] = pd.to_datetime(initial_data[timestamp_col], errors='coerce')

    # Print the first few rows to understand the structure
    st.write("Initial data preview:")
    st.write(initial_data.head())

    # Replace commas with periods and convert to float for relevant columns
    if 'amount' in initial_data.columns:
        initial_data['amount'] = initial_data['amount'].str.replace(r'[\$,]', '', regex=True).astype(float)
    
    # Drop rows where the timestamp conversion failed
    initial_data = initial_data.dropna(subset=[timestamp_col])
    
    return initial_data, timestamp_col

# Function to calculate performance metrics
def calculate_performance_metrics(data, timestamp_col):
    st.write("Calculating performance metrics...")

    # Ensure correct types
    data['amount'] = pd.to_numeric(data['amount'], errors='coerce')

    # Separate trades and strategy/account events
    trades = data[data['event'].str.contains('Profit|Loss', case=False)]
    strategy_account_events = data[data['event'].str.contains('Strategy|Account', case=False)]

    # Calculate metrics
    total_trades = len(trades)
    winning_trades = trades[trades['event'].str.contains('Profit', case=False)].shape[0]
    losing_trades = trades[trades['event'].str.contains('Loss', case=False)].shape[0]
    total_gross_profit = trades[trades['event'].str.contains('Profit', case=False)]['amount'].sum()
    total_gross_loss = trades[trades['event'].str.contains('Loss', case=False)]['amount'].sum()
    net_profit_loss = total_gross_profit + total_gross_loss

    # Calculate maximum drawdown
    trades['cum_return'] = trades['amount'].cumsum()
    trades['cum_max'] = trades['cum_return'].cummax()
    trades['drawdown'] = trades['cum_max'] - trades['cum_return']
    max_drawdown = trades['drawdown'].max()

    # Calculate other metrics based on NinjaTrader definitions
    percent_profitable = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    avg_trade = net_profit_loss / total_trades if total_trades > 0 else 0
    avg_winning_trade = total_gross_profit / winning_trades if winning_trades > 0 else 0
    avg_losing_trade = total_gross_loss / losing_trades if losing_trades > 0 else 0
    ratio_avg_win_avg_loss = avg_winning_trade / abs(avg_losing_trade) if avg_losing_trade != 0 else 0

    metrics_df = pd.DataFrame({
        "Metric": [
            "Total Trades",
            "Winning Trades", 
            "Losing Trades", 
            "Total Gross Profit ($)", 
            "Total Gross Loss ($)", 
            "Net Profit/Loss ($)", 
            "Max Drawdown ($)",
            "Percent Profitable (%)",
            "Average Trade ($)",
            "Average Winning Trade ($)",
            "Average Losing Trade ($)",
            "Ratio Avg Win / Avg Loss"
        ],
        "Value": [
            total_trades,
            winning_trades,
            losing_trades,
            f"${total_gross_profit:.2f}",
            f"${total_gross_loss:.2f}",
            f"${net_profit_loss:.2f}",
            f"${max_drawdown:.2f}",
            f"{percent_profitable:.2f}%",
            f"${avg_trade:.2f}",
            f"${avg_winning_trade:.2f}",
            f"${avg_losing_trade:.2f}",
            f"{ratio_avg_win_avg_loss:.2f}"
        ]
    })

    st.write("### Key Performance Metrics")
    st.dataframe(metrics_df)

    # Display cumulative return over time
    st.write("### Cumulative Return Over Time")
    st.line_chart(trades.set_index(timestamp_col)['cum_return'])

    return trades, metrics_df

# Function to generate quantstats tearsheet
def generate_quantstats_tearsheet(data, timestamp_col, output_path="tearsheet.html"):
    data.set_index(timestamp_col, inplace=True)
    returns = data['amount'].pct_change().dropna()

    qs.reports.html(returns, output=output_path, title="Strategy Performance Tearsheet")

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
        initial_data = load_event_data(base_dir)
        if initial_data is not None:
            data, timestamp_col = process_initial_data(initial_data)

            if data is not None:
                data, metrics_df = calculate_performance_metrics(data, timestamp_col)
                st.write("Performance analysis completed.")
                generate_quantstats_tearsheet(data, timestamp_col)

if __name__ == "__main__":
    run_strategy_performance()
