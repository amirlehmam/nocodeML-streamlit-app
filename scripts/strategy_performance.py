import streamlit as st
import pandas as pd
import hashlib
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sqlalchemy import create_engine

# Database connection details
DB_CONFIG = {
    'dbname': 'defaultdb',
    'user': 'doadmin',
    'password': 'AVNS_hnzmIdBmiO7aj5nylWW',
    'host': 'nocodemldb-do-user-16993120-0.c.db.ondigitalocean.com',
    'port': 25060,
    'sslmode': 'require'
}

def get_db_connection():
    connection_str = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
    engine = create_engine(connection_str)
    return engine

# Function to calculate hash of the data
def calculate_data_hash(data):
    return hashlib.sha256(pd.util.hash_pandas_object(data, index=True).values).hexdigest()

# Function to load event_data from the database
@st.cache_data(show_spinner=False, persist=True, ttl=3600)
def load_event_data_from_db(data_hash):
    try:
        engine = get_db_connection()
        event_data_query = "SELECT * FROM event_data"
        merged_data_query = "SELECT * FROM merged_trade_indicator_event"
        
        event_data = pd.read_sql_query(event_data_query, engine)
        merged_data = pd.read_sql_query(merged_data_query, engine)
        
        return event_data, merged_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Function to get the hash of the data
def get_data_hash():
    try:
        engine = get_db_connection()
        event_data_query = "SELECT * FROM event_data"
        merged_data_query = "SELECT * FROM merged_trade_indicator_event"
        
        event_data = pd.read_sql_query(event_data_query, engine)
        merged_data = pd.read_sql_query(merged_data_query, engine)
        
        event_data_hash = calculate_data_hash(event_data)
        merged_data_hash = calculate_data_hash(merged_data)
        
        combined_hash = hashlib.sha256((event_data_hash + merged_data_hash).encode()).hexdigest()
        
        return combined_hash
    except Exception as e:
        st.error(f"Error calculating data hash: {e}")
        return None

# Function to process initial data
def process_initial_data(event_data, merged_data):
    timestamp_col = 'time'
    event_data[timestamp_col] = pd.to_datetime(event_data[timestamp_col], errors='coerce')
    merged_data[timestamp_col] = pd.to_datetime(merged_data[timestamp_col], errors='coerce')

    if 'amount' in event_data.columns:
        event_data['amount'] = event_data['amount'].str.replace(r'[\$,]', '', regex=True).astype(float)
    
    event_data = event_data.dropna(subset=[timestamp_col])
    merged_data = merged_data.dropna(subset=[timestamp_col])
    
    return event_data, merged_data, timestamp_col

# Function to calculate additional metrics
def calculate_additional_metrics(event_data, trades, net_profit_loss, total_gross_profit, total_gross_loss):
    timestamp_col = 'time'

    # Calculate Profit Factor
    profit_factor = total_gross_profit / abs(total_gross_loss) if total_gross_loss != 0 else np.nan

    # Calculate Sharpe Ratio
    risk_free_rate = 0.01  # Example risk-free rate, adjust as necessary
    monthly_returns = trades.resample('M', on=timestamp_col)['amount'].sum()
    sharpe_ratio = (monthly_returns.mean() - risk_free_rate) / monthly_returns.std() if monthly_returns.std() != 0 else np.nan

    # Calculate Sortino Ratio
    downside_risk = np.sqrt(np.mean(np.minimum(0, monthly_returns - monthly_returns.mean()) ** 2))
    sortino_ratio = (monthly_returns.mean() - risk_free_rate) / downside_risk if downside_risk != 0 else np.nan

    return profit_factor, sharpe_ratio, sortino_ratio

# Function to calculate performance metrics
def calculate_performance_metrics(event_data, merged_data, timestamp_col, entry_multiplier):
    event_data['amount'] = pd.to_numeric(event_data['amount'], errors='coerce')

    trades = event_data[event_data['event'].str.contains('Profit|Loss', case=False)]
    strategy_account_events = event_data[event_data['event'].str.contains('Strategy|Account', case=False)]

    winning_trades = len(merged_data[merged_data['result'] == 'win']) * entry_multiplier
    losing_trades = len(merged_data[merged_data['result'] == 'loss']) * entry_multiplier
    total_trades = winning_trades + losing_trades
    total_gross_profit = trades[trades['event'].str.contains('Profit', case=False)]['amount'].sum()
    total_gross_loss = trades[trades['event'].str.contains('Loss', case=False)]['amount'].sum()
    net_profit_loss = total_gross_profit - abs(total_gross_loss)

    trades['cum_return'] = trades['amount'].cumsum()
    trades['cum_max'] = trades['cum_return'].cummax()
    trades['drawdown'] = trades['cum_max'] - trades['cum_return']
    max_drawdown = trades['drawdown'].max()

    percent_profitable = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    avg_trade = net_profit_loss / total_trades if total_trades > 0 else 0
    avg_winning_trade = total_gross_profit / winning_trades if winning_trades > 0 else 0
    avg_losing_trade = total_gross_loss / losing_trades if losing_trades > 0 else 0
    ratio_avg_win_avg_loss = avg_winning_trade / abs(avg_losing_trade) if avg_losing_trade != 0 else 0

    profit_factor, sharpe_ratio, sortino_ratio = calculate_additional_metrics(event_data, trades, net_profit_loss, total_gross_profit, total_gross_loss)

    metrics_df = pd.DataFrame({
        "Metric": [
            "Total Trades",
            "Winning Trades", 
            "Losing Trades", 
            "Total Gross Profit ($)", 
            "Total Gross Loss ($)", 
            "Net Profit/Loss ($)", 
            "Max Drawdown ($)",
            "Percent Profitable (%)"
        ],
        "Value": [
            total_trades,
            winning_trades,
            losing_trades,
            total_gross_profit,
            total_gross_loss,
            net_profit_loss,
            max_drawdown,
            percent_profitable
        ]
    })

    additional_metrics_df = pd.DataFrame({
        "Metric": [
            "Ratio Avg Win / Avg Loss",
            "Profit Factor",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Average Trade ($)",
            "Average Winning Trade ($)",
            "Average Losing Trade ($)"
        ],
        "Value": [
            ratio_avg_win_avg_loss,
            profit_factor,
            sharpe_ratio,
            sortino_ratio,            
            avg_trade,
            avg_winning_trade,
            avg_losing_trade
        ]
    })

    # Format the numbers to 2 decimal places
    metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.2f}")
    additional_metrics_df['Value'] = additional_metrics_df['Value'].apply(lambda x: f"{x:.2f}")

    # Apply conditional formatting for the main metrics
    def apply_styles(row):
        value = float(row['Value'].replace(',', ''))
        metric = row['Metric']
        background_color = ''
        if metric == "Net Profit/Loss ($)":
            background_color = 'background-color: red' if value < 0 else 'background-color: green'
        elif metric == "Max Drawdown ($)":
            if value <= 5000:
                background_color = 'background-color: green'
            elif value <= 10000:
                background_color = 'background-color: lightgreen'
            elif value <= 15000:
                background_color = 'background-color: yellow'
            elif value <= 20000:
                background_color = 'background-color: orange'
            elif value <= 30000:
                background_color = 'background-color: red'
            else:
                background_color = 'background-color: darkred'
        elif metric == "Percent Profitable (%)":
            if value < 45:
                background_color = 'background-color: red'
            elif 45 <= value <= 55:
                background_color = 'background-color: yellow'
            else:
                background_color = 'background-color: green'
        return [''] * len(row) if background_color == '' else [background_color if col == 'Value' else '' for col in row.index]

    # Apply conditional formatting for additional metrics
    def apply_additional_styles(row):
        value = float(row['Value'].replace(',', ''))
        metric = row['Metric']
        background_color = ''
        if metric == "Ratio Avg Win / Avg Loss":
            if value > 1.5:
                background_color = 'background-color: green'
            elif 1.0 < value <= 1.5:
                background_color = 'background-color: lightgreen'
            elif 0.5 < value <= 1.0:
                background_color = 'background-color: yellow'
            else:
                background_color = 'background-color: red'
        elif metric == "Profit Factor":
            if value > 2:
                background_color = 'background-color: green'
            elif 1.5 < value <= 2:
                background_color = 'background-color: lightgreen'
            elif 1.0 < value <= 1.5:
                background_color = 'background-color: yellow'
            else:
                background_color = 'background-color: red'
        elif metric == "Sharpe Ratio":
            if value > 2:
                background_color = 'background-color: green'
            elif 1.5 < value <= 2:
                background_color = 'background-color: lightgreen'
            elif 1.0 < value <= 1.5:
                background_color = 'background-color: yellow'
            else:
                background_color = 'background-color: red'
        elif metric == "Sortino Ratio":
            if value > 2:
                background_color = 'background-color: green'
            elif 1.5 < value <= 2:
                background_color = 'background-color: lightgreen'
            elif 1.0 < value <= 1.5:
                background_color = 'background-color: yellow'
            else:
                background_color = 'background-color: red'
        return [''] * len(row) if background_color == '' else [background_color if col == 'Value' else '' for col in row.index]

    styled_metrics_df = metrics_df.style.apply(apply_styles, axis=1)
    styled_additional_metrics_df = additional_metrics_df.style.apply(apply_additional_styles, axis=1)

    # Display dataframes side by side
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Key Performance Metrics")
        st.dataframe(styled_metrics_df)
    with col2:
        st.write("### Additional Metrics")
        st.dataframe(styled_additional_metrics_df)

    st.write("### Cumulative Return Over Time")
    st.line_chart(trades.set_index(timestamp_col)['cum_return'])

    return trades, metrics_df, additional_metrics_df

# Function to generate quantstats tearsheet
#def generate_quantstats_tearsheet(data, timestamp_col, output_path="tearsheet.html"):
#    data.set_index(timestamp_col, inplace=True)
#    returns = data['amount'].pct_change().dropna()
#
#    qs.reports.html(returns, output=output_path, title="Strategy Performance Tearsheet")
#
#    with open(output_path, 'r') as f:
#        html_content = f.read()
#
#    css_path = os.path.join(os.path.dirname(__file__), "custom_style.css")
#    with open(css_path, "r") as f:
#        custom_css = f.read()
#
#    html_content = html_content.replace('</head>', f'<style>{custom_css}</style></head>')
#
#    st.components.v1.html(html_content, height=800, scrolling=True)

# Function to plot additional metrics
def plot_additional_metrics(trades, timestamp_col):
    st.write("### Equity Curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trades[timestamp_col], y=trades['cum_return'], mode='lines', name='Equity Curve'))
    fig.update_layout(title='Equity Curve', xaxis_title='Date', yaxis_title='Cumulative Return')
    st.plotly_chart(fig)

    st.write("### Daily Returns")
    daily_returns = trades.set_index(timestamp_col)['amount'].pct_change().dropna()
    fig = px.histogram(daily_returns, x=daily_returns, nbins=50, title='Daily Returns Distribution')
    st.plotly_chart(fig)

    st.write("### Daily Returns Heatmap")
    daily_returns = daily_returns.to_frame(name='daily_returns').astype(float)
    daily_returns['date'] = daily_returns.index.date
    daily_returns['week'] = daily_returns.index.isocalendar().week
    daily_returns['day'] = daily_returns.index.dayofweek

    # Aggregating daily returns to handle duplicates
    daily_returns_agg = daily_returns.groupby(['week', 'day'])['daily_returns'].mean().reset_index()
    daily_returns_pivot = daily_returns_agg.pivot(index='week', columns='day', values='daily_returns')

    fig, ax = plt.subplots(figsize=(16, 8))  # Adjusted figure size
    sns.heatmap(daily_returns_pivot, annot=True, fmt=".2f", cmap='RdYlGn', center=0, ax=ax, cbar_kws={'label': 'Daily Returns'})
    ax.set_title('Daily Returns Heatmap')
    ax.set_yticklabels([f"Week {int(label.get_text())}" for label in ax.get_yticklabels()])
    
    # Adjust tick labels based on the actual number of days in the data
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
    if daily_returns_pivot.shape[1] == 7:
        day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax.set_xticks(range(len(daily_returns_pivot.columns)))
    ax.set_xticklabels(day_labels, rotation=45)  # Rotated labels for clarity
    st.pyplot(fig)

    st.write("### Drawdown Periods")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trades[timestamp_col], y=-trades['drawdown'], fill='tozeroy', fillcolor='rgba(255,0,0,0.3)', mode='lines', name='Drawdown'))
    fig.update_layout(title='Drawdown Periods', xaxis_title='Date', yaxis_title='Drawdown ($)')
    st.plotly_chart(fig)

    st.write("### Profit and Loss Distribution")
    fig = px.histogram(trades['amount'], x=trades['amount'], nbins=50, title='Profit and Loss Distribution')
    st.plotly_chart(fig)

# Main function for Streamlit app
def run_strategy_performance():
    st.title("Strategy Performance Analysis")

    entry_multiplier = st.slider("Select the number of entries per trade", min_value=1, max_value=5, value=5)

    if st.button("Load and Analyze Data"):
        data_hash = get_data_hash()
        event_data, merged_data = load_event_data_from_db(data_hash)
        if event_data is not None and merged_data is not None:
            event_data, merged_data, timestamp_col = process_initial_data(event_data, merged_data)

            if event_data is not None:
                trades, metrics_df, additional_metrics_df = calculate_performance_metrics(event_data, merged_data, timestamp_col, entry_multiplier)
                st.write("Performance analysis completed.")
                
                # Plot Max Drawdown using Plotly
                st.write("### Max Drawdown Over Time")
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=trades[timestamp_col],
                    y=-trades['drawdown'],
                    marker_color='red'
                ))
                fig.update_layout(
                    title='Max Drawdown Over Time',
                    xaxis_title='Date',
                    yaxis_title='Drawdown ($)',
                    yaxis=dict(autorange='reversed')
                )
                st.plotly_chart(fig)

                # Plot additional metrics
                plot_additional_metrics(trades, timestamp_col)

                #generate_quantstats_tearsheet(trades, timestamp_col)

if __name__ == "__main__":
    run_strategy_performance()
