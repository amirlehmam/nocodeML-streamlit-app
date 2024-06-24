import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Function to load event_data from the processed directory
def load_event_data(base_dir):
    try:
        event_data_path = os.path.join(base_dir, "event_data.csv")
        merged_data_path = os.path.join(base_dir, "merged_trade_indicator_event.csv")
        
        event_data = pd.read_csv(event_data_path)
        merged_data = pd.read_csv(merged_data_path)
        
        return event_data, merged_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Function to process initial data
def process_initial_data(event_data, merged_data):
    timestamp_col = 'time'
    event_data[timestamp_col] = pd.to_datetime(event_data[timestamp_col], errors='coerce')
    merged_data[timestamp_col] = pd.to_datetime(merged_data[timestamp_col], errors='coerce')

    if 'amount' in event_data.columns:
        event_data['amount'] = event_data['amount'].str.replace(r'[\$,]', '', regex=True)
        event_data['amount'] = pd.to_numeric(event_data['amount'], errors='coerce')
    
    if 'amount' in merged_data.columns:
        merged_data['amount'] = merged_data['amount'].str.replace(r'[\$,]', '', regex=True)
        merged_data['amount'] = pd.to_numeric(merged_data['amount'], errors='coerce')
    
    event_data = event_data.dropna(subset=[timestamp_col])
    merged_data = merged_data.dropna(subset=[timestamp_col])
    
    return event_data, merged_data, timestamp_col

# Function to analyze trading hours
def analyze_trading_hours(merged_data, timestamp_col):
    merged_data['hour'] = merged_data[timestamp_col].dt.hour
    merged_data['profit'] = merged_data['amount'].apply(lambda x: x if x > 0 else 0)
    merged_data['loss'] = merged_data['amount'].apply(lambda x: x if x < 0 else 0)
    
    performance_by_hour = merged_data.groupby('hour').agg(
        total_trades=('result', 'count'),
        total_profit=('profit', 'sum'),
        total_loss=('loss', 'sum'),
        win_rate=('result', lambda x: (x == 'win').mean())
    ).reset_index()

    return performance_by_hour

# Function to plot total trades by hour
def plot_total_trades(performance_by_hour):
    fig = px.bar(performance_by_hour, x='hour', y='total_trades', title='Total Trades by Hour',
                 labels={'hour': 'Hour of Day', 'total_trades': 'Total Trades'},
                 color='total_trades', color_continuous_scale='Viridis')
    st.plotly_chart(fig)

# Function to plot total profit by hour
def plot_total_profit(performance_by_hour):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=performance_by_hour['hour'], y=performance_by_hour['total_profit'], 
                         name='Profit', marker_color='green', 
                         text=performance_by_hour['total_profit'], textposition='auto'))
    fig.update_layout(title='Total Profit by Hour',
                      xaxis_title='Hour of Day',
                      yaxis_title='Total Profit ($)')
    st.plotly_chart(fig)

# Function to plot total loss by hour
def plot_total_loss(performance_by_hour):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=performance_by_hour['hour'], y=performance_by_hour['total_loss'], 
                         name='Loss', marker_color='red', 
                         text=performance_by_hour['total_loss'], textposition='auto'))
    fig.update_layout(title='Total Loss by Hour',
                      xaxis_title='Hour of Day',
                      yaxis_title='Total Loss ($)')
    st.plotly_chart(fig)

# Function to plot win rate by hour
def plot_win_rate(performance_by_hour):
    fig = px.bar(performance_by_hour, x='hour', y='win_rate', title='Win Rate by Hour',
                 labels={'hour': 'Hour of Day', 'win_rate': 'Win Rate'},
                 color='win_rate', color_continuous_scale='Magma')
    st.plotly_chart(fig)

def main(base_dir):
    event_data, merged_data = load_event_data(base_dir)
    if event_data is None or merged_data is None:
        return
    
    event_data, merged_data, timestamp_col = process_initial_data(event_data, merged_data)
    performance_by_hour = analyze_trading_hours(merged_data, timestamp_col)
    
    st.title("Trading Hours Analysis")
    
    plot_total_trades(performance_by_hour)
    plot_total_profit(performance_by_hour)
    plot_total_loss(performance_by_hour)
    plot_win_rate(performance_by_hour)

if __name__ == "__main__":
    base_dir = "./data/processed/"  # Replace with your data directory
    main(base_dir)
