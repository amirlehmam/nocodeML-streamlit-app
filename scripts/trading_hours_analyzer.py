import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine
import pytz

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

# Function to load event_data from the database
@st.cache_data
def load_event_data_from_db():
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

# Function to process initial data
def process_initial_data(event_data, merged_data):
    timestamp_col = 'time'
    event_data[timestamp_col] = pd.to_datetime(event_data[timestamp_col], errors='coerce').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    merged_data[timestamp_col] = pd.to_datetime(merged_data[timestamp_col], errors='coerce').dt.tz_localize('UTC').dt.tz_convert('America/New_York')

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
        net_profit=('amount', 'sum'),
        avg_trade_value=('amount', 'mean'),
        win_rate=('result', lambda x: (x == 'win').mean())
    ).reset_index()

    performance_by_hour['profit_loss_ratio'] = performance_by_hour['total_profit'] / performance_by_hour['total_loss'].abs()

    return performance_by_hour

# Function to plot total trades by hour
def plot_total_trades(performance_by_hour):
    fig = px.bar(performance_by_hour, x='hour', y='total_trades', title='Total Trades by Hour',
                 labels={'hour': 'Hour of Day', 'total_trades': 'Total Trades'},
                 color='total_trades', color_continuous_scale='Viridis')
    st.plotly_chart(fig)
    st.markdown(f"### Analysis:")
    st.markdown(f"Hours with the highest trading volumes are: {performance_by_hour.loc[performance_by_hour['total_trades'].idxmax()]['hour']} with {performance_by_hour['total_trades'].max()} trades.")
    st.markdown(f"Hours with the lowest trading volumes are: {performance_by_hour.loc[performance_by_hour['total_trades'].idxmin()]['hour']} with {performance_by_hour['total_trades'].min()} trades.")

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
    st.markdown(f"### Analysis:")
    st.markdown(f"Hours with the highest total profit are: {performance_by_hour.loc[performance_by_hour['total_profit'].idxmax()]['hour']} with ${performance_by_hour['total_profit'].max():,.2f} profit.")
    st.markdown(f"Hours with the lowest total profit are: {performance_by_hour.loc[performance_by_hour['total_profit'].idxmin()]['hour']} with ${performance_by_hour['total_profit'].min():,.2f} profit.")

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
    st.markdown(f"### Analysis:")
    st.markdown(f"Hours with the highest total loss are: {performance_by_hour.loc[performance_by_hour['total_loss'].idxmin()]['hour']} with ${performance_by_hour['total_loss'].min():,.2f} loss.")
    st.markdown(f"Hours with the lowest total loss are: {performance_by_hour.loc[performance_by_hour['total_loss'].idxmax()]['hour']} with ${performance_by_hour['total_loss'].max():,.2f} loss.")

# Function to plot net profit by hour
def plot_net_profit(performance_by_hour):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=performance_by_hour['hour'], y=performance_by_hour['net_profit'], 
                         name='Net Profit', marker_color='blue', 
                         text=performance_by_hour['net_profit'], textposition='auto'))
    fig.update_layout(title='Net Profit by Hour',
                      xaxis_title='Hour of Day',
                      yaxis_title='Net Profit ($)')
    st.plotly_chart(fig)
    st.markdown(f"### Analysis:")
    max_net_profit_hour = performance_by_hour.loc[performance_by_hour['net_profit'].idxmax()]['hour']
    max_net_profit = performance_by_hour['net_profit'].max()
    min_net_profit_hour = performance_by_hour.loc[performance_by_hour['net_profit'].idxmin()]['hour']
    min_net_profit = performance_by_hour['net_profit'].min()
    st.markdown(f"Hour with the highest net profit: {max_net_profit_hour} with ${max_net_profit:,.2f} net profit.")
    st.markdown(f"Hour with the lowest net profit: {min_net_profit_hour} with ${min_net_profit:,.2f} net profit.")

# Function to plot profit/loss ratio by hour
def plot_profit_loss_ratio(performance_by_hour):
    fig = px.bar(performance_by_hour, x='hour', y='profit_loss_ratio', title='Profit/Loss Ratio by Hour',
                 labels={'hour': 'Hour of Day', 'profit_loss_ratio': 'Profit/Loss Ratio'},
                 color='profit_loss_ratio', color_continuous_scale='Bluered')
    st.plotly_chart(fig)
    st.markdown(f"### Analysis:")
    st.markdown(f"Hours with the highest profit/loss ratio are: {performance_by_hour.loc[performance_by_hour['profit_loss_ratio'].idxmax()]['hour']} with a ratio of {performance_by_hour['profit_loss_ratio'].max():.2f}.")
    st.markdown(f"Hours with the lowest profit/loss ratio are: {performance_by_hour.loc[performance_by_hour['profit_loss_ratio'].idxmin()]['hour']} with a ratio of {performance_by_hour['profit_loss_ratio'].min():.2f}.")

# Function to plot win rate by hour
def plot_win_rate(performance_by_hour):
    fig = px.bar(performance_by_hour, x='hour', y='win_rate', title='Win Rate by Hour',
                 labels={'hour': 'Hour of Day', 'win_rate': 'Win Rate'},
                 color='win_rate', color_continuous_scale='Magma')
    st.plotly_chart(fig)
    st.markdown(f"### Analysis:")
    st.markdown(f"Hour with the highest win rate: {performance_by_hour.loc[performance_by_hour['win_rate'].idxmax()]['hour']} with a win rate of {performance_by_hour['win_rate'].max() * 100:.2f}%.")
    st.markdown(f"Hour with the lowest win rate: {performance_by_hour.loc[performance_by_hour['win_rate'].idxmin()]['hour']} with a win rate of {performance_by_hour['win_rate'].min() * 100:.2f}%.")

# Function to plot trade volume vs. net profit
def plot_trade_volume_vs_net_profit(performance_by_hour):
    fig = px.scatter(performance_by_hour, x='total_trades', y='net_profit', title='Trade Volume vs. Net Profit',
                     labels={'total_trades': 'Total Trades', 'net_profit': 'Net Profit ($)'},
                     color='net_profit', color_continuous_scale='Viridis', size='total_trades')
    st.plotly_chart(fig)
    st.markdown(f"### Analysis:")
    st.markdown(f"Overall, hours with higher trading volumes tend to show {('greater net profits' if performance_by_hour['net_profit'].corr(performance_by_hour['total_trades']) > 0 else 'greater net losses')}, indicating a {'positive' if performance_by_hour['net_profit'].corr(performance_by_hour['total_trades']) > 0 else 'negative'} correlation.")

# Function to plot average trade value by hour
def plot_avg_trade_value(performance_by_hour):
    fig = px.bar(performance_by_hour, x='hour', y='avg_trade_value', title='Average Trade Value by Hour',
                 labels={'hour': 'Hour of Day', 'avg_trade_value': 'Average Trade Value ($)'},
                 color='avg_trade_value', color_continuous_scale='Cividis')
    st.plotly_chart(fig)
    st.markdown(f"### Analysis:")
    st.markdown(f"Hour with the highest average trade value: {performance_by_hour.loc[performance_by_hour['avg_trade_value'].idxmax()]['hour']} with an average value of ${performance_by_hour['avg_trade_value'].max():,.2f}.")
    st.markdown(f"Hour with the lowest average trade value: {performance_by_hour.loc[performance_by_hour['avg_trade_value'].idxmin()]['hour']} with an average value of ${performance_by_hour['avg_trade_value'].min():,.2f}.")

# Function to plot box plot of trade values by hour
def plot_box_trade_values(merged_data):
    fig = px.box(merged_data, x='hour', y='amount', title='Trade Values Distribution by Hour',
                 labels={'hour': 'Hour of Day', 'amount': 'Trade Value ($)'})
    st.plotly_chart(fig)
    st.markdown(f"### Analysis:")
    st.markdown(f"The distribution of trade values by hour indicates the range and outliers of trade amounts for each hour. This can help identify periods of high volatility or stability.")

# Function to display information and plots
def display_info_and_plots(performance_by_hour, merged_data):
    st.title("Trading Hours Analysis")

    st.markdown("## Total Trades by Hour")
    st.markdown("This chart shows the total number of trades made during each hour of the day.")
    col1, col2 = st.columns(2)
    with col1:
        plot_total_trades(performance_by_hour)
    with col2:
        plot_avg_trade_value(performance_by_hour)
    
    st.markdown("## Profit and Loss Analysis")
    st.markdown("The following charts display the profit and loss incurred during each hour of the day.")
    col1, col2 = st.columns(2)
    with col1:
        plot_total_profit(performance_by_hour)
    with col2:
        plot_total_loss(performance_by_hour)

    st.markdown("## Net Profit by Hour")
    st.markdown("This chart displays the net profit (total profit minus total loss) for each hour of the day.")
    plot_net_profit(performance_by_hour)
    
    st.markdown("## Profit/Loss Ratio by Hour")
    st.markdown("This chart shows the ratio of total profit to total loss for each hour, indicating the profitability of each hour.")
    plot_profit_loss_ratio(performance_by_hour)
    
    st.markdown("## Win Rate by Hour")
    st.markdown("This chart shows the win rate (percentage of winning trades) for each hour of the day.")
    plot_win_rate(performance_by_hour)
    
    st.markdown("## Trade Volume vs. Net Profit")
    st.markdown("This scatter plot shows the relationship between the number of trades and net profit for each hour.")
    plot_trade_volume_vs_net_profit(performance_by_hour)
    
    st.markdown("## Trade Values Distribution by Hour")
    st.markdown("This box plot shows the distribution of trade values for each hour, highlighting the spread and outliers.")
    plot_box_trade_values(merged_data)

def main(base_dir=None):  # Accept an argument even if unused
    event_data, merged_data = load_event_data_from_db()
    if event_data is None or merged_data is None:
        return
    
    event_data, merged_data, timestamp_col = process_initial_data(event_data, merged_data)
    performance_by_hour = analyze_trading_hours(merged_data, timestamp_col)
    
    display_info_and_plots(performance_by_hour, merged_data)

if __name__ == "__main__":
    main()
