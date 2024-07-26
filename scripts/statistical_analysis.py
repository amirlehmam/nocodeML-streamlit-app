# statistical_analysis.py
import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ttest_ind, gaussian_kde
from tqdm import tqdm
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

@st.cache_data
def load_data():
    engine = get_db_connection()
    query = "SELECT * FROM merged_trade_indicator_event"
    df = pd.read_sql(query, engine, parse_dates=['time'])  # Ensure 'time' column is parsed as datetime
    return df

@st.cache_data
def preprocess_data(data):
    # Convert 'result' column to numeric: 0 for 'loss' and 1 for 'win'
    data['result'] = data['result'].map({'loss': 0, 'win': 1})
    
    # Ensure all other data is numeric and fill NaNs with 0
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    data.fillna(0, inplace=True)
    return data

def calculate_optimal_win_ranges(data, target='result', features=None, trade_type=None):
    optimal_ranges = []

    if features is None:
        features = data.columns.drop([target])

    for feature in tqdm(features, desc=f"Calculating Optimal Win Ranges ({trade_type})"):
        data[feature] = pd.to_numeric(data[feature], errors='coerce')
        
        if trade_type:
            trade_data = data[data['event'].str.startswith(trade_type)]
        else:
            trade_data = data

        win_values = trade_data[trade_data[target] == 1][feature].dropna().values.astype(float)
        loss_values = trade_data[trade_data[target] == 0][feature].dropna().values.astype(float)

        if len(win_values) == 0 or len(loss_values) == 0:
            continue

        # Check if the data is binary
        if np.unique(win_values).size == 2 and np.array_equal(np.unique(win_values), [0, 1]) and np.unique(loss_values).size == 2 and np.array_equal(np.unique(loss_values), [0, 1]):
            optimal_ranges.append({
                'feature': feature,
                'optimal_win_ranges': [(0, 1)]
            })
            continue

        try:
            win_kde = gaussian_kde(win_values)
            loss_kde = gaussian_kde(loss_values)

            x_grid = np.linspace(min(trade_data[feature].dropna()), max(trade_data[feature].dropna()), 1000)
            win_density = win_kde(x_grid)
            loss_density = loss_kde(x_grid)

            is_winning = win_density > loss_density
            ranges = []
            start_idx = None

            for i in range(len(is_winning)):
                if is_winning[i] and start_idx is None:
                    start_idx = i
                elif not is_winning[i] and start_idx is not None:
                    ranges.append((x_grid[start_idx], x_grid[i - 1]))
                    start_idx = None
            if start_idx is not None:
                ranges.append((x_grid[start_idx], x_grid[-1]))

            optimal_ranges.append({
                'feature': feature,
                'optimal_win_ranges': ranges
            })
        except np.linalg.LinAlgError:
            st.warning(f"Skipped KDE for {feature} due to singular covariance matrix.")

    return optimal_ranges

def plot_kde_distribution(data, trade_type, optimal_ranges):
    plots = []
    descriptions = []
    feature_info = []

    for item in optimal_ranges:
        feature = item['feature']
        
        win_values = data[(data['result'] == 1) & (data['event'].str.startswith(trade_type))][feature].dropna()
        loss_values = data[(data['result'] == 0) & (data['event'].str.startswith(trade_type))][feature].dropna()

        win_count = len(win_values)
        loss_count = len(loss_values)

        feature_info.append(f"Plotting feature: {feature}, trade type: {trade_type}\n"
                            f"Win values count: {win_count}, Loss values count: {loss_count}")

        if win_count == 0 or loss_count == 0:
            continue

        if data[feature].nunique() == 2:  # Check if the feature is binary
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Under', 'Above'],
                y=[(win_values == 0).sum(), (win_values == 1).sum()],
                name='Win',
                marker_color='blue',
                opacity=0.75
            ))
            fig.add_trace(go.Bar(
                x=['Under', 'Above'],
                y=[(loss_values == 0).sum(), (loss_values == 1).sum()],
                name='Loss',
                marker_color='red',
                opacity=0.75
            ))

            fig.update_layout(
                title=f'Binary Indicator Distribution for {feature} ({trade_type})',
                xaxis_title=feature,
                yaxis_title='Count',
                barmode='group'  # Change here to display side-by-side bars
            )
        else:
            kde_win = gaussian_kde(win_values)
            kde_loss = gaussian_kde(loss_values)
            x_grid = np.linspace(min(data[feature].dropna()), max(data[feature].dropna()), 1000)
            kde_win_density = kde_win(x_grid)
            kde_loss_density = kde_loss(x_grid)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_grid, y=kde_win_density, mode='lines', name='Win', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=x_grid, y=kde_loss_density, mode='lines', name='Loss', line=dict(color='red')))
            for range_item in optimal_ranges:
                if range_item['feature'] == feature:
                    for start, end in range_item['optimal_win_ranges']:
                        fig.add_vrect(x0=start, x1=end, fillcolor="blue", opacity=0.3, line_width=0)

            fig.update_layout(
                title=f'KDE Plot with Optimal Win Ranges for {feature} ({trade_type})',
                xaxis_title=feature,
                yaxis_title='Density',
                width=800,
                height=400
            )

        plots.append(fig)
        descriptions.append(f'Optimal Win Ranges for {feature} ({trade_type})')

    return plots, descriptions, feature_info

def summarize_optimal_win_ranges(optimal_ranges, trade_type):
    summary = []
    for item in optimal_ranges:
        for start, end in item['optimal_win_ranges']:
            summary.append({
                'feature': item['feature'],
                'trade_type': trade_type,
                'optimal_win_range_start': start,
                'optimal_win_range_end': end
            })
    return pd.DataFrame(summary)

def statistical_analysis():
    df = load_data()
    df = preprocess_data(df)

    st.title("Trade Indicator Analysis Dashboard")

    st.sidebar.header("Filter Options")

    # Default date range
    min_date = df['time'].min()
    max_date = df['time'].max()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], help="Select the date range for the analysis. By default, it covers the entire date range of the dataset.")

    # Feature type selection
    selected_feature_types = st.sidebar.multiselect("Select Feature Types", 
                                                    ["Non-Market Value Data", "Percent Away Indicators", "Binary Indicators"],
                                                    help="Choose the types of indicators you want to analyze. You can select multiple feature types.")

    # Indicator selection based on feature types
    non_indicator_columns = ["time", "event", "qty", "price", "event_event", "amount", "result", "strategy_amount", "account_amount"]
    all_indicators = [col for col in df.columns if col not in non_indicator_columns and df[col].dtype in [np.float64, np.int64]]

    indicator_columns = []
    if "Non-Market Value Data" in selected_feature_types:
        indicator_columns.extend([col for col in all_indicators if "_percent_away" not in col and "_binary" not in col])
    if "Percent Away Indicators" in selected_feature_types:
        indicator_columns.extend([col for col in all_indicators if "_percent_away" in col])
    if "Binary Indicators" in selected_feature_types:
        indicator_columns.extend([col for col in all_indicators if "_binary" in col])

    selected_indicators = st.sidebar.multiselect("Select Indicators", indicator_columns, help="Choose specific indicators from the selected feature types for a detailed analysis.")

    # Set 'result' as the only target variable
    target_variable = 'result'
    
    # Ensure target variable is numeric
    df[target_variable] = pd.to_numeric(df[target_variable], errors='coerce')

    if date_range and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]

    st.header("Data Overview", help="This section provides a quick glance at the first few rows of your dataset to understand its structure and contents.")
    st.write(df.head())

    st.header("Statistical Analysis", help="Perform various statistical analyses to understand the relationships between indicators and the target variable, which is the trading result (win or loss).")

    if selected_indicators:
        st.subheader("Correlation Analysis", help="Analyze the correlation between different indicators to see how they are related to each other. A high correlation between indicators might indicate redundancy.")
        correlation_matrix = df[selected_indicators].corr()
        fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto", title="Correlation Matrix")
        st.plotly_chart(fig)

        st.subheader("Regression Analysis", help="Use regression analysis to identify the impact of different indicators on the target variable. This helps in understanding which indicators are most influential in predicting trading results.")
        X = df[selected_indicators]
        y = df[target_variable]

        # Ensure X and y are numeric
        X = X.select_dtypes(include=[np.number])
        y = pd.to_numeric(y, errors='coerce')

        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        st.write(model.summary())

        st.subheader("Principal Component Analysis (PCA)", help="Use PCA to reduce the dimensionality of the dataset while preserving as much variability as possible. This helps in visualizing the data in a lower-dimensional space and identifying key patterns.")
        numeric_indicators = [indicator for indicator in selected_indicators if df[indicator].dtype in [np.float64, np.int64]]
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_indicators]), columns=numeric_indicators)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(df_scaled)
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

        fig = px.scatter(pca_df, x='PC1', y='PC2', color=df[target_variable].astype(str), title='PCA of Selected Indicators')
        st.plotly_chart(fig)

    st.header("Visualizations", help="Visualize the data using different plots to gain insights into the distribution and behavior of various indicators.")

    if selected_indicators:
        # Calculate optimal win ranges for long, short, and both trades
        optimal_ranges_long = calculate_optimal_win_ranges(df, target=target_variable, features=selected_indicators, trade_type='LE')
        optimal_ranges_short = calculate_optimal_win_ranges(df, target=target_variable, features=selected_indicators, trade_type='SE')
        optimal_ranges_both = calculate_optimal_win_ranges(df, target=target_variable, features=selected_indicators)

        # Plot KDE distributions for Long, Short, and Both trades
        st.subheader("KDE Distribution for Long, Short, and Both Trades", help="Kernel Density Estimation (KDE) plots show the distribution of indicators for winning and losing trades. This helps in identifying optimal ranges for indicators that maximize winning trades.")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Long Trades")
            long_plots, long_descriptions, long_feature_info = plot_kde_distribution(df, 'LE', optimal_ranges_long)
            for fig in long_plots:
                st.plotly_chart(fig)
        
        with col2:
            st.markdown("### Short Trades")
            short_plots, short_descriptions, short_feature_info = plot_kde_distribution(df, 'SE', optimal_ranges_short)
            for fig in short_plots:
                st.plotly_chart(fig)

        with col3:
            st.markdown("### Both Long and Short Trades")
            both_plots, both_descriptions, both_feature_info = plot_kde_distribution(df, '', optimal_ranges_both)
            for fig in both_plots:
                st.plotly_chart(fig)

        # Summarize optimal win ranges for each trade type
        optimal_win_ranges_summary_long = summarize_optimal_win_ranges(optimal_ranges_long, 'LE')
        optimal_win_ranges_summary_short = summarize_optimal_win_ranges(optimal_ranges_short, 'SE')
        optimal_win_ranges_summary_both = summarize_optimal_win_ranges(optimal_ranges_both, '')

        st.write("Optimal Win Ranges Summary for Long Trades")
        st.dataframe(optimal_win_ranges_summary_long)
        st.write("Optimal Win Ranges Summary for Short Trades")
        st.dataframe(optimal_win_ranges_summary_short)
        st.write("Optimal Win Ranges Summary for Both Long and Short Trades")
        st.dataframe(optimal_win_ranges_summary_both)

        st.header("Additional Visualizations", help="Explore additional visualizations to gain deeper insights into the data and indicators.")

        st.subheader("Indicator Trends Over Time", help="Plot the trends of selected indicators over time to understand how they evolve and interact with trading results.")
        selected_time_indicators = st.multiselect("Select Time-based Indicators", selected_indicators, default=selected_indicators[:2])
        if selected_time_indicators:
            for indicator in selected_time_indicators:
                fig = px.line(df, x='time', y=indicator, title=f"{indicator} Over Time")
                st.plotly_chart(fig)

        st.subheader("Pairplot of Selected Indicators", help="Create a pairplot to visualize relationships between selected indicators. This helps in identifying potential correlations and interactions.")
        pairplot_indicators = st.multiselect("Select Indicators for Pairplot", selected_indicators, default=selected_indicators[:5])
        if pairplot_indicators:
            fig = px.scatter_matrix(df, dimensions=pairplot_indicators, title="Pairplot of Selected Indicators")
            st.plotly_chart(fig)

        st.subheader("Boxplot of Indicators", help="Use boxplots to visualize the distribution of selected indicators. This helps in identifying outliers and understanding the spread of indicator values.")
        boxplot_indicators = st.multiselect("Select Indicators for Boxplot", selected_indicators, default=selected_indicators[:5])
        if boxplot_indicators:
            fig = px.box(df, y=boxplot_indicators, title="Boxplot of Selected Indicators")
            st.plotly_chart(fig)

    st.header("Download Data", help="Download the filtered and processed data as a CSV file for further analysis.")
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='filtered_data.csv',
        mime='text/csv',
    )

if __name__ == "__main__":
    statistical_analysis()
