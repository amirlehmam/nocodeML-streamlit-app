# statistical_analysis.py
import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
from scipy.stats import ttest_ind

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
    # Ensure all data is numeric and fill NaNs with 0
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    data.fillna(0, inplace=True)
    return data

def statistical_analysis():
    df = load_data()
    df = preprocess_data(df)

    st.title("Trade Indicator Analysis Dashboard")

    st.sidebar.header("Filter Options")

    # Default date range
    min_date = df['time'].min()
    max_date = df['time'].max()
    date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date])
    st.sidebar.write("Select the date range for the analysis. By default, it covers the entire date range of the dataset.")

    # Feature type selection
    selected_feature_types = st.sidebar.multiselect("Select Feature Types", 
                                                    ["Non-Market Value Data", "Percent Away Indicators", "Binary Indicators"])
    st.sidebar.write("Choose the types of indicators you want to analyze. You can select multiple feature types.")

    # Indicator selection based on feature types
    non_indicator_columns = ["time", "event", "qty", "price", "event_event", "amount", "result"]
    all_indicators = [col for col in df.columns if col not in non_indicator_columns and df[col].dtype in [np.float64, np.int64]]

    indicator_columns = []
    if "Non-Market Value Data" in selected_feature_types:
        indicator_columns.extend([col for col in all_indicators if "_percent_away" not in col and "_binary" not in col])
    if "Percent Away Indicators" in selected_feature_types:
        indicator_columns.extend([col for col in all_indicators if "_percent_away" in col])
    if "Binary Indicators" in selected_feature_types:
        indicator_columns.extend([col for col in all_indicators if "_binary" in col])

    selected_indicators = st.sidebar.multiselect("Select Indicators", indicator_columns)
    st.sidebar.write("Choose specific indicators from the selected feature types for a detailed analysis.")

    target_variable = st.sidebar.selectbox("Select Target Variable", non_indicator_columns)
    st.sidebar.write("Select the target variable you want to analyze against the indicators.")

    if date_range and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df = df[(df['time'] >= start_date) & (df['time'] <= end_date)]

    st.header("Data Overview")
    st.write(df.head())

    st.header("Statistical Analysis")

    if selected_indicators:
        st.subheader("Correlation Analysis")
        correlation_matrix = df[selected_indicators].corr()
        fig, ax = plt.subplots(figsize=(16, 12))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        st.subheader("Regression Analysis")
        X = df[selected_indicators]
        y = df[target_variable]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        st.write(model.summary())

        st.subheader("Hypothesis Testing")
        for indicator in selected_indicators:
            if indicator != target_variable:
                group1 = df[df[indicator] == 1][target_variable]
                group2 = df[df[indicator] == 0][target_variable]
                if len(group1) > 0 and len(group2) > 0:
                    t_stat, p_value = ttest_ind(group1, group2)
                    st.write(f"T-Statistic for {indicator}: {t_stat}, P-Value: {p_value}")

        st.subheader("Principal Component Analysis (PCA)")
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df[selected_indicators]), columns=selected_indicators)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(df_scaled)
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='PC1', y='PC2', data=pca_df, hue=df[target_variable], ax=ax)
        st.pyplot(fig)

    st.header("Visualizations")

    if selected_indicators:
        st.subheader(f"Scatter Plot: {selected_indicators[0]} vs {selected_indicators[1]}")
        fig = px.scatter(df, x=selected_indicators[0], y=selected_indicators[1], color=target_variable, title=f"{selected_indicators[0]} vs {selected_indicators[1]}")
        st.plotly_chart(fig)

        for indicator in selected_indicators:
            st.subheader(f"Histogram: {indicator}")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df[indicator], bins=30, kde=True, ax=ax)
            st.pyplot(fig)

        st.subheader("Heatmap: Correlation Matrix")
        fig, ax = plt.subplots(figsize=(16, 12))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        st.header("Additional Visualizations")

        st.subheader("Indicator Trends Over Time")
        selected_time_indicators = st.multiselect("Select Time-based Indicators", selected_indicators, default=selected_indicators[:2])
        if selected_time_indicators:
            for indicator in selected_time_indicators:
                fig = px.line(df, x='time', y=indicator, title=f"{indicator} Over Time")
                st.plotly_chart(fig)

        st.subheader("Pairplot of Selected Indicators")
        pairplot_indicators = st.multiselect("Select Indicators for Pairplot", selected_indicators, default=selected_indicators[:5])
        if pairplot_indicators:
            sns.pairplot(df[pairplot_indicators])
            st.pyplot()

        st.subheader("Boxplot of Indicators")
        boxplot_indicators = st.multiselect("Select Indicators for Boxplot", selected_indicators, default=selected_indicators[:5])
        if boxplot_indicators:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.boxplot(data=df[boxplot_indicators], ax=ax)
            st.pyplot(fig)

    st.header("Download Data")
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='filtered_data.csv',
        mime='text/csv',
    )

if __name__ == "__main__":
    statistical_analysis()
