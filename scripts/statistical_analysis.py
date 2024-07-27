# statistical_analysis.py
import pandas as pd
import numpy as np
import streamlit as st
from sqlalchemy import create_engine
from scipy.stats import ttest_ind, gaussian_kde
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from tqdm import tqdm

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
    # Convert 'result' column to numeric: 0 for 'Loss' and 1 for 'Profit'
    data['result'] = data['event_event'].map({'Loss': 0, 'Profit': 1})
    
    # Ensure all other data is numeric and fill NaNs with 0
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    data.fillna(0, inplace=True)
    
    # Log the number of rows and columns after preprocessing
    st.write(f"Data after preprocessing: {data.shape[0]} rows, {data.shape[1]} columns")
    
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

def calculate_descriptive_statistics(df, selected_indicators):
    desc_stats = df[selected_indicators].describe().transpose()
    return desc_stats

def feature_importance_analysis(df, selected_indicators, target_variable):
    X = df[selected_indicators]
    y = df[target_variable]

    # Ensure X and y are numeric
    X = X.select_dtypes(include=[np.number])
    y = pd.to_numeric(y, errors='coerce')

    # Filter out rows where y is NaN
    X = X[~y.isna()]
    y = y[~y.isna()]

    if X.shape[0] == 0 or y.shape[0] == 0:
        st.error("No samples available for feature importance analysis.")
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Correlation with the target variable
    correlation_with_target = X.corrwith(y).sort_values(ascending=False)

    # ANOVA F-test
    f_values, p_values = f_classif(X, y)
    anova_df = pd.DataFrame({'feature': X.columns, 'F-value': f_values, 'p-value': p_values}).sort_values(by='F-value', ascending=False)

    # Mutual Information
    mi = mutual_info_classif(X, y)
    mi_df = pd.DataFrame({'feature': X.columns, 'MI': mi}).sort_values(by='MI', ascending=False)

    # Permutation Importance
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    perm_importance = permutation_importance(model, X, y, n_repeats=30, random_state=42, n_jobs=2)
    perm_importance_df = pd.DataFrame({'feature': X.columns, 'Importance': perm_importance.importances_mean}).sort_values(by='Importance', ascending=False)

    return correlation_with_target, anova_df, mi_df, perm_importance_df

def explain_feature_importance(df, feature_importance_df, method_name, win_or_loss):
    explanation = []
    for feature, score in feature_importance_df.items():
        if score > 0.7:
            explanation.append(f"The feature '{feature}' has a high {method_name} score of {score:.2f} for {win_or_loss} trades, indicating a strong influence on the target variable. This means changes in '{feature}' are strongly associated with {win_or_loss} outcomes. Consider using this feature for developing strategies or setting thresholds.")
        elif score > 0.4:
            explanation.append(f"The feature '{feature}' has a moderate {method_name} score of {score:.2f} for {win_or_loss} trades, suggesting a notable impact on the target variable. It is worth considering '{feature}' when analyzing your trading strategy. Look at how this feature interacts with others for a combined effect.")
        else:
            explanation.append(f"The feature '{feature}' has a low {method_name} score of {score:.2f} for {win_or_loss} trades, indicating a weaker relationship with the target variable. '{feature}' might not be very influential in determining {win_or_loss} outcomes. However, do not discard it outright; it may still provide value in combination with other indicators.")
    return explanation

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

    if selected_indicators:
        st.subheader("Descriptive Statistics", help="Begin with basic descriptive statistics to understand the distribution and central tendency of each indicator.")
        desc_stats = calculate_descriptive_statistics(df, selected_indicators)
        st.write(desc_stats)

        st.subheader("Visualizations", help="Visualize the distribution and relationships of indicators using different plots.")

        st.markdown("### Histograms and Box Plots", help="Visualize the distribution of each indicator to spot patterns or outliers.")
        
        col1, col2, col3 = st.columns(3)
        for idx, indicator in enumerate(selected_indicators):
            with [col1, col2, col3][idx % 3]:
                st.markdown(f"#### {indicator}")
                fig_hist = px.histogram(df, x=indicator, nbins=50, title=f'Histogram of {indicator}')
                st.plotly_chart(fig_hist)
                fig_box = px.box(df, y=indicator, title=f'Box Plot of {indicator}')
                st.plotly_chart(fig_box)

        st.markdown("### Scatter Plots", help="Plot indicators against performance metrics to visually assess relationships.")
        for idx, indicator in enumerate(selected_indicators):
            with [col1, col2, col3][idx % 3]:
                fig_scatter = px.scatter(df, x=indicator, y=target_variable, title=f'Scatter Plot of {indicator} vs {target_variable}')
                st.plotly_chart(fig_scatter)

        st.markdown("### Correlation Heatmaps", help="Use correlation heatmaps to identify strongly correlated indicators.")
        correlation_matrix = df[selected_indicators].corr()
        fig_heatmap = px.imshow(correlation_matrix, text_auto=True, aspect="auto", title="Correlation Matrix")
        st.plotly_chart(fig_heatmap)

        st.subheader("Feature Importance Analysis", help="Use various statistical techniques to rank indicators by their potential impact on the target variable.")

        # Feature importance analysis for all trades
        correlation_with_target, anova_df, mi_df, perm_importance_df = feature_importance_analysis(df, selected_indicators, target_variable)

        st.markdown("### Correlation with Target Variable (All Trades)", help="Correlation scores of each indicator with the target variable.")
        st.write(correlation_with_target)
        with st.expander("Correlation Analysis Explanation (All Trades)"):
            explanations_corr = explain_feature_importance(df, correlation_with_target, "correlation", "all")
            for explanation in explanations_corr:
                st.write(explanation)

        st.markdown("### ANOVA F-Test Scores (All Trades)", help="ANOVA F-test scores for each indicator.")
        st.write(anova_df)
        with st.expander("ANOVA F-Test Explanation (All Trades)"):
            explanations_anova = explain_feature_importance(df, anova_df.set_index('feature')['F-value'], "ANOVA F-Test", "all")
            for explanation in explanations_anova:
                st.write(explanation)

        st.markdown("### Mutual Information Scores (All Trades)", help="Mutual Information scores for each indicator.")
        st.write(mi_df)
        with st.expander("Mutual Information Explanation (All Trades)"):
            explanations_mi = explain_feature_importance(df, mi_df.set_index('feature')['MI'], "Mutual Information", "all")
            for explanation in explanations_mi:
                st.write(explanation)

        st.markdown("### Permutation Importance Scores (All Trades)", help="Permutation importance scores for each indicator.")
        st.write(perm_importance_df)
        with st.expander("Permutation Importance Explanation (All Trades)"):
            explanations_perm = explain_feature_importance(df, perm_importance_df.set_index('feature')['Importance'], "Permutation Importance", "all")
            for explanation in explanations_perm:
                st.write(explanation)

        st.markdown("### Feature Importance Bar Plots (All Trades)", help="Bar plots displaying the importance of each feature.")
        col1, col2 = st.columns(2)
        with col1:
            fig_corr = px.bar(correlation_with_target, title="Correlation with Target Variable (All Trades)")
            st.plotly_chart(fig_corr)
        with col2:
            fig_anova = px.bar(anova_df, x='feature', y='F-value', title="ANOVA F-Test Scores (All Trades)")
            st.plotly_chart(fig_anova)

        col1, col2 = st.columns(2)
        with col1:
            fig_mi = px.bar(mi_df, x='feature', y='MI', title="Mutual Information Scores (All Trades)")
            st.plotly_chart(fig_mi)
        with col2:
            fig_perm = px.bar(perm_importance_df, x='feature', y='Importance', title="Permutation Importance Scores (All Trades)")
            st.plotly_chart(fig_perm)

        st.header("Optimal Win Ranges and KDE Distribution", help="Calculate and visualize the optimal win ranges for each indicator.")
        optimal_ranges_long = calculate_optimal_win_ranges(df, target=target_variable, features=selected_indicators, trade_type='LE')
        optimal_ranges_short = calculate_optimal_win_ranges(df, target=target_variable, features=selected_indicators, trade_type='SE')
        optimal_ranges_both = calculate_optimal_win_ranges(df, target=target_variable, features=selected_indicators)

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

        optimal_win_ranges_summary_long = summarize_optimal_win_ranges(optimal_ranges_long, 'LE')
        optimal_win_ranges_summary_short = summarize_optimal_win_ranges(optimal_ranges_short, 'SE')
        optimal_win_ranges_summary_both = summarize_optimal_win_ranges(optimal_ranges_both, '')

        st.write("Optimal Win Ranges Summary for Long Trades")
        st.dataframe(optimal_win_ranges_summary_long)
        st.write("Optimal Win Ranges Summary for Short Trades")
        st.dataframe(optimal_win_ranges_summary_short)
        st.write("Optimal Win Ranges Summary for Both Long and Short Trades")
        st.dataframe(optimal_win_ranges_summary_both)

        st.header("Indicator Trends Over Time", help="Plot the trends of selected indicators over time to understand how they evolve and interact with trading results.")
        selected_time_indicators = st.multiselect("Select Time-based Indicators", selected_indicators, default=selected_indicators[:2])
        if selected_time_indicators:
            for idx, indicator in enumerate(selected_time_indicators):
                with [col1, col2, col3][idx % 3]:
                    fig_time = px.line(df, x='time', y=indicator, title=f"{indicator} Over Time")
                    st.plotly_chart(fig_time)

        st.header("Pairplot of Selected Indicators", help="Create a pairplot to visualize relationships between selected indicators. This helps in identifying potential correlations and interactions.")
        pairplot_indicators = st.multiselect("Select Indicators for Pairplot", selected_indicators, default=selected_indicators[:5])
        if pairplot_indicators:
            fig_pairplot = px.scatter_matrix(df, dimensions=pairplot_indicators, title="Pairplot of Selected Indicators")
            st.plotly_chart(fig_pairplot)

        st.header("Boxplot of Indicators", help="Use boxplots to visualize the distribution of selected indicators. This helps in identifying outliers and understanding the spread of indicator values.")
        boxplot_indicators = st.multiselect("Select Indicators for Boxplot", selected_indicators, default=selected_indicators[:5])
        if boxplot_indicators:
            fig_boxplot = px.box(df, y=boxplot_indicators, title="Boxplot of Selected Indicators")
            st.plotly_chart(fig_boxplot)

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
