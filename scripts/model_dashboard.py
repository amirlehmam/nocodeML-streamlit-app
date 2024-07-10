import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
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

# SQLAlchemy connection string
DATABASE_URL = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}?sslmode={DB_CONFIG['sslmode']}"

def load_merged_data_from_db():
    engine = create_engine(DATABASE_URL)
    query = "SELECT * FROM merged_trade_indicator_event"
    data = pd.read_sql(query, engine)
    return data

# Load data
def load_data_from_db():
    st.write("Loading data from database...")
    data = load_merged_data_from_db()
    st.write(f"Data loaded with shape: {data.shape}")
    return data

def preprocess_data(data):
    st.write("Preprocessing data...")
    if data.shape[1] < 8:
        raise ValueError("Data does not have enough columns for indicators. Ensure indicators start after the 7th column.")

    indicator_columns = data.columns[7:]
    st.write(f"Indicator columns: {indicator_columns}")

    # Drop columns with all missing values
    data = data.dropna(axis=1, how='all')
    indicator_columns = data.columns[7:]

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    indicators_imputed = imputer.fit_transform(data[indicator_columns])

    # Check unique values before conversion
    st.write(f"Unique values in 'result' column before conversion: {data['result'].unique()}")

    # Convert result to binary
    data.loc[:, 'result'] = data['result'].apply(lambda x: 1 if x.lower() == 'win' else 0).astype(int)

    # Ensure result column is integer type
    data['result'] = data['result'].astype(int)

    # Check unique values after conversion
    st.write(f"Unique values in 'result' column after conversion: {data['result'].unique()}")
    st.write(f"Data type of 'result' column: {data['result'].dtype}")

    labeled_data = data.sample(frac=0.1, random_state=42)
    unlabeled_data = data.drop(labeled_data.index)

    X_labeled = labeled_data[indicator_columns]
    y_labeled = labeled_data['result']

    X_unlabeled = unlabeled_data[indicator_columns]

    # Apply imputer to labeled and unlabeled data
    X_labeled_imputed = imputer.transform(X_labeled)
    X_unlabeled_imputed = imputer.transform(X_unlabeled)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_labeled_imputed, y_labeled, test_size=0.2, random_state=42)
    
    # Print shapes and types of training and test data
    st.write(f"Shape of X_train: {X_train.shape}")
    st.write(f"Shape of X_test: {X_test.shape}")
    st.write(f"Shape of y_train: {y_train.shape}")
    st.write(f"Shape of y_test: {y_test.shape}")
    st.write(f"Data type of y_train: {y_train.dtype}")
    st.write(f"Data type of y_test: {y_test.dtype}")

    st.write("Data preprocessing completed.")
    return data, X_train, X_test, y_train, y_test, indicator_columns

def calculate_optimal_win_ranges(data, target='result', features=None):
    optimal_ranges = []

    if features is None:
        features = data.columns.drop([target])

    for feature in tqdm(features, desc="Calculating Optimal Win Ranges"):
        data[feature] = pd.to_numeric(data[feature], errors='coerce')
        
        win_values = data[data[target] == 1][feature].dropna().values.astype(float)
        loss_values = data[data[target] == 0][feature].dropna().values.astype(float)

        if len(win_values) == 0 or len(loss_values) == 0:
            continue

        # Dimensionality reduction for high-dimensional data
        if len(win_values.shape) > 1:
            pca = PCA(n_components=1)
            win_values = pca.fit_transform(win_values)
            loss_values = pca.transform(loss_values)

        try:
            win_kde = gaussian_kde(win_values)
            loss_kde = gaussian_kde(loss_values)
        except np.linalg.LinAlgError:
            continue

        x_grid = np.linspace(min(data[feature].dropna()), max(data[feature].dropna()), 1000)
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

    return optimal_ranges

def plot_pie_distribution(data, feature, target='result'):
    win_data = data[data[target] == 1][feature].dropna()
    loss_data = data[data[target] == 0][feature].dropna()

    win_counts = win_data.value_counts()
    loss_counts = loss_data.value_counts()

    labels = win_counts.index.union(loss_counts.index)

    win_values = [win_counts.get(label, 0) for label in labels]
    loss_values = [loss_counts.get(label, 0) for label in labels]

    fig = go.Figure(data=[
        go.Pie(labels=labels, values=win_values, name='Win', hole=0.3, domain=dict(x=[0, 0.5])),
        go.Pie(labels=labels, values=loss_values, name='Loss', hole=0.3, domain=dict(x=[0.5, 1.0]))
    ])

    fig.update_layout(title=f'Win vs Loss Distribution for {feature}')
    return fig

def plot_optimal_win_ranges(data, optimal_ranges, target='result', trade_type='', model_name=''):
    for item in optimal_ranges:
        feature = item['feature']
        ranges = item['optimal_win_ranges']
        
        win_values = data[data[target] == 1][feature].dropna()
        loss_values = data[data[target] == 0][feature].dropna()
        
        plt.figure(figsize=(12, 6))
        sns.histplot(win_values, color='blue', label='Win', kde=True, stat='density', element='step', fill=True)
        sns.histplot(loss_values, color='red', label='Loss', kde=True, stat='density', element='step', fill=True)
        
        for range_start, range_end in ranges:
            plt.axvspan(range_start, range_end, color='blue', alpha=0.3)

        plt.title(f'Optimal Win Ranges for {feature} ({trade_type}, {model_name})')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        st.pyplot(plt.gcf())
        plt.close()

def summarize_optimal_win_ranges(optimal_ranges):
    summary = []
    for item in optimal_ranges:
        feature = item['feature']
        for range_start, range_end in item['optimal_win_ranges']:
            summary.append({
                'feature': feature,
                'optimal_win_range_start': range_start,
                'optimal_win_range_end': range_end
            })
    return pd.DataFrame(summary)

def run_model_dashboard():
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=0.1, algorithm='SAMME'),
        'LogisticRegression': LogisticRegression(max_iter=2000),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, use_label_encoder=False, eval_metric='logloss')
    }

    def load_and_preprocess_data():
        data = load_data_from_db()
        return preprocess_data(data)

    st.title("Advanced Trading Dashboard")

    if st.button("Load Data"):
        st.write("Loading data...")
        try:
            data, X_train, X_test, y_train, y_test, indicator_columns = load_and_preprocess_data()
            st.success("Data loaded and preprocessed successfully.")
            st.session_state.data = data
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.indicator_columns = indicator_columns

            results = []
            feature_importances = {}

            for clf_name, clf in classifiers.items():
                st.write(f"Training {clf_name}...")
                clf.fit(X_train, y_train)
                accuracy = clf.score(X_test, y_test)
                st.write(f"Trained {clf_name} with accuracy: {accuracy}")
                results.append((clf_name, accuracy))
                if hasattr(clf, 'feature_importances_'):
                    feature_importances[clf_name] = clf.feature_importances_

            results_df = pd.DataFrame(results, columns=['Classifier', 'Accuracy'])
            results_df.sort_values(by='Accuracy', ascending=False, inplace=True)
            st.session_state.results_df = results_df
            st.session_state.feature_importances = feature_importances
            st.write("Model training and evaluation completed.")

        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

    if "results_df" in st.session_state and "feature_importances" in st.session_state:
        st.write("Displaying model results...")

        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.expander("Model Accuracy"):
                st.dataframe(st.session_state.results_df)

            with st.expander("Feature Importance"):
                classifier = st.selectbox("Select Classifier for Feature Importance", list(st.session_state.feature_importances.keys()), key='feature_importance_classifier')

                if classifier:
                    importance = st.session_state.feature_importances[classifier]
                    # Debug prints to verify lengths
                    st.write(f"Length of indicator columns: {len(st.session_state.indicator_columns)}")
                    st.write(f"Length of importance values: {len(importance)}")
                    st.write(f"Indicator columns: {st.session_state.indicator_columns}")
                    st.write(f"Importance values: {importance}")
                    
                    # Ensure lengths match before creating DataFrame
                    if len(st.session_state.indicator_columns) == len(importance):
                        importance_df = pd.DataFrame({
                            'Feature': st.session_state.indicator_columns,
                            'Importance': importance
                        }).sort_values(by='Importance', ascending=False)

                        top_n = st.selectbox("Select Top N Indicators", [3, 5, 10, len(importance_df)], index=2)
                        selected_features = importance_df.head(top_n)

                        fig = px.bar(selected_features, x='Importance', y='Feature', orientation='h')
                        st.plotly_chart(fig)
                    else:
                        st.error("Mismatch in lengths of indicator columns and importance values.")

            with st.expander("Optimal Win Ranges"):
                st.subheader("Calculate Optimal Win Ranges")

                trade_type = st.selectbox("Select Trade Type", ["Long Only", "Short Only", "Long & Short"], key='win_ranges_trade_type')
                model_name = st.selectbox("Select Model", list(classifiers.keys()), key='win_ranges_model')
                top_n = st.selectbox("Select Top N Indicators", [None, 10, 5, 3, "ALL"], key='win_ranges_top_n')
                individual_indicator = st.selectbox("Select Individual Indicator", st.session_state.indicator_columns, key='win_ranges_individual')

                if st.button("Calculate and Plot Optimal Win Ranges", key='plot_win_ranges'):
                    if top_n == "ALL":
                        selected_features = st.session_state.indicator_columns
                    elif top_n is not None:
                        selected_features = importance_df.head(int(top_n))['Feature'].tolist()
                    else:
                        selected_features = [individual_indicator]

                    optimal_ranges = calculate_optimal_win_ranges(st.session_state.data, features=selected_features)
                    plot_optimal_win_ranges(st.session_state.data, optimal_ranges, trade_type=trade_type, model_name=model_name)
                    
                    optimal_win_ranges_summary = summarize_optimal_win_ranges(optimal_ranges)
                    st.write(optimal_win_ranges_summary)
                    
                    # Save summary to CSV
                    output_path = f'optimal_win_ranges_summary_{model_name}.csv'
                    optimal_win_ranges_summary.to_csv(output_path, index=False)
                    st.write(f"Saved optimal win ranges summary to {output_path}")

        with col2:
            with st.expander("Individual Indicator Analysis"):
                model_name = st.selectbox("Select Model", list(classifiers.keys()), key='individual_indicator_model')
                individual_indicator = st.selectbox("Select Individual Indicator", st.session_state.indicator_columns, key='individual_indicator')
                
                if st.button("Plot Indicator Distribution", key='plot_individual'):
                    # Histogram plot
                    fig = px.histogram(st.session_state.data, x=individual_indicator, color='result', marginal="rug", hover_data=st.session_state.data.columns)
                    st.plotly_chart(fig)

                    # Pie distribution plot
                    pie_fig = plot_pie_distribution(st.session_state.data, individual_indicator)
                    st.plotly_chart(pie_fig)

            with st.expander("Multiple Indicators Analysis"):
                model_name = st.selectbox("Select Model", list(classifiers.keys()), key='multiple_indicators_model')
                selected_indicators = st.multiselect("Select Indicators to Analyze Together", st.session_state.indicator_columns, key='multiple_indicators')
                
                if st.button("Plot Multiple Indicators", key='plot_multiple'):
                    if len(selected_indicators) > 1:
                        fig = px.scatter_matrix(st.session_state.data, dimensions=selected_indicators, color='result', title="Scatter Matrix of Selected Indicators")
                        st.plotly_chart(fig)
                    else:
                        st.warning("Please select at least two indicators.")

            with st.expander("Additional EDA"):
                st.subheader("Additional Exploratory Data Analysis")
                with st.spinner("Generating additional EDA plots..."):
                    # Correlation matrix
                    selected_model = st.selectbox("Select Model for Correlation", list(st.session_state.feature_importances.keys()), key='correlation_model')
                    feature_importances_df = pd.DataFrame({
                        'Feature': st.session_state.indicator_columns,
                        'Importance': st.session_state.feature_importances[selected_model]
                    }).sort_values(by='Importance', ascending=False)

                    top_features = feature_importances_df.head(10)['Feature']
                    top_features_list = top_features.tolist()

                    st.write("Top 10 Features for Correlation Matrix:")
                    st.write(top_features_list)

                    corr = st.session_state.data[top_features_list].corr()
                    fig, ax = plt.subplots(figsize=(12, 10))
                    sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f', ax=ax)
                    st.write("Correlation Matrix of Top 10 Features")
                    st.pyplot(fig)

                    # Pie chart of wins vs losses
                    win_loss_counts = st.session_state.data['result'].value_counts()
                    fig = px.pie(values=win_loss_counts, names=win_loss_counts.index, title="Win vs Loss Distribution")
                    st.plotly_chart(fig)

if __name__ == "__main__":
    run_model_dashboard()
