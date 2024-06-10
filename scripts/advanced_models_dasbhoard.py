# advanced_models_dashboard.py
import os
import pandas as pd
import numpy as np
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
from tqdm import tqdm

# Load data
def load_data(data_dir):
    st.write(f"Loading data from {data_dir}...")
    data = pd.read_csv(os.path.join(data_dir, "merged_trade_indicator_event.csv"))
    st.write(f"Data loaded with shape: {data.shape}")
    return data

def preprocess_data(data):
    st.write("Preprocessing data...")
    if data.shape[1] < 8:
        raise ValueError("Data does not have enough columns for indicators. Ensure indicators start after the 7th column.")

    indicator_columns = data.columns[7:]
    st.write(f"Indicator columns: {indicator_columns}")

    imputer = SimpleImputer(strategy='mean')
    indicators_imputed = imputer.fit_transform(data[indicator_columns])

    data['result'] = data['result'].apply(lambda x: 1 if x == 'win' else 0)

    labeled_data = data.sample(frac=0.1, random_state=42)
    unlabeled_data = data.drop(labeled_data.index)

    X_labeled = labeled_data[indicator_columns]
    y_labeled = labeled_data['result']

    X_unlabeled = unlabeled_data[indicator_columns]

    X_labeled_imputed = imputer.transform(X_labeled)
    X_unlabeled_imputed = imputer.transform(X_unlabeled)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_labeled_imputed, y_labeled, test_size=0.2, random_state=42)
    
    st.write("Data preprocessing completed.")
    return data, X_train, X_test, y_train, y_test, indicator_columns

def run_advanced_models_dashboard():
    st.write("Starting advanced models dashboard...")
    
    def load_and_preprocess_data(base_dir):
        st.write(f"Base directory: {base_dir}")
        data_dir = os.path.join(base_dir, "data/processed")
        data = load_data(data_dir)
        return preprocess_data(data)

    st.title("Advanced Models and Hybrid Strategies Dashboard")

    if "base_dir" not in st.session_state:
        st.session_state.base_dir = "."

    base_dir = st.text_input("Base Directory", value=st.session_state.base_dir)

    if st.button("Load Data"):
        st.write("Loading data...")
        try:
            data, X_train, X_test, y_train, y_test, indicator_columns = load_and_preprocess_data(base_dir)
            st.success("Data loaded and preprocessed successfully.")
            st.session_state.data = data
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.indicator_columns = indicator_columns

            results = []
            feature_importances = {}

            classifiers = {
                'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10),
                'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),
                'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=0.1),
                'LogisticRegression': LogisticRegression(max_iter=1000),
                'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, use_label_encoder=False, eval_metric='logloss')
            }

            for clf_name, clf in tqdm(classifiers.items(), desc="Training classifiers"):
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
                classifier = st.selectbox("Select Classifier for Feature Importance", list(st.session_state.feature_importances.keys()))

                if classifier:
                    importance = st.session_state.feature_importances[classifier]
                    importance_df = pd.DataFrame({
                        'Feature': st.session_state.indicator_columns,
                        'Importance': importance
                    }).sort_values(by='Importance', ascending=False)

                    top_n = st.selectbox("Select Top N Indicators", [3, 5, 10, len(importance_df)], index=2)
                    selected_features = importance_df.head(top_n)

                    fig = px.bar(selected_features, x='Importance', y='Feature', orientation='h')
                    st.plotly_chart(fig)

            with st.expander("Optimal Win Ranges"):
                st.subheader("Calculate Optimal Win Ranges")

                trade_type = st.selectbox("Select Trade Type", ["Long Only", "Short Only", "Long & Short"])
                model_name = st.selectbox("Select Model", ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"])
                top_n = st.selectbox("Select Top N Indicators", [None, 10, 5, 3, "ALL"], key='win_ranges_top_n')
                individual_indicator = st.selectbox("Select Individual Indicator", st.session_state.indicator_columns, key='individual_win_ranges')
                
                if st.button("Calculate and Plot Optimal Win Ranges"):
                    if top_n:
                        selected_features = st.session_state.feature_importances[model_name].index if top_n == "ALL" else st.session_state.feature_importances[model_name].index[:top_n]
                    else:
                        selected_features = [individual_indicator]

                    optimal_ranges = calculate_optimal_win_ranges(st.session_state.data, features=selected_features)
                    plot_optimal_win_ranges(st.session_state.data, optimal_ranges, trade_type=trade_type, model_name=model_name)
                    
                    optimal_win_ranges_summary = summarize_optimal_win_ranges(optimal_ranges)
                    st.write(optimal_win_ranges_summary)
                    
                    output_path = os.path.join(base_dir, f'docs/ml_analysis/win_ranges_summary/optimal_win_ranges_summary_{model_name}.csv')
                    optimal_win_ranges_summary.to_csv(output_path, index=False)
                    st.write(f"Saved optimal win ranges summary to {output_path}")

        with col2:
            with st.expander("Individual Indicator Analysis"):
                model_name = st.selectbox("Select Model", list(classifiers.keys()), key='individual_indicator_model')
                individual_indicator = st.selectbox("Select Individual Indicator", st.session_state.indicator_columns, key='individual_indicator')
                
                if st.button("Plot Indicator Distribution", key='plot_individual'):
                    fig = px.histogram(st.session_state.data, x=individual_indicator, color='result', marginal="rug", hover_data=st.session_state.data.columns)
                    st.plotly_chart(fig)

                if st.button("Plot Win vs Loss Distribution for All Indicators"):
                    win_loss_counts = st.session_state.data['result'].value_counts()
                    fig = px.pie(values=win_loss_counts, names=win_loss_counts.index, title=f"Win vs Loss Distribution for {individual_indicator}")
                    st.plotly_chart(fig)

            with st.expander("Multiple Indicators Analysis"):
                selected_indicators = st.multiselect("Select Indicators to Analyze Together", st.session_state.indicator_columns, key='multiple')
                
                if st.button("Plot Multiple Indicators", key='plot_multiple'):
                    if len(selected_indicators) > 1:
                        fig = px.scatter_matrix(st.session_state.data, dimensions=selected_indicators, color='result')
                        st.plotly_chart(fig)
                    else:
                        st.warning("Please select at least two indicators.")

        with st.expander("Additional EDA"):
            st.subheader("Additional Exploratory Data Analysis")
            with st.spinner("Generating additional EDA plots..."):
                # Correlation matrix
                selected_model = st.selectbox("Select Model for Correlation", list(st.session_state.feature_importances.keys()), key='correlation_model')
                feature_importances = st.session_state.feature_importances[selected_model]
                feature_importances = pd.DataFrame({
                    'Feature': st.session_state.indicator_columns,
                    'Importance': feature_importances
                }).sort_values(by='Importance', ascending=False)

                top_n_features = st.selectbox("Select Top N Features for Correlation Matrix", [3, 5, 10], index=2)
                top_features = feature_importances.head(top_n_features)
                top_features_list = top_features['Feature'].tolist()

                st.write("Top Features for Correlation Matrix:")
                st.write(top_features_list)

                corr = st.session_state.data[top_features_list].corr()
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f', ax=ax)
                st.write("Correlation Matrix of Top Features")
                st.pyplot(fig)

                # Pie chart of wins vs losses for top features
                st.write("Win vs Loss Distribution for Top Features")
                for feature in top_features_list:
                    win_loss_counts = st.session_state.data[feature].value_counts()
                    fig = px.pie(values=win_loss_counts, names=win_loss_counts.index, title=f"Win vs Loss Distribution for {feature}")
                    st.plotly_chart(fig)

if __name__ == "__main__":
    run_advanced_models_dashboard()
