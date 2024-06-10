# model_dashboard.py
import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Load data
def load_data(data_dir):
    st.write(f"Loading data from {data_dir}...")
    data = pd.read_csv(os.path.join(data_dir, "sample data.csv"))
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

def run_model_dashboard():
    st.write("Starting model dashboard...")
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=0.1),
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, use_label_encoder=False, eval_metric='logloss')
    }

    def load_and_preprocess_data(base_dir):
        st.write(f"Base directory: {base_dir}")
        data_dir = os.path.join(base_dir, "data/processed")
        data = load_data(data_dir)
        return preprocess_data(data)

    st.title("Advanced Trading Dashboard")

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

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Individual Indicator Analysis")
            individual_indicator = st.selectbox("Select Individual Indicator", st.session_state.indicator_columns)
            
            if st.button("Plot Indicator Distribution"):
                fig = px.histogram(st.session_state.data, x=individual_indicator, color='result', marginal="rug", hover_data=st.session_state.data.columns)
                st.plotly_chart(fig)

        with col2:
            st.subheader("Multiple Indicators Analysis")
            selected_indicators = st.multiselect("Select Indicators to Analyze Together", st.session_state.indicator_columns)
            
            if st.button("Plot Multiple Indicators"):
                if len(selected_indicators) > 1:
                    fig = px.scatter_matrix(st.session_state.data, dimensions=selected_indicators, color='result')
                    st.plotly_chart(fig)
                else:
                    st.warning("Please select at least two indicators.")

        with st.expander("Winning Range Values"):
            top_indicators = st.selectbox("Select Top N Indicators for Winning Range Analysis", [3, 5, 10, len(st.session_state.indicator_columns)], index=2)
            selected_top_features = importance_df.head(top_indicators)['Feature'].tolist()

            if st.button("Plot Winning Range Values"):
                winning_data = st.session_state.data[st.session_state.data['result'] == 1]
                losing_data = st.session_state.data[st.session_state.data['result'] == 0]

                fig = go.Figure()
                for feature in selected_top_features:
                    fig.add_trace(go.Violin(x=winning_data[feature], line=dict(color='green'), name=f'{feature} (Winning)', spanmode='hard'))
                    fig.add_trace(go.Violin(x=losing_data[feature], line=dict(color='red'), name=f'{feature} (Losing)', spanmode='hard'))
                fig.update_layout(title='Winning vs Losing Indicator Value Distribution', xaxis_title='Indicator Value', yaxis_title='Density')
                st.plotly_chart(fig)

                winning_ranges = winning_data[selected_top_features].describe()
                st.write("Winning Range Values for Selected Indicators:")
                st.write(winning_ranges)

if __name__ == "__main__":
    run_model_dashboard()
