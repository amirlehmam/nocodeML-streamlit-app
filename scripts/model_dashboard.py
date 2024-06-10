# model_dashboard.py
import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
def load_data(data_dir):
    st.write(f"Loading data from {data_dir}...")  # Debug statement
    data = pd.read_csv(os.path.join(data_dir, "merged_trade_indicator_event.csv"))
    st.write(f"Data loaded with shape: {data.shape}")  # Debug statement
    return data

def preprocess_data(data):
    st.write("Preprocessing data...")  # Debug statement
    # Ensure the data contains the expected columns
    if data.shape[1] < 8:
        raise ValueError("Data does not have enough columns for indicators. Ensure indicators start after the 7th column.")

    # Select indicator columns from the data (after the first 7 columns)
    indicator_columns = data.columns[7:]
    st.write(f"Indicator columns: {indicator_columns}")  # Debug statement

    # Handle missing values by imputing them with the mean of the column
    imputer = SimpleImputer(strategy='mean')
    indicators_imputed = imputer.fit_transform(data[indicator_columns])

    # Encode target variable
    data['result'] = data['result'].apply(lambda x: 1 if x == 'win' else 0)

    # Create labeled and unlabeled datasets by random selection
    labeled_data = data.sample(frac=0.1, random_state=42)
    unlabeled_data = data.drop(labeled_data.index)

    X_labeled = labeled_data[indicator_columns]
    y_labeled = labeled_data['result']

    X_unlabeled = unlabeled_data[indicator_columns]

    # Impute missing values in both labeled and unlabeled datasets
    X_labeled_imputed = imputer.transform(X_labeled)
    X_unlabeled_imputed = imputer.transform(X_unlabeled)

    # Split labeled data into training and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_labeled_imputed, y_labeled, test_size=0.2, random_state=42)
    
    st.write("Data preprocessing completed.")  # Debug statement
    return data, X_train, X_test, y_train, y_test, indicator_columns

def run_model_dashboard():
    st.write("Starting model dashboard...")  # Debug statement
    # Define classifiers with parameter tuning
    classifiers = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, learning_rate=0.1),
        'SVC': SVC(probability=True, C=1.0, kernel='linear'),
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, use_label_encoder=False, eval_metric='logloss')
    }

    # Load data with base_dir and data_dir
    def load_and_preprocess_data(base_dir):
        st.write(f"Base directory: {base_dir}")  # Debug statement
        data_dir = os.path.join(base_dir, "data/processed")
        data = load_data(data_dir)
        return preprocess_data(data)

    # Title
    st.title("Advanced Trading Dashboard")

    # Sidebar for Base Directory input
    if "base_dir" not in st.session_state:
        st.session_state.base_dir = "."

    base_dir = st.text_input("Base Directory", value=st.session_state.base_dir)

    # Load and preprocess data
    if st.button("Load Data"):
        st.write("Loading data...")  # Debug statement
        try:
            data, X_train, X_test, y_train, y_test, indicator_columns = load_and_preprocess_data(base_dir)
            st.success("Data loaded and preprocessed successfully.")
            st.session_state.data = data
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.indicator_columns = indicator_columns

            # Train models and get feature importances
            results = []
            feature_importances = {}

            for clf_name, clf in classifiers.items():
                st.write(f"Training {clf_name}...")  # Debug statement
                clf.fit(X_train, y_train)
                accuracy = clf.score(X_test, y_test)
                results.append((clf_name, accuracy))
                if hasattr(clf, 'feature_importances_'):
                    feature_importances[clf_name] = clf.feature_importances_

            results_df = pd.DataFrame(results, columns=['Classifier', 'Accuracy'])
            results_df.sort_values(by='Accuracy', ascending=False, inplace=True)
            st.session_state.results_df = results_df
            st.session_state.feature_importances = feature_importances

        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

    # Check if data is loaded and preprocessed
    if "results_df" in st.session_state and "feature_importances" in st.session_state:
        # Display model results
        st.header("Model Accuracy")
        st.dataframe(st.session_state.results_df)

        # Feature Importance
        st.header("Feature Importance")
        classifier = st.selectbox("Select Classifier for Feature Importance", list(st.session_state.feature_importances.keys()))

        if classifier:
            importance = st.session_state.feature_importances[classifier]
            importance_df = pd.DataFrame({
                'Feature': st.session_state.indicator_columns,
                'Importance': importance
            }).sort_values(by='Importance', ascending=False)

            top_n = st.selectbox("Select Top N Indicators", [3, 5, 10, len(importance_df)], index=2)
            selected_features = importance_df.head(top_n)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=selected_features, ax=ax)
            st.pyplot(fig)

        # Dropdown for individual indicator analysis
        st.header("Individual Indicator Analysis")
        individual_indicator = st.selectbox("Select Individual Indicator", st.session_state.indicator_columns)
        
        if st.button("Plot Indicator Distribution"):
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(st.session_state.data[individual_indicator], kde=True, ax=ax)
            ax.set_title(f'Distribution of {individual_indicator}')
            st.pyplot(fig)

        # Dropdown for analyzing multiple indicators together
        st.header("Multiple Indicators Analysis")
        selected_indicators = st.multiselect("Select Indicators to Analyze Together", st.session_state.indicator_columns)
        
        if st.button("Plot Multiple Indicators"):
            if len(selected_indicators) > 1:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.pairplot(st.session_state.data[selected_indicators])
                st.pyplot(fig)
            else:
                st.warning("Please select at least two indicators.")

        # Winning Range Values
        st.header("Winning Range Values")
        top_indicators = st.selectbox("Select Top N Indicators for Winning Range Analysis", [3, 5, 10, len(st.session_state.indicator_columns)], index=2)
        selected_top_features = importance_df.head(top_indicators)['Feature'].tolist()

        if st.button("Plot Winning Range Values"):
            winning_data = st.session_state.data[st.session_state.data['result'] == 1]
            losing_data = st.session_state.data[st.session_state.data['result'] == 0]

            fig, ax = plt.subplots(figsize=(10, 6))
            for feature in selected_top_features:
                sns.kdeplot(winning_data[feature], label=f'{feature} (Winning)', ax=ax)
                sns.kdeplot(losing_data[feature], label=f'{feature} (Losing)', ax=ax)
            ax.set_xlabel('Indicator Value')
            ax.set_ylabel('Density')
            ax.set_title('Winning vs Losing Indicator Value Distribution')
            ax.legend()
            st.pyplot(fig)

            # Print Winning Range Values
            winning_ranges = winning_data[selected_top_features].describe()
            st.write("Winning Range Values for Selected Indicators:")
            st.write(winning_ranges)

# If running this script directly
if __name__ == "__main__":
    run_model_dashboard()
