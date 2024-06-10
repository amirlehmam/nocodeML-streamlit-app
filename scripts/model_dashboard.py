# model_dashboard.py
import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
def load_data(data_dir):
    data = pd.read_csv(os.path.join(data_dir, "merged_trade_indicator_event.csv"))
    return data

def preprocess_data(data):
    # Ensure the data contains the expected columns
    if data.shape[1] < 8:
        raise ValueError("Data does not have enough columns for indicators. Ensure indicators start after the 7th column.")

    # Select indicator columns from the data (after the first 7 columns)
    indicator_columns = data.columns[7:]

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
    
    return data, X_train, X_test, y_train, y_test, indicator_columns

def run_model_dashboard():
    # Define classifiers
    classifiers = {
        'RandomForest': RandomForestClassifier(),
        'GradientBoosting': GradientBoostingClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'SVC': SVC(probability=True),
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'LightGBM': lgb.LGBMClassifier(),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    # Load data with base_dir and data_dir
    def load_and_preprocess_data(base_dir):
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
        data, X_train, X_test, y_train, y_test, indicator_columns = load_and_preprocess_data(base_dir)
        st.success("Data loaded and preprocessed successfully.")

        # Train models and get feature importances
        results = []
        feature_importances = {}

        for clf_name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            accuracy = clf.score(X_test, y_test)
            results.append((clf_name, accuracy))
            if hasattr(clf, 'feature_importances_'):
                feature_importances[clf_name] = clf.feature_importances_

        results_df = pd.DataFrame(results, columns=['Classifier', 'Accuracy'])
        results_df.sort_values(by='Accuracy', ascending=False, inplace=True)

        # Display model results
        st.header("Model Accuracy")
        st.dataframe(results_df)

        # Feature Importance
        st.header("Feature Importance")
        classifier = st.selectbox("Select Classifier for Feature Importance", list(feature_importances.keys()))

        if classifier:
            importance = feature_importances[classifier]
            importance_df = pd.DataFrame({
                'Feature': indicator_columns,
                'Importance': importance
            }).sort_values(by='Importance', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
            st.pyplot(fig)

        # Display data summary
        st.header("Data Summary")
        st.write(data.describe())

        # Plotting
        st.header("Feature Distribution")
        plt.figure(figsize=(10, 6))
        sns.histplot(data, kde=True)
        st.pyplot(plt)

        # Winning Range Values
        st.header("Winning Range Values")
        winning_data = data[data['result'] == 1]
        losing_data = data[data['result'] == 0]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.kdeplot(winning_data['price'], label='Winning', ax=ax)
        sns.kdeplot(losing_data['price'], label='Losing', ax=ax)
        ax.set_xlabel('Price')
        ax.set_ylabel('Density')
        ax.set_title('Winning vs Losing Trade Price Distribution')
        ax.legend()
        st.pyplot(fig)

        # Additional visualizations as needed

# If running this script directly
if __name__ == "__main__":
    run_model_dashboard()
