import pandas as pd
import numpy as np
import os
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
def load_data(data_dir):
    merged_data = pd.read_csv(os.path.join(data_dir, "merged_trade_indicator_event.csv"))
    return merged_data

# Prepare data
def prepare_percent_away_data(merged_data):
    percent_away_features = [col for col in merged_data.columns if 'percent_away' in col]
    X = merged_data[percent_away_features]
    y = merged_data['result'].map({'win': 0, 'loss': 1})
    
    # Impute NaN values with mean
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    return train_test_split(X_imputed, y, test_size=0.3, random_state=42), percent_away_features

# Randomized search with cross-validation
def randomized_search_model(model, param_distributions, X_train, y_train, n_iter=10):
    random_search = RandomizedSearchCV(model, param_distributions, n_iter=n_iter, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

# Train models with hyperparameter tuning
def train_models(X_train, y_train):
    param_distributions_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    param_distributions_gb = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }

    param_distributions_xgb = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }

    param_distributions_lgb = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'num_leaves': [31, 50, 100]
    }

    best_rf = randomized_search_model(RandomForestClassifier(random_state=42), param_distributions_rf, X_train, y_train)
    best_gb = randomized_search_model(GradientBoostingClassifier(random_state=42), param_distributions_gb, X_train, y_train)
    best_xgb = randomized_search_model(xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), param_distributions_xgb, X_train, y_train)
    best_lgb = randomized_search_model(lgb.LGBMClassifier(random_state=42), param_distributions_lgb, X_train, y_train)
    
    return {
        'Random Forest': best_rf,
        'Gradient Boosting': best_gb,
        'XGBoost': best_xgb,
        'LightGBM': best_lgb
    }

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'classification_report': classification_report(y_test, y_pred)
    }
    return metrics

# SHAP feature importance
def plot_shap_importance(model, X_train, percent_away_features):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar", feature_names=percent_away_features)

# Plot top features
def plot_top_features(model, feature_names, num_features=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-num_features:]
    plt.figure(figsize=(10, 8))
    plt.title("Feature Importances")
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    st.pyplot(plt.gcf())
    plt.close()

# Advanced EDA for percent away values
def advanced_eda_percent_away(data, feature_importances, target='result', save_path=None):
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    top_features = feature_importances['LightGBM']['feature'].head(10)
    top_features = [feature for feature in top_features if feature in data.columns]
    
    if not top_features:
        st.write("No matching features found in the data.")
        return

    descriptive_stats = data[top_features].describe()
    st.write("\nDescriptive Statistics for Top Features:")
    st.write(descriptive_stats)

    corr_matrix = data[top_features].corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Top Features')
    if save_path:
        plt.savefig(os.path.join(save_path, 'correlation_matrix.png'))
    st.pyplot(plt.gcf())
    plt.close()

    for feature in top_features:
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        sns.histplot(data=data, x=feature, hue=target, kde=True, element="step", stat="density", common_norm=False)
        plt.title(f'Distribution of {feature}')

        plt.subplot(1, 2, 2)
        sns.boxplot(x=target, y=feature, data=data)
        plt.title(f'{feature} by {target}')
        
        if save_path:
            plt.savefig(os.path.join(save_path, f'{feature}_distribution.png'))
        plt.close()
        
        if save_path:
            plt.savefig(os.path.join(save_path, f'{feature}_kde_plot.png'))
        st.pyplot(plt.gcf())
        plt.close()

def run_model_percentage_away():
    st.subheader("Model on % Away Indicators")

    if "base_dir" not in st.session_state:
        st.session_state.base_dir = "."

    base_dir = st.text_input("Base Directory", value=st.session_state.base_dir)
    data_dir = os.path.join(base_dir, "data/processed")

    if st.button("Load Data"):
        merged_data = load_data(data_dir)
        (X_train, X_test, y_train, y_test), percent_away_features = prepare_percent_away_data(merged_data)
        
        st.write("Training models with hyperparameter tuning...")
        models = train_models(X_train, y_train)
        
        st.write("Evaluating models...")
        results = {name: evaluate_model(model, X_test, y_test) for name, model in models.items()}
        
        for name, metrics in results.items():
            st.write(f"\n{name} Evaluation:")
            for metric, value in metrics.items():
                if metric != 'classification_report':
                    st.write(f"{metric}: {value}")
                else:
                    st.write(f"\nClassification Report:\n{value}")

        st.write("Plotting SHAP feature importance for LightGBM...")
        plot_shap_importance(models['LightGBM'], X_train, percent_away_features)

        st.write("Plotting top features for LightGBM...")
        plot_top_features(models['LightGBM'], percent_away_features)

        st.write("Performing advanced EDA for percent away values...")
        analysis_save_path = os.path.join(data_dir, "ml_analysis_percent_away")
        advanced_eda_percent_away(merged_data, {'LightGBM': pd.DataFrame({'feature': percent_away_features, 'importance': models['LightGBM'].feature_importances_})}, save_path=analysis_save_path)

if __name__ == "__main__":
    run_model_percentage_away()
