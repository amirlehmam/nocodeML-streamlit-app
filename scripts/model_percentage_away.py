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
def randomized_search_model(model, param_distributions, X_train, y_train, n_iter=2):
    random_search = RandomizedSearchCV(model, param_distributions, n_iter=n_iter, cv=3, scoring='accuracy', n_jobs=1, random_state=42)
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_

# Train models with hyperparameter tuning
def train_models(X_train, y_train):
    param_distributions_rf = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }

    param_distributions_gb = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }

    param_distributions_xgb = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }

    param_distributions_lgb = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 50]
    }

    st.write("Starting Random Forest tuning...")
    best_rf = randomized_search_model(RandomForestClassifier(random_state=42), param_distributions_rf, X_train, y_train)
    st.write("Random Forest tuning completed.")

    st.write("Starting Gradient Boosting tuning...")
    best_gb = randomized_search_model(GradientBoostingClassifier(random_state=42), param_distributions_gb, X_train, y_train)
    st.write("Gradient Boosting tuning completed.")

    st.write("Starting XGBoost tuning...")
    best_xgb = randomized_search_model(xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), param_distributions_xgb, X_train, y_train)
    st.write("XGBoost tuning completed.")

    st.write("Starting LightGBM tuning...")
    best_lgb = randomized_search_model(lgb.LGBMClassifier(random_state=42), param_distributions_lgb, X_train, y_train)
    st.write("LightGBM tuning completed.")
    
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
def plot_shap_importance(model, X_train, percent_away_features, model_name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar", feature_names=percent_away_features)
    plt.title(f'SHAP Feature Importance - {model_name}')
    st.pyplot(plt.gcf())
    plt.close()

# Plot top features
def plot_top_features(model, feature_names, num_features=10, model_name=None):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-num_features:]
    plt.figure(figsize=(10, 8))
    plt.title(f"Feature Importances - {model_name}")
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    st.pyplot(plt.gcf())
    plt.close()

# Advanced EDA for percent away values
def advanced_eda_percent_away(data, feature_importances, model_name, target='result', save_path=None):
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    top_features = feature_importances.head(10)['feature']
    top_features = [feature for feature in top_features if feature in data.columns]
    
    if not top_features:
        st.write("No matching features found in the data.")
        return

    descriptive_stats = data[top_features].describe()
    st.write(f"\nDescriptive Statistics for Top Features ({model_name}):")
    st.write(descriptive_stats)

    corr_matrix = data[top_features].corr()
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Correlation Matrix of Top Features ({model_name})')
    if save_path:
        plt.savefig(os.path.join(save_path, 'correlation_matrix.png'))
    st.pyplot(plt.gcf())
    plt.close()

    for feature in top_features:
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        sns.histplot(data=data, x=feature, hue=target, kde=True, element="step", stat="density", common_norm=False)
        plt.title(f'Distribution of {feature} ({model_name})')

        plt.subplot(1, 2, 2)
        sns.boxplot(x=target, y=feature, data=data)
        plt.title(f'{feature} by {target} ({model_name})')
        
        if save_path:
            plt.savefig(os.path.join(save_path, f'{feature}_distribution.png'))
        plt.close()

        win_values = data[data[target] == 0][feature]
        loss_values = data[data[target] == 1][feature]
        st.write(f"\n{feature} Statistics ({model_name}):")
        st.write(f"Mean (Win): {win_values.mean():.2f}, Mean (Loss): {loss_values.mean():.2f}")
        st.write(f"Median (Win): {win_values.median():.2f}, Median (Loss): {loss_values.median():.2f}")
        st.write(f"Standard Deviation (Win): {win_values.std():.2f}, Standard Deviation (Loss): {loss_values.std():.2f}")
        st.write(f"Skewness (Win): {win_values.skew():.2f}, Skewness (Loss): {loss_values.skew():.2f}")
        st.write(f"Kurtosis (Win): {win_values.kurtosis():.2f}, Kurtosis (Loss): {loss_values.kurtosis():.2f}")
        
        t_stat, p_value = stats.ttest_ind(win_values, loss_values, equal_var=False)
        st.write(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.3f}")
        
        sns.kdeplot(win_values, label='Win', fill=True)
        sns.kdeplot(loss_values, label='Loss', fill=True)
        plt.title(f'KDE Plot of {feature} for Win and Loss ({model_name})')
        plt.legend()
        
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
        
        best_model_name = max(results, key=lambda name: results[name]['accuracy'])
        best_model = models[best_model_name]
        
        st.write(f"The best model is: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")
        
        for name, metrics in results.items():
            st.write(f"\n{name} Evaluation:")
            for metric, value in metrics.items():
                if metric != 'classification_report':
                    st.write(f"{metric}: {value}")
                else:
                    st.write(f"\nClassification Report:\n{value}")

        st.write(f"Plotting SHAP feature importance for {best_model_name}...")
        plot_shap_importance(best_model, X_train, percent_away_features, best_model_name)

        st.write(f"Plotting top features for {best_model_name}...")
        plot_top_features(best_model, percent_away_features, model_name=best_model_name)

        st.write(f"Performing advanced EDA for percent away values with {best_model_name}...")
        analysis_save_path = os.path.join(data_dir, "ml_analysis_percent_away")
        advanced_eda_percent_away(merged_data, pd.DataFrame({'feature': percent_away_features, 'importance': best_model.feature_importances_}), best_model_name, save_path=analysis_save_path)

if __name__ == "__main__":
    run_model_percentage_away()
