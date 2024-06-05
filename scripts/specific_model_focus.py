import pandas as pd
import numpy as np
import os
import streamlit as st
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.impute import SimpleImputer
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from tqdm import tqdm
from scipy import stats
import time

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

# Grid search with progress bar
def grid_search_model(model, param_grid, X_train, y_train, X_test, y_test):
    n_iter = np.prod([len(v) for v in param_grid.values()])
    pbar = tqdm(total=n_iter, desc="Hyperparameter Tuning")
    
    best_estimator = None
    best_score = -np.inf
    start_time = time.time()

    for i, params in enumerate(ParameterGrid(param_grid), 1):
        model.set_params(**params)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        
        if score > best_score:
            best_score = score
            best_estimator = model
        
        elapsed_time = time.time() - start_time
        remaining_time = (elapsed_time / i) * (n_iter - i)
        pbar.set_postfix({
            'Best Score': best_score,
            'Elapsed Time': f'{elapsed_time:.2f}s',
            'Remaining Time': f'{remaining_time:.2f}s'
        })
        
        pbar.update(1)
    
    pbar.close()
    return best_estimator

# Train LightGBM model with hyperparameter tuning
def train_model(X_train, y_train, X_test, y_test):
    param_grid_lgb = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 50]
    }

    best_lgb = grid_search_model(lgb.LGBMClassifier(random_state=42), param_grid_lgb, X_train, y_train, X_test, y_test)
    return best_lgb

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }
    return metrics

# SHAP feature importance
def plot_shap_importance(model, X_train, percent_away_features):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar", feature_names=percent_away_features)

# Plot top features
def plot_top_features(importance_df, num_features=10):
    top_features = importance_df.head(num_features)
    plt.figure(figsize=(10, 8))
    plt.title("Top Feature Importances")
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.xlabel("Mean SHAP Value (Importance)")
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

        win_values = data[data[target] == 0][feature]
        loss_values = data[data[target] == 1][feature]
        st.write(f"\n{feature} Statistics:")
        st.write(f"Mean (Win): {win_values.mean():.2f}, Mean (Loss): {loss_values.mean():.2f}")
        st.write(f"Median (Win): {win_values.median():.2f}, Median (Loss): {loss_values.median():.2f}")
        st.write(f"Standard Deviation (Win): {win_values.std():.2f}, Standard Deviation (Loss): {loss_values.std():.2f}")
        st.write(f"Skewness (Win): {win_values.skew():.2f}, Skewness (Loss): {loss_values.skew():.2f}")
        st.write(f"Kurtosis (Win): {win_values.kurtosis():.2f}, Kurtosis (Loss): {loss_values.kurtosis():.2f}")
        
        t_stat, p_value = stats.ttest_ind(win_values, loss_values, equal_var=False)
        st.write(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.3f}")
        
        sns.kdeplot(win_values, label='Win', fill=True)
        sns.kdeplot(loss_values, label='Loss', fill=True)
        plt.title(f'KDE Plot of {feature} for Win and Loss')
        plt.legend()
        
        if save_path:
            plt.savefig(os.path.join(save_path, f'{feature}_kde_plot.png'))
        st.pyplot(plt.gcf())
        plt.close()

def run_specific_model_focus():
    st.subheader("Specific Model Focus")

    base_dir = st.text_input("Base Directory", "C:/Users/Administrator/Desktop/nocodeML")
    data_dir = os.path.join(base_dir, "data/processed")

    if st.button("Load Data"):
        merged_data = load_data(data_dir)
        (X_train, X_test, y_train, y_test), percent_away_features = prepare_percent_away_data(merged_data)
        
        st.write("Training LightGBM model with hyperparameter tuning...")
        best_lgb = train_model(X_train, y_train, X_test, y_test)
        
        st.write("Evaluating LightGBM model...")
        results = evaluate_model(best_lgb, X_test, y_test)
        
        for metric, value in results.items():
            st.write(f"{metric}: {value}")

        st.write("Plotting SHAP feature importance for LightGBM...")
        plot_shap_importance(best_lgb, X_train, percent_away_features)

        st.write("Extracting and ranking top features...")
        importance_df = pd.DataFrame({
            'feature': percent_away_features,
            'importance': np.abs(shap.TreeExplainer(best_lgb).shap_values(X_train)).mean(axis=0)
        }).sort_values(by='importance', ascending=False)

        st.write("\nTop 10 Important Features:")
        st.write(importance_df.head(10))

        st.write("Plotting top features...")
        plot_top_features(importance_df)

        st.write("Performing advanced EDA for percent away values...")
        analysis_save_path = os.path.join(data_dir, "ml_analysis_percent_away")
        advanced_eda_percent_away(merged_data, {'LightGBM': importance_df}, save_path=analysis_save_path)
