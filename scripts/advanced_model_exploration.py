# advanced_model_exploration.py
import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten
from scikeras.wrappers import KerasClassifier
from tqdm import tqdm
from tqdm.keras import TqdmCallback
from scipy.stats import gaussian_kde
import shap

# Load data
def load_data(data_dir):
    st.write(f"Loading data from {data_dir}...")
    data = pd.read_csv(os.path.join(data_dir, "merged_trade_indicator_event.csv"))
    st.write(f"Data loaded with shape: {data.shape}")
    return data

def preprocess_data(data, selected_feature_types):
    st.write("Preprocessing data...")
    if data.shape[1] < 8:
        raise ValueError("Data does not have enough columns for indicators. Ensure indicators start after the 7th column.")

    all_indicators = data.columns[7:]
    st.write(f"All indicators: {all_indicators}")

    # Filter features based on user selection
    indicator_columns = []
    if "Non-Market Value Data" in selected_feature_types:
        indicator_columns.extend([col for col in all_indicators if "_percent_away" not in col and "_binary" not in col])
    if "Percent Away Indicators" in selected_feature_types:
        indicator_columns.extend([col for col in all_indicators if "_percent_away" in col])
    if "Binary Indicators" in selected_feature_types:
        indicator_columns.extend([col for col in all_indicators if "_binary" in col])

    st.write(f"Selected indicators: {indicator_columns}")

    data['result'] = data['result'].apply(lambda x: 1 if x == 'win' else 0)

    # Fill NaN values with column mean
    data[indicator_columns] = data[indicator_columns].apply(lambda col: col.fillna(col.mean()))

    X = data[indicator_columns]
    y = data['result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.write("Data preprocessing completed.")
    return X_train_scaled, X_test_scaled, y_train, y_test, indicator_columns

# Define Keras models
def create_nn_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_rnn_model(input_shape, rnn_type='LSTM'):
    model = Sequential()
    if rnn_type == 'LSTM':
        model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    else:
        model.add(GRU(64, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(32))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def calculate_optimal_win_ranges(data, target='result', features=None):
    optimal_ranges = []

    if features is None:
        features = data.columns.drop([target])

    for feature in tqdm(features, desc="Calculating Optimal Win Ranges"):
        data[feature] = pd.to_numeric(data[feature], errors='coerce')
        
        win_values = data[data[target] == 0][feature].dropna().values.astype(float)
        loss_values = data[data[target] == 1][feature].dropna().values.astype(float)

        if len(win_values) == 0 or len(loss_values) == 0:
            continue

        win_kde = gaussian_kde(win_values)
        loss_kde = gaussian_kde(loss_values)

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

def plot_optimal_win_ranges(data, optimal_ranges, target='result', trade_type='', model_name=''):
    for item in optimal_ranges:
        feature = item['feature']
        ranges = item['optimal_win_ranges']
        
        win_values = data[data[target] == 0][feature].dropna()
        loss_values = data[data[target] == 1][feature].dropna()
        
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

def run_advanced_model_exploration():
    st.title("Advanced Model Exploration")

    if "base_dir" not in st.session_state:
        st.session_state.base_dir = "./data/processed"

    base_dir = st.text_input("Base Directory", value=st.session_state.base_dir)

    # Initialize feature_importances in session state
    if "feature_importances" not in st.session_state:
        st.session_state.feature_importances = {}

    # Initialize session state for current step and model type
    if "current_step" not in st.session_state:
        st.session_state.current_step = "load_data"
    if "model_type" not in st.session_state:
        st.session_state.model_type = "Random Forest"

    # Add multiselect for feature types
    selected_feature_types = st.multiselect(
        "Select Feature Types",
        ["Non-Market Value Data", "Percent Away Indicators", "Binary Indicators"],
        ["Non-Market Value Data", "Percent Away Indicators", "Binary Indicators"]
    )

    if st.session_state.current_step == "load_data" and st.button("Load Data"):
        st.write("Loading data...")
        try:
            data = load_data(base_dir)
            X_train, X_test, y_train, y_test, indicator_columns = preprocess_data(data, selected_feature_types)
            st.session_state.data = data
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.indicator_columns = indicator_columns
            st.session_state.current_step = "model_selection"
            st.success("Data loaded and preprocessed successfully.")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

    if "data" in st.session_state and "X_train" in st.session_state and st.session_state.current_step == "model_selection":
        st.write("Select model and hyperparameters for exploration")

        model_type = st.selectbox("Select Model Type", ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "Neural Network", "RNN (LSTM)", "RNN (GRU)", "CNN", "Stacking Ensemble"], key="model_type_select")
        st.session_state.model_type_selected = model_type

        model_params = {}
        model = None  # Initialize model as None
        if model_type == "Random Forest":
            st.subheader("Random Forest Parameters")
            model_params['n_estimators'] = st.slider("Number of Trees", min_value=10, max_value=500, value=100)
            model_params['max_depth'] = st.slider("Max Depth of Trees", min_value=1, max_value=20, value=10)
            model = RandomForestClassifier(n_estimators=model_params['n_estimators'], max_depth=model_params['max_depth'], random_state=42)
        
        elif model_type == "Gradient Boosting":
            st.subheader("Gradient Boosting Parameters")
            model_params['n_estimators'] = st.slider("Number of Trees", min_value=10, max_value=500, value=100)
            model_params['learning_rate'] = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1)
            model_params['max_depth'] = st.slider("Max Depth of Trees", min_value=1, max_value=20, value=3)
            model = GradientBoostingClassifier(n_estimators=model_params['n_estimators'], learning_rate=model_params['learning_rate'], max_depth=model_params['max_depth'], random_state=42)

        elif model_type == "XGBoost":
            st.subheader("XGBoost Parameters")
            model_params['n_estimators'] = st.slider("Number of Trees", min_value=10, max_value=500, value=100)
            model_params['learning_rate'] = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1)
            model_params['max_depth'] = st.slider("Max Depth of Trees", min_value=1, max_value=20, value=3)
            model = xgb.XGBClassifier(n_estimators=model_params['n_estimators'], learning_rate=model_params['learning_rate'], max_depth=model_params['max_depth'], use_label_encoder=False, eval_metric='logloss', random_state=42)

        elif model_type == "LightGBM":
            st.subheader("LightGBM Parameters")
            model_params['n_estimators'] = st.slider("Number of Trees", min_value=10, max_value=500, value=100)
            model_params['learning_rate'] = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1)
            model_params['max_depth'] = st.slider("Max Depth of Trees", min_value=1, max_value=20, value=3)
            model = lgb.LGBMClassifier(n_estimators=model_params['n_estimators'], learning_rate=model_params['learning_rate'], max_depth=model_params['max_depth'], random_state=42)

        elif model_type == "Neural Network":
            st.subheader("Neural Network Parameters")
            model_params['epochs'] = st.slider("Number of Epochs", min_value=10, max_value=1000, value=100)
            model_params['batch_size'] = st.slider("Batch Size", min_value=10, max_value=128, value=32)
            input_dim = st.session_state.X_train.shape[1]
            model = KerasClassifier(model=create_nn_model, model__input_dim=input_dim, epochs=model_params['epochs'], batch_size=model_params['batch_size'], verbose=0)

        elif model_type == "RNN (LSTM)":
            st.subheader("RNN (LSTM) Parameters")
            model_params['epochs'] = st.slider("Number of Epochs", min_value=10, max_value=1000, value=100)
            model_params['batch_size'] = st.slider("Batch Size", min_value=10, max_value=128, value=32)
            input_shape = (st.session_state.X_train.shape[1], 1)
            model = KerasClassifier(model=create_rnn_model, model__input_shape=input_shape, model__rnn_type='LSTM', epochs=model_params['epochs'], batch_size=model_params['batch_size'], verbose=0)

        elif model_type == "RNN (GRU)":
            st.subheader("RNN (GRU) Parameters")
            model_params['epochs'] = st.slider("Number of Epochs", min_value=10, max_value=1000, value=100)
            model_params['batch_size'] = st.slider("Batch Size", min_value=10, max_value=128, value=32)
            input_shape = (st.session_state.X_train.shape[1], 1)
            model = KerasClassifier(model=create_rnn_model, model__input_shape=input_shape, model__rnn_type='GRU', epochs=model_params['epochs'], batch_size=model_params['batch_size'], verbose=0)

        elif model_type == "CNN":
            st.subheader("CNN Parameters")
            model_params['epochs'] = st.slider("Number of Epochs", min_value=10, max_value=1000, value=100)
            model_params['batch_size'] = st.slider("Batch Size", min_value=10, max_value=128, value=32)
            input_shape = (st.session_state.X_train.shape[1], 1)
            model = KerasClassifier(model=create_cnn_model, model__input_shape=input_shape, epochs=model_params['epochs'], batch_size=model_params['batch_size'], verbose=0)
        
        elif model_type == "Stacking Ensemble":
            st.subheader("Stacking Ensemble Parameters")
            base_learners = [
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
                ('xgb', xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, use_label_encoder=False, eval_metric='logloss', random_state=42))
            ]
            final_estimator = LogisticRegression(max_iter=1000)
            model = StackingClassifier(estimators=base_learners, final_estimator=final_estimator)

        if st.button("Train Model"):
            st.write(f"Training {model_type}...")
            with st.spinner("Training in progress..."):
                try:
                    if model_type in ["Neural Network", "RNN (LSTM)", "RNN (GRU)", "CNN"]:
                        model.fit(st.session_state.X_train, st.session_state.y_train, callbacks=[TqdmCallback(verbose=1)])
                    else:
                        for _ in tqdm(range(1), desc=f"Training {model_type}"):
                            model.fit(st.session_state.X_train, st.session_state.y_train)
                    y_pred = model.predict(st.session_state.X_test)
                    accuracy = accuracy_score(st.session_state.y_test, y_pred)
                    st.write(f"Accuracy: {accuracy}")
                    st.write("Classification Report:")
                    st.text(classification_report(st.session_state.y_test, y_pred))
                    st.write("Confusion Matrix:")
                    cm = confusion_matrix(st.session_state.y_test, y_pred)
                    fig = px.imshow(cm, labels=dict(x="Predicted", y="True", color="Count"), x=["Negative", "Positive"], y=["Negative", "Positive"], color_continuous_scale='Blues')
                    st.plotly_chart(fig)

                    # Save the model
                    model_save_path = os.path.join(base_dir, f"models/{model_type}_model.pkl")
                    pd.to_pickle(model, model_save_path)
                    st.write(f"Model saved to {model_save_path}")

                    # Feature importance
                    importance_df = pd.DataFrame({
                        'Feature': st.session_state.indicator_columns,
                        'Importance': [0] * len(st.session_state.indicator_columns)
                    })
                    if model_type not in ["Neural Network", "RNN (LSTM)", "RNN (GRU)", "CNN"] and hasattr(model, 'feature_importances_'):
                        st.write("Feature Importances:")
                        feature_importances = model.feature_importances_
                        importance_df['Importance'] = feature_importances
                        importance_df.sort_values(by='Importance', ascending=False, inplace=True)
                        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
                        st.plotly_chart(fig)
                        
                        # Initialize feature_importances in session state
                        st.session_state.feature_importances[model_type] = feature_importances
                    elif model_type in ["Neural Network", "RNN (LSTM)", "RNN (GRU)", "CNN"]:
                        st.write("Calculating feature importances using SHAP...")
                        # Extract the underlying Keras model from the KerasClassifier
                        underlying_model = model.model_
                        explainer = shap.Explainer(underlying_model, st.session_state.X_train)
                        shap_values = explainer(st.session_state.X_test)
                        feature_importances = np.abs(shap_values.values).mean(axis=0)
                        importance_df['Importance'] = feature_importances
                        importance_df.sort_values(by='Importance', ascending=False, inplace=True)
                        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h')
                        st.plotly_chart(fig)
                        
                        # Initialize feature_importances in session state
                        st.session_state.feature_importances[model_type] = feature_importances

                    st.session_state.current_step = "eda"
                    st.session_state.model_type_selected = model_type

                    # Optimal Win Ranges
                    st.subheader("Optimal Win Ranges")
                    top_n = st.selectbox("Select Top N Indicators", [3, 5, 10, len(st.session_state.indicator_columns)], index=2)
                    if not importance_df.empty:
                        selected_features = importance_df.head(top_n)['Feature']
                    else:
                        selected_features = st.session_state.indicator_columns[:top_n]

                    optimal_ranges = calculate_optimal_win_ranges(st.session_state.data, features=selected_features)
                    st.session_state.optimal_ranges = optimal_ranges
                    plot_optimal_win_ranges(st.session_state.data, optimal_ranges, trade_type='', model_name=model_type)

                    optimal_win_ranges_summary = summarize_optimal_win_ranges(optimal_ranges)
                    st.write(optimal_win_ranges_summary)
                    output_path = os.path.join(base_dir, f'docs/ml_analysis/win_ranges_summary/optimal_win_ranges_summary_{model_type}.csv')
                    optimal_win_ranges_summary.to_csv(output_path, index=False)
                    st.write(f"Saved optimal win ranges summary to {output_path}")

                except Exception as e:
                    st.error(f"Error during model training: {e}")

    if st.session_state.current_step == "eda":
        # Additional Exploratory Data Analysis
        st.subheader("Additional Exploratory Data Analysis")
        with st.spinner("Generating additional EDA plots..."):
            model_type = st.session_state.model_type_selected  # Ensure model_type is retrieved from session state
            optimal_ranges = st.session_state.optimal_ranges  # Ensure optimal_ranges is retrieved from session state

            # Feature importance heatmap
            if model_type in st.session_state.feature_importances:
                st.write("Feature Importance Heatmap:")
                feature_importance_values = st.session_state.feature_importances[model_type]
                fig = px.imshow([feature_importance_values], labels=dict(x="Features", color="Importance"), x=st.session_state.indicator_columns, color_continuous_scale='Blues')
                st.plotly_chart(fig)

            # Correlation matrix
            st.write("Correlation Matrix of Top Indicators:")
            selected_model = st.selectbox("Select Model for Correlation", list(st.session_state.feature_importances.keys()), key='correlation_model')
            if selected_model in st.session_state.feature_importances:
                top_features = st.session_state.feature_importances[selected_model][:10]
                top_features_list = [st.session_state.indicator_columns[i] for i in np.argsort(top_features)[::-1][:10]]
                corr = st.session_state.data[top_features_list].corr()
                fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='Blues')
                st.plotly_chart(fig)

            # Detailed Indicator Analysis
            st.write("Detailed Indicator Analysis")
            selected_indicator = st.selectbox("Select Indicator for Detailed Analysis", st.session_state.indicator_columns)
            if selected_indicator:
                win_data = st.session_state.data[st.session_state.data['result'] == 0][selected_indicator].dropna()
                loss_data = st.session_state.data[st.session_state.data['result'] == 1][selected_indicator].dropna()

                fig = go.Figure()
                fig.add_trace(go.Histogram(x=win_data, name='Win', marker_color='blue', opacity=0.75))
                fig.add_trace(go.Histogram(x=loss_data, name='Loss', marker_color='red', opacity=0.75))
                fig.update_layout(barmode='overlay', title_text=f'Distribution of {selected_indicator} for Winning and Losing Trades', width=800, height=400)
                fig.update_xaxes(title_text=selected_indicator)
                fig.update_yaxes(title_text='Count')
                st.plotly_chart(fig)

                # KDE plot with winning ranges
                kde_win = gaussian_kde(win_data)
                kde_loss = gaussian_kde(loss_data)
                x_grid = np.linspace(min(st.session_state.data[selected_indicator].dropna()), max(st.session_state.data[selected_indicator].dropna()), 1000)
                kde_win_density = kde_win(x_grid)
                kde_loss_density = kde_loss(x_grid)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_grid, y=kde_win_density, mode='lines', name='Win', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=x_grid, y=kde_loss_density, mode='lines', name='Loss', line=dict(color='red')))
                for range_item in optimal_ranges:
                    if range_item['feature'] == selected_indicator:
                        for start, end in range_item['optimal_win_ranges']:
                            fig.add_vrect(x0=start, x1=end, fillcolor="blue", opacity=0.3, line_width=0)

                fig.update_layout(title_text=f'KDE Plot with Optimal Win Ranges for {selected_indicator}', xaxis_title=selected_indicator, yaxis_title='Density', width=800, height=400)
                st.plotly_chart(fig)

            # Feature Importance vs Prediction
            st.write("Feature Importance vs Prediction Analysis")
            if 'importance_df' in locals() and not importance_df.empty:
                feature_importance_threshold = st.slider("Select Feature Importance Threshold", min_value=0.0, max_value=float(importance_df['Importance'].max()), value=0.1)
                important_features = importance_df[importance_df['Importance'] >= feature_importance_threshold]['Feature']
                if not important_features.empty:
                    for feature in important_features:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=st.session_state.data[feature], y=st.session_state.data['result'], mode='markers', name='Actual', marker=dict(color='blue')))
                        fig.add_trace(go.Scatter(x=st.session_state.data[feature], y=y_pred, mode='markers', name='Predicted', marker=dict(color='red')))
                        fig.update_layout(title_text=f'Actual vs Predicted Results for {feature}', xaxis_title=feature, yaxis_title='Result')
                        st.plotly_chart(fig)

            st.success("EDA plots generated successfully.")

if __name__ == "__main__":
    run_advanced_model_exploration()
