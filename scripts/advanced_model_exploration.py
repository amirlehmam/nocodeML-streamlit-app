import os
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
import keras
from keras import layers
import shap
from PIL import Image
from io import BytesIO
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    mean_squared_error, roc_curve, auc
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    StackingClassifier, RandomForestRegressor, 
    GradientBoostingRegressor, StackingRegressor
)
from sklearn.linear import LogisticRegression
from scikeras.wrappers import KerasClassifier, KerasRegressor
from tqdm import tqdm
from tqdm.keras import TqdmCallback
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import joblib

# Suppress TensorFlow warnings and messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Enable eager execution for TensorFlow 2.x
tf.config.experimental_run_functions_eagerly(True)

# Utility function to save plots to PDF
def save_plots_to_pdf(c, plots, descriptions):
    width, height = letter
    for plot, description in zip(plots, descriptions):
        c.drawString(10, height - 20, description)
        plot_data = BytesIO()
        img_bytes = plot.to_image(format='png')
        plot_data.write(img_bytes)
        plot_data.seek(0)
        img = Image.open(plot_data)
        img_width, img_height = img.size
        aspect = img_height / float(img_width)
        img_width = width - 20
        img_height = aspect * img_width

        if img_height > height - 60:
            img_height = height - 60
            img_width = img_height / aspect

        c.drawImage(ImageReader(img), 10, height - img_height - 30, width=img_width, height=img_height)
        c.showPage()

# Utility function to save text sections to PDF
def save_text_to_pdf(c, text_sections):
    width, height = letter
    for description, text in text_sections:
        c.drawString(10, height - 20, description)
        text_lines = text.split('\n')
        line_height = 12
        lines_per_page = (height - 40) // line_height
        current_line = 0

        for line in text_lines:
            if current_line * line_height + 30 > height:
                c.showPage()
                c.drawString(10, height - 20, description)
                current_line = 0
            c.drawString(10, height - 30 - current_line * line_height, line)
            current_line += 1
        c.showPage()

# Utility function to save dataframes to PDF
def save_dataframe_to_pdf(c, dataframes, descriptions):
    width, height = letter
    for df, description in zip(dataframes, descriptions):
        c.drawString(10, height - 20, description)
        text = df.to_string()
        text_lines = text.split('\n')
        line_height = 12
        lines_per_page = (height - 40) // line_height
        current_line = 0

        for line in text_lines:
            if current_line * line_height + 30 > height:
                c.showPage()
                c.drawString(10, height - 20, description)
                current_line = 0
            c.drawString(10, height - 30 - current_line * line_height, line)
            current_line += 1
        c.showPage()

# Utility function to save all elements to PDF
def save_all_to_pdf(pdf_filename, text_sections, dataframes, dataframe_descriptions, plots, plot_descriptions):
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    save_text_to_pdf(c, text_sections)
    save_dataframe_to_pdf(c, dataframes, dataframe_descriptions)
    save_plots_to_pdf(c, plots, plot_descriptions)
    c.save()

# Function to load data
def load_data(data_dir):
    st.write(f"Loading data from {data_dir}...")
    data_path = os.path.join(data_dir, "merged_trade_indicator_event.csv")
    if not os.path.exists(data_path):
        st.error(f"File not found: {data_path}")
        return None
    data = pd.read_csv(data_path)
    st.write(f"Data loaded with shape: {data.shape}")
    return data

# Function to preprocess data
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

    # Feature Engineering: Add derived metrics (changes, slopes)
    changes = data[indicator_columns].diff().add_suffix('_change')
    slopes = data[indicator_columns].diff().diff().add_suffix('_slope')
    
    data = pd.concat([data, changes, slopes], axis=1)

    # Fill NaN values with column mean
    data[indicator_columns] = data[indicator_columns].apply(lambda col: col.fillna(col.mean()))

    # Normalize the data
    scaler = StandardScaler()
    data[indicator_columns] = scaler.fit_transform(data[indicator_columns])

    X = data[indicator_columns]
    y = data['result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.write("Data preprocessing completed.")
    return X_train, X_test, y_train, y_test, indicator_columns, data

# Function to create neural network model
def create_nn_model(input_dim):
    tf.keras.backend.clear_session()  # Ensure graph is reset before creating a new model
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, input_dim=input_dim, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Function to create RNN model
def create_rnn_model(input_shape, rnn_type='LSTM'):
    tf.keras.backend.clear_session()  # Ensure graph is reset before creating a new model
    model = keras.Sequential()
    if rnn_type == 'LSTM':
        model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    else:
        model.add(keras.layers.GRU(64, input_shape=input_shape, return_sequences=True))
    model.add(keras.layers.LSTM(32))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Function to create CNN model
def create_cnn_model(input_shape):
    tf.keras.backend.clear_session()  # Ensure graph is reset before creating a new model
    model = keras.Sequential()
    model.add(keras.layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling1D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Function to calculate optimal win ranges
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

# Function to plot optimal win ranges using Plotly
def plot_optimal_win_ranges(data, optimal_ranges, target='result', trade_type='', model_name=''):
    plots = []
    descriptions = []
    for item in optimal_ranges:
        feature = item['feature']
        ranges = item['optimal_win_ranges']

        win_values = data[data[target] == 0][feature].dropna()
        loss_values = data[data[target] == 1][feature].dropna()

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=win_values, name='Win', marker_color='blue', opacity=0.75, nbinsx=50))
        fig.add_trace(go.Histogram(x=loss_values, name='Loss', marker_color='red', opacity=0.75, nbinsx=50))

        for range_start, range_end in ranges:
            fig.add_shape(type="rect", x0=range_start, x1=range_end, y0=0, y1=1,
                          fillcolor="blue", opacity=0.3, layer="below", line_width=0)

        fig.update_layout(
            title=f'Optimal Win Ranges for {feature} ({trade_type}, {model_name})',
            xaxis_title=feature,
            yaxis_title='Count',
            barmode='overlay'
        )
        st.plotly_chart(fig)
        plots.append(fig)
        descriptions.append(f"Optimal Win Ranges for {feature} ({trade_type}, {model_name})")
    return plots, descriptions

# Function to summarize optimal win ranges
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

# Function to reset session state
def reset_session_state():
    st.session_state.current_step = "load_data"
    st.session_state.model_type = "Random Forest"
    st.session_state.feature_importances = {}

# Function to run advanced model exploration
def run_advanced_model_exploration():
    st.title("Advanced Model Exploration")

    if "base_dir" not in st.session_state:
        st.session_state.base_dir = "./data/processed"

    base_dir = st.text_input("Base Directory", value=st.session_state.base_dir)

    if "feature_importances" not in st.session_state:
        st.session_state.feature_importances = {}

    if "current_step" not in st.session_state:
        st.session_state.current_step = "load_data"
    if "model_type" not in st.session_state:
        st.session_state.model_type = "Random Forest"

    selected_feature_types = st.multiselect(
        "Select Feature Types",
        ["Non-Market Value Data", "Percent Away Indicators", "Binary Indicators"],
        ["Non-Market Value Data", "Percent Away Indicators", "Binary Indicators"]
    )

    if st.session_state.current_step == "load_data" and st.button("Load Data"):
        st.write("Loading data...")
        try:
            data = load_data(base_dir)
            if data is not None:
                X_train, X_test, y_train, y_test, indicator_columns, data = preprocess_data(data, selected_feature_types)
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

        task_type = st.radio("Select Task Type", ["Classification", "Regression"], index=0)
        model_type = st.selectbox(
            "Select Model Type", 
            ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "Neural Network", "RNN (LSTM)", "RNN (GRU)", "CNN", "Stacking Ensemble"],
            key="model_type_select"
        )

        st.session_state.model_type_selected = model_type
        st.session_state.task_type_selected = task_type

        model_params = {}
        model = None

        if model_type == "Random Forest":
            model_params['n_estimators'] = st.slider("Number of Trees", min_value=10, max_value=500, value=100)
            model_params['max_depth'] = st.slider("Max Depth of Trees", min_value=1, max_value=20, value=10)
            if task_type == "Classification":
                model = RandomForestClassifier(n_estimators=model_params['n_estimators'], max_depth=model_params['max_depth'], random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=model_params['n_estimators'], max_depth=model_params['max_depth'], random_state=42)

        elif model_type == "Gradient Boosting":
            model_params['n_estimators'] = st.slider("Number of Trees", min_value=10, max_value=500, value=100)
            model_params['learning_rate'] = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1)
            model_params['max_depth'] = st.slider("Max Depth of Trees", min_value=1, max_value=20, value=3)
            if task_type == "Classification":
                model = GradientBoostingClassifier(n_estimators=model_params['n_estimators'], learning_rate=model_params['learning_rate'], max_depth=model_params['max_depth'], random_state=42)
            else:
                model = GradientBoostingRegressor(n_estimators=model_params['n_estimators'], learning_rate=model_params['learning_rate'], max_depth=model_params['max_depth'], random_state=42)

        elif model_type == "XGBoost":
            model_params['n_estimators'] = st.slider("Number of Trees", min_value=10, max_value=500, value=100)
            model_params['learning_rate'] = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1)
            model_params['max_depth'] = st.slider("Max Depth of Trees", min_value=1, max_value=20, value=3)
            if task_type == "Classification":
                model = xgb.XGBClassifier(n_estimators=model_params['n_estimators'], learning_rate=model_params['learning_rate'], max_depth=model_params['max_depth'], use_label_encoder=False, eval_metric='logloss', random_state=42)
            else:
                model = xgb.XGBRegressor(n_estimators=model_params['n_estimators'], learning_rate=model_params['learning_rate'], max_depth=model_params['max_depth'], random_state=42)

        elif model_type == "LightGBM":
            model_params['n_estimators'] = st.slider("Number of Trees", min_value=10, max_value=500, value=100)
            model_params['learning_rate'] = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1)
            model_params['max_depth'] = st.slider("Max Depth of Trees", min_value=1, max_value=20, value=3)
            if task_type == "Classification":
                model = lgb.LGBMClassifier(n_estimators=model_params['n_estimators'], learning_rate=model_params['learning_rate'], max_depth=model_params['max_depth'], random_state=42)
            else:
                model = lgb.LGBMRegressor(n_estimators=model_params['n_estimators'], learning_rate=model_params['learning_rate'], max_depth=model_params['max_depth'], random_state=42)

        elif model_type == "Neural Network":
            model_params['epochs'] = st.slider("Number of Epochs", min_value=10, max_value=1000, value=100)
            model_params['batch_size'] = st.slider("Batch Size", min_value=10, max_value=128, value=32)
            input_dim = st.session_state.X_train.shape[1]
            if task_type == "Classification":
                model = KerasClassifier(model=create_nn_model, model__input_dim=input_dim, epochs=model_params['epochs'], batch_size=model_params['batch_size'], verbose=0)
            else:
                model = KerasRegressor(model=create_nn_model, model__input_dim=input_dim, epochs=model_params['epochs'], batch_size=model_params['batch_size'], verbose=0)

        elif model_type == "RNN (LSTM)":
            model_params['epochs'] = st.slider("Number of Epochs", min_value=10, max_value=1000, value=100)
            model_params['batch_size'] = st.slider("Batch Size", min_value=10, max_value=128, value=32)
            input_shape = (st.session_state.X_train.shape[1], 1)
            if task_type == "Classification":
                model = KerasClassifier(model=create_rnn_model, model__input_shape=input_shape, model__rnn_type='LSTM', epochs=model_params['epochs'], batch_size=model_params['batch_size'], verbose=0)
            else:
                model = KerasRegressor(model=create_rnn_model, model__input_shape=input_shape, model__rnn_type='LSTM', epochs=model_params['epochs'], batch_size=model_params['batch_size'], verbose=0)

        elif model_type == "RNN (GRU)":
            model_params['epochs'] = st.slider("Number of Epochs", min_value=10, max_value=1000, value=100)
            model_params['batch_size'] = st.slider("Batch Size", min_value=10, max_value=128, value=32)
            input_shape = (st.session_state.X_train.shape[1], 1)
            if task_type == "Classification":
                model = KerasClassifier(model=create_rnn_model, model__input_shape=input_shape, model__rnn_type='GRU', epochs=model_params['epochs'], batch_size=model_params['batch_size'], verbose=0)
            else:
                model = KerasRegressor(model=create_rnn_model, model__input_shape=input_shape, model__rnn_type='GRU', epochs=model_params['epochs'], batch_size=model_params['batch_size'], verbose=0)

        elif model_type == "CNN":
            model_params['epochs'] = st.slider("Number of Epochs", min_value=10, max_value=1000, value=100)
            model_params['batch_size'] = st.slider("Batch Size", min_value=10, max_value=128, value=32)
            input_shape = (st.session_state.X_train.shape[1], 1)
            if task_type == "Classification":
                model = KerasClassifier(model=create_cnn_model, model__input_shape=input_shape, epochs=model_params['epochs'], batch_size=model_params['batch_size'], verbose=0)
            else:
                model = KerasRegressor(model=create_cnn_model, model__input_shape=input_shape, epochs=model_params['epochs'], batch_size=model_params['batch_size'], verbose=0)
        
        elif model_type == "Stacking Ensemble":
            if task_type == "Classification":
                base_learners = [
                    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
                    ('gb', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
                    ('xgb', xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, use_label_encoder=False, eval_metric='logloss', random_state=42))
                ]
                final_estimator = LogisticRegression(max_iter=1000)
                model = StackingClassifier(estimators=base_learners, final_estimator=final_estimator)
            else:
                base_learners = [
                    ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
                    ('gb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
                    ('xgb', xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
                ]
                final_estimator = LogisticRegression(max_iter=1000)
                model = StackingRegressor(estimators=base_learners, final_estimator=final_estimator)

        if st.button("Train Model"):
            st.write(f"Training {model_type} for {task_type} task...")
            with st.spinner("Training in progress..."):
                try:
                    if model_type in ["Neural Network", "RNN (LSTM)", "RNN (GRU)", "CNN"]:
                        model.fit(st.session_state.X_train, st.session_state.y_train, callbacks=[TqdmCallback(verbose=1)])
                    else:
                        model.fit(st.session_state.X_train, st.session_state.y_train)
                    y_pred = model.predict(st.session_state.X_test)

                    plots = []
                    descriptions = []
                    dataframes = []
                    dataframe_descriptions = []
                    text_sections = []

                    if task_type == "Classification":
                        accuracy = accuracy_score(st.session_state.y_test, y_pred)
                        st.write(f"Accuracy: {accuracy}")
                        st.write("Classification Report:")
                        class_report = classification_report(st.session_state.y_test, y_pred)
                        st.text(class_report)
                        st.write("Confusion Matrix:")
                        cm = confusion_matrix(st.session_state.y_test, y_pred)
                        fig_cm = px.imshow(cm, labels=dict(x="Predicted", y="True", color="Count"), x=["Negative", "Positive"], y=["Negative", "Positive"], color_continuous_scale='Blues')
                        st.plotly_chart(fig_cm)
                        text_sections.append(("Classification Report", class_report))
                        text_sections.append(("Accuracy", f"Accuracy: {accuracy}"))
                        plots.append(fig_cm)
                        descriptions.append("Confusion Matrix")
                    else:
                        mse = mean_squared_error(st.session_state.y_test, y_pred)
                        st.write(f"Mean Squared Error: {mse}")
                        text_sections.append(("Mean Squared Error", f"Mean Squared Error: {mse}"))

                    # Save the model
                    model_save_path = os.path.join(base_dir, f"models/{model_type}_{task_type}_model.pkl")
                    joblib.dump(model, model_save_path)
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
                        top_15_importance_df = importance_df.head(15)
                        fig_feat_imp = px.bar(top_15_importance_df, x='Importance', y='Feature', orientation='h')
                        st.plotly_chart(fig_feat_imp)
                        st.session_state.feature_importances[model_type] = feature_importances
                        plots.append(fig_feat_imp)
                        descriptions.append("Feature Importance")
                    elif model_type in ["Neural Network", "RNN (LSTM)", "RNN (GRU)", "CNN"]:
                        st.write("Calculating feature importances using SHAP...")
                        underlying_model = model.model_
                        explainer = shap.Explainer(underlying_model, st.session_state.X_train)
                        shap_values = explainer(st.session_state.X_test)
                        feature_importances = np.abs(shap_values.values).mean(axis=0)
                        importance_df['Importance'] = feature_importances
                        importance_df.sort_values(by='Importance', ascending=False, inplace=True)
                        top_15_importance_df = importance_df.head(15)
                        fig_feat_imp = px.bar(top_15_importance_df, x='Importance', y='Feature', orientation='h')
                        st.plotly_chart(fig_feat_imp)
                        st.session_state.feature_importances[model_type] = feature_importances
                        plots.append(fig_feat_imp)
                        descriptions.append("Feature Importance")

                    st.session_state.current_step = "eda"
                    st.session_state.model_type_selected = model_type

                    st.subheader("Optimal Win Ranges")
                    top_n = st.selectbox("Select Top N Indicators", [3, 5, 10, len(st.session_state.indicator_columns)], index=2)
                    if not importance_df.empty:
                        selected_features = importance_df.head(top_n)['Feature']
                    else:
                        selected_features = st.session_state.indicator_columns[:top_n]

                    optimal_ranges = calculate_optimal_win_ranges(st.session_state.data, features=selected_features)
                    st.session_state.optimal_ranges = optimal_ranges
                    opt_plots, opt_descriptions = plot_optimal_win_ranges(st.session_state.data, optimal_ranges, trade_type='', model_name=model_type)
                    plots.extend(opt_plots)
                    descriptions.extend(opt_descriptions)

                    optimal_win_ranges_summary = summarize_optimal_win_ranges(optimal_ranges)
                    st.write(optimal_win_ranges_summary)
                    output_path = os.path.join(base_dir, f'docs/ml_analysis/win_ranges_summary/optimal_win_ranges_summary_{model_type}.csv')
                    optimal_win_ranges_summary.to_csv(output_path, index=False)
                    st.write(f"Saved optimal win ranges summary to {output_path}")
                    dataframes.append(optimal_win_ranges_summary)
                    dataframe_descriptions.append("Optimal Win Ranges Summary")

                    model_type = st.session_state.model_type_selected
                    optimal_ranges = st.session_state.optimal_ranges

                    if model_type in st.session_state.feature_importances:
                        st.write("Feature Importance Heatmap:")
                        feature_importance_values = st.session_state.feature_importances[model_type]
                        fig = px.imshow([feature_importance_values], labels=dict(x="Features", color="Importance"), x=st.session_state.indicator_columns, color_continuous_scale='Blues')
                        st.plotly_chart(fig)
                        plots.append(fig)
                        descriptions.append("Feature Importance Heatmap")

                    st.write("Correlation Matrix of Top Indicators:")
                    selected_model = st.selectbox("Select Model for Correlation", list(st.session_state.feature_importances.keys()), key='correlation_model')
                    if selected_model in st.session_state.feature_importances:
                        top_features = st.session_state.feature_importances[selected_model][:10]
                        top_features_list = [st.session_state.indicator_columns[i] for i in np.argsort(top_features)[::-1][:10]]
                        corr = st.session_state.data[top_features_list].corr()
                        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='Blues')
                        st.plotly_chart(fig)
                        plots.append(fig)
                        descriptions.append("Correlation Matrix of Top Indicators")

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
                        plots.append(fig)
                        descriptions.append(f"Distribution of {selected_indicator} for Winning and Losing Trades")

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
                        plots.append(fig)
                        descriptions.append(f'KDE Plot with Optimal Win Ranges for {selected_indicator}')

                        st.write("Loss Mitigation Analysis")
                        loss_conditions = st.session_state.data[st.session_state.data['result'] == 1][st.session_state.indicator_columns].describe().transpose()
                        st.write(loss_conditions)
                        fig_loss_cond = px.bar(loss_conditions, x=loss_conditions.index, y="mean", labels={'x': 'Indicators', 'y': 'Mean Value'}, title='Mean Indicator Values for Losses')
                        st.plotly_chart(fig_loss_cond)
                        plots.append(fig_loss_cond)
                        descriptions.append("Mean Indicator Values for Losses")
                        dataframes.append(loss_conditions)
                        dataframe_descriptions.append("Loss Mitigation Analysis")

                    pdf_filename = os.path.join(base_dir, f'docs/ml_analysis/{model_type}_{task_type}_complete_analysis.pdf')
                    save_all_to_pdf(pdf_filename, text_sections, dataframes, dataframe_descriptions, plots, descriptions)
                    st.write(f"Saved complete analysis to {pdf_filename}")

                except Exception as e:
                    st.error(f"Error during model training: {e}")

    if st.session_state.current_step == "eda":
        if st.button("Restart Exploration"):
            reset_session_state()

if __name__ == "__main__":
    run_advanced_model_exploration()
