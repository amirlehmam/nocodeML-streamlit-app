import os
import warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import tensorflow as tf
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
from scikeras.wrappers import KerasClassifier, KerasRegressor
import shap
from PIL import Image
from io import BytesIO
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib
from tqdm import tqdm
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import logging
from sqlalchemy import create_engine

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings and messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Enable eager execution for TensorFlow 2.x
tf.config.run_functions_eagerly(True)

# Constants
BASE_DIR = "./data/processed/"
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models")
PDF_SAVE_PATH = os.path.join(BASE_DIR, "docs/ml_analysis")

# Database connection details
DB_CONFIG = {
    'dbname': 'defaultdb',
    'user': 'doadmin',
    'password': 'AVNS_hnzmIdBmiO7aj5nylWW',
    'host': 'nocodemldb-do-user-16993120-0.c.db.ondigitalocean.com',
    'port': 25060,
    'sslmode': 'require'
}

def get_db_connection():
    connection_str = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
    engine = create_engine(connection_str)
    return engine

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

# Load data with caching
@st.cache_data
def load_data_from_db():
    st.write("Loading data from the database...")
    try:
        engine = get_db_connection()
        query = "SELECT * FROM merged_trade_indicator_event"
        data = pd.read_sql_query(query, engine)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Preprocess data
@st.cache_data
def preprocess_data(data, selected_feature_types):
    st.write("Preprocessing data...")

    if data is None:
        raise ValueError("The provided data is None.")
    
    if data.shape[1] < 8:
        raise ValueError("Data does not have enough columns for indicators. Ensure indicators start after the 7th column.")
    
    all_indicators = data.columns[7:]
    st.write(f"All indicators: {all_indicators}")

    # Exclude 'strategy_amount' and 'account_amount' from the indicators
    excluded_columns = ['strategy_amount', 'account_amount']
    all_indicators = [col for col in all_indicators if col not in excluded_columns]

    # Filter features based on user selection
    indicator_columns = []
    if "Non-Market Value Data" in selected_feature_types:
        indicator_columns.extend([col for col in all_indicators if "_percent_away" not in col and "_binary" not in col])
    if "Percent Away Indicators" in selected_feature_types:
        indicator_columns.extend([col for col in all_indicators if "_percent_away" in col])
    if "Binary Indicators" in selected_feature_types:
        indicator_columns.extend([col for col in all_indicators if "_binary" in col])

    st.write(f"Selected indicators: {indicator_columns}")

    # Ensure 'result' column is correctly processed
    data['result'] = data['result'].apply(lambda x: 1 if x == 'win' else 0)

    # Fill NaN values in indicator columns
    data[indicator_columns] = data[indicator_columns].fillna(0)

    # Ensure binary columns contain only 0 and 1
    for col in indicator_columns:
        if '_binary' in col:
            data[col] = data[col].apply(lambda x: 1 if x > 0 else 0)

    # Split columns into binary and non-binary
    binary_columns = [col for col in indicator_columns if '_binary' in col]
    non_binary_columns = [col for col in indicator_columns if col not in binary_columns]

    if non_binary_columns:
        # Normalize only non-binary columns
        scaler = StandardScaler()
        data[non_binary_columns] = scaler.fit_transform(data[non_binary_columns])

    # Feature Engineering: Add derived metrics (changes, slopes)
    changes = data[indicator_columns].diff().add_suffix('_change')
    slopes = data[indicator_columns].diff().diff().add_suffix('_slope')
    
    # Ensure changes and slopes do not contain None values
    changes = changes.fillna(0)
    slopes = slopes.fillna(0)
    
    data = pd.concat([data, changes, slopes], axis=1)

    # Fill NaN values with column mean
    data[indicator_columns] = data[indicator_columns].apply(lambda col: col.fillna(col.mean()))

    X = data[indicator_columns]
    y = data['result'].astype(int)  # Ensure target is of integer type

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, indicator_columns, data


# Create neural network model
def create_nn_model(input_dim):
    tf.keras.backend.clear_session()  # Ensure graph is reset before creating a new model
    model = Sequential()
    model.add(layers.Dense(64, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# Create RNN model
def create_rnn_model(input_shape, rnn_type='LSTM'):
    tf.keras.backend.clear_session()  # Ensure graph is reset before creating a new model
    model = Sequential()
    if rnn_type == 'LSTM':
        model.add(layers.LSTM(64, input_shape=input_shape, return_sequences=True))
    else:
        model.add(layers.GRU(64, input_shape=input_shape, return_sequences=True))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# Create CNN model
def create_cnn_model(input_shape):
    tf.keras.backend.clear_session()  # Ensure graph is reset before creating a new model
    model = Sequential()
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# Calculate optimal win ranges
def calculate_optimal_win_ranges(data, target='result', features=None, trade_type=None):
    optimal_ranges = []

    if features is None:
        features = data.columns.drop([target])

    for feature in tqdm(features, desc=f"Calculating Optimal Win Ranges ({trade_type})"):
        data[feature] = pd.to_numeric(data[feature], errors='coerce')
        
        if trade_type:
            trade_data = data[data['event'].str.startswith(trade_type)]
        else:
            trade_data = data

        win_values = trade_data[trade_data[target] == 1][feature].dropna().values.astype(float)
        loss_values = trade_data[trade_data[target] == 0][feature].dropna().values.astype(float)

        if len(win_values) == 0 or len(loss_values) == 0:
            continue

        # Check if the data is binary
        if np.unique(win_values).size == 2 and np.array_equal(np.unique(win_values), [0, 1]) and \
           np.unique(loss_values).size == 2 and np.array_equal(np.unique(loss_values), [0, 1]):
            optimal_ranges.append({
                'feature': feature,
                'optimal_win_ranges': [(0, 1)]
            })
            continue

        try:
            win_kde = gaussian_kde(win_values)
            loss_kde = gaussian_kde(loss_values)

            x_grid = np.linspace(min(trade_data[feature].dropna()), max(trade_data[feature].dropna()), 1000)
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
        except np.linalg.LinAlgError:
            st.warning(f"Skipped KDE for {feature} due to singular covariance matrix.")

    return optimal_ranges
    
# Plot optimal win ranges using Plotly
def plot_optimal_win_ranges(data, optimal_ranges, target='result', trade_type='', model_name=''):
    plots = []
    descriptions = []
    for item in optimal_ranges:
        feature = item['feature']
        ranges = item['optimal_win_ranges']

        if data[feature].nunique() == 2:  # Check if the feature is binary
            win_values = data[data[target] == 0][feature].dropna()
            loss_values = data[data[target] == 1][feature].dropna()

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Under', 'Above'],
                y=[(win_values == 0).sum(), (win_values == 1).sum()],
                name='Win',
                marker_color='blue',
                opacity=0.75
            ))
            fig.add_trace(go.Bar(
                x=['Under', 'Above'],
                y=[(loss_values == 0).sum(), (loss_values == 1).sum()],
                name='Loss',
                marker_color='red',
                opacity=0.75
            ))

            fig.update_layout(
                title=f'Binary Indicator Distribution for {feature} ({trade_type}, {model_name})',
                xaxis_title=feature,
                yaxis_title='Count',
                barmode='overlay'
            )
        else:
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
        
        plots.append(fig)
        descriptions.append(f"Optimal Win Ranges for {feature} ({trade_type}, {model_name})")
    
    return plots, descriptions

# Summarize optimal win ranges
def summarize_optimal_win_ranges(optimal_ranges, trade_type):
    summary = []
    for item in optimal_ranges:
        feature = item['feature']
        for range_start, range_end in item['optimal_win_ranges']:
            summary.append({
                'feature': feature,
                'trade_type': trade_type,
                'optimal_win_range_start': range_start,
                'optimal_win_range_end': range_end
            })
    return pd.DataFrame(summary)

# Reset session state
def reset_session_state():
    # Clear specific session state variables instead of all
    st.session_state.current_step = "load_data"
    st.session_state.model_type = "Random Forest"
    st.session_state.feature_importances = {}
    st.session_state.data = None
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None
    st.session_state.indicator_columns = None
    st.session_state.optimal_ranges_long = None
    st.session_state.optimal_ranges_short = None
    st.session_state.optimal_ranges_both = None
    st.rerun()

# Function to create CatBoost model
def create_catboost_model(task_type):
    if task_type == "Classification":
        model = cb.CatBoostClassifier(silent=True)
    else:
        model = cb.CatBoostRegressor(silent=True)
    return model

# Hyperparameter tuning function using GridSearchCV
def perform_hyperparameter_tuning(model, param_grid, X_train, y_train):
    search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_

# Function to perform model training with optional hyperparameter tuning
def train_model(model, X_train, y_train, X_test, y_test, task_type, perform_tuning=False, param_grid=None):
    # Make sure the arrays are writable
    X_train = np.array(X_train, copy=True, subok=True)
    y_train = np.array(y_train, copy=True, subok=True)
    X_test = np.array(X_test, copy=True, subok=True)
    y_test = np.array(y_test, copy=True, subok=True)
    
    X_train.setflags(write=1)
    y_train.setflags(write=1)
    X_test.setflags(write=1)
    y_test.setflags(write=1)

    if perform_tuning and param_grid:
        model, best_params = perform_hyperparameter_tuning(model, param_grid, X_train, y_train)
        st.write(f"Best Hyperparameters: {best_params}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if task_type == "Classification":
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, labels=dict(x="Predicted", y="True", color="Count"), x=["Negative", "Positive"], y=["Negative", "Positive"], color_continuous_scale='Blues')
        st.plotly_chart(fig_cm)
        return model, y_pred, accuracy, fig_cm
    else:
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse}")
        return model, y_pred, mse

# Function to display SHAP explanations
def display_shap_explanations(model, X_train, X_test):
    st.subheader("Model Explainability with SHAP")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    
    st.write("SHAP Summary Plot")
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(bbox_inches='tight')
    
    st.write("SHAP Force Plot")
    shap.initjs()
    force_plot_html = shap.force_plot(explainer.expected_value, shap_values.values, X_test)
    st.components.v1.html(force_plot_html, height=500)

# Function to create and plot KDE for different trade types
def plot_kde_distribution(data, trade_type, optimal_ranges):
    plots = []
    descriptions = []
    feature_info = []

    for item in optimal_ranges:
        feature = item['feature']
        
        win_values = data[(data['result'] == 1) & (data['event'].str.startswith(trade_type))][feature].dropna()
        loss_values = data[(data['result'] == 0) & (data['event'].str.startswith(trade_type))][feature].dropna()

        win_count = len(win_values)
        loss_count = len(loss_values)

        feature_info.append(f"Plotting feature: {feature}, trade type: {trade_type}\n"
                            f"Win values count: {win_count}, Loss values count: {loss_count}")

        if win_count == 0 or loss_count == 0:
            continue

        if data[feature].nunique() == 2:  # Check if the feature is binary
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Under', 'Above'],
                y=[(win_values == 0).sum(), (win_values == 1).sum()],
                name='Win',
                marker_color='blue',
                opacity=0.75
            ))
            fig.add_trace(go.Bar(
                x=['Under', 'Above'],
                y=[(loss_values == 0).sum(), (loss_values == 1).sum()],
                name='Loss',
                marker_color='red',
                opacity=0.75
            ))

            fig.update_layout(
                title=f'Binary Indicator Distribution for {feature} ({trade_type})',
                xaxis_title=feature,
                yaxis_title='Count',
                barmode='group'  # Change here to display side-by-side bars
            )
        else:
            kde_win = gaussian_kde(win_values)
            kde_loss = gaussian_kde(loss_values)
            x_grid = np.linspace(min(data[feature].dropna()), max(data[feature].dropna()), 1000)
            kde_win_density = kde_win(x_grid)
            kde_loss_density = kde_loss(x_grid)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_grid, y=kde_win_density, mode='lines', name='Win', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=x_grid, y=kde_loss_density, mode='lines', name='Loss', line=dict(color='red')))
            for range_item in optimal_ranges:
                if range_item['feature'] == feature:
                    for start, end in range_item['optimal_win_ranges']:
                        fig.add_vrect(x0=start, x1=end, fillcolor="blue", opacity=0.3, line_width=0)

            fig.update_layout(
                title=f'KDE Plot with Optimal Win Ranges for {feature} ({trade_type})',
                xaxis_title=feature,
                yaxis_title='Density',
                width=800,
                height=400
            )

        plots.append(fig)
        descriptions.append(f'Optimal Win Ranges for {feature} ({trade_type})')

    return plots, descriptions, feature_info

#def display_feature_info(feature_info):
#    st.subheader("Feature Information")
#    for info in feature_info:
#        st.text(info)

# Run advanced model exploration
def run_advanced_model_exploration():
    st.title("Advanced Model Exploration")

    st.session_state.base_dir = BASE_DIR

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
            data = load_data_from_db()
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
            ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost", "Neural Network", "RNN (LSTM)", "RNN (GRU)", "CNN", "Stacking Ensemble"],
            key="model_type_select"
        )

        st.session_state.model_type_selected = model_type
        st.session_state.task_type_selected = task_type

        model_params = {}
        model = None
        param_grid = None

        if model_type == "Random Forest":
            model_params['n_estimators'] = st.slider("Number of Trees", min_value=10, max_value=500, value=100)
            model_params['max_depth'] = st.slider("Max Depth of Trees", min_value=1, max_value=20, value=10)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30]
            }
            if task_type == "Classification":
                model = RandomForestClassifier(n_estimators=model_params['n_estimators'], max_depth=model_params['max_depth'], random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=model_params['n_estimators'], max_depth=model_params['max_depth'], random_state=42)

        elif model_type == "Gradient Boosting":
            model_params['n_estimators'] = st.slider("Number of Trees", min_value=10, max_value=500, value=100)
            model_params['learning_rate'] = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1)
            model_params['max_depth'] = st.slider("Max Depth of Trees", min_value=1, max_value=20, value=3)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            if task_type == "Classification":
                model = GradientBoostingClassifier(n_estimators=model_params['n_estimators'], learning_rate=model_params['learning_rate'], max_depth=model_params['max_depth'], random_state=42)
            else:
                model = GradientBoostingRegressor(n_estimators=model_params['n_estimators'], learning_rate=model_params['learning_rate'], max_depth=model_params['max_depth'], random_state=42)

        elif model_type == "XGBoost":
            model_params['n_estimators'] = st.slider("Number of Trees", min_value=10, max_value=500, value=100)
            model_params['learning_rate'] = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1)
            model_params['max_depth'] = st.slider("Max Depth of Trees", min_value=1, max_value=20, value=3)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            if task_type == "Classification":
                model = xgb.XGBClassifier(n_estimators=model_params['n_estimators'], learning_rate=model_params['learning_rate'], max_depth=model_params['max_depth'], use_label_encoder=False, eval_metric='logloss', random_state=42)
            else:
                model = xgb.XGBRegressor(n_estimators=model_params['n_estimators'], learning_rate=model_params['learning_rate'], max_depth=model_params['max_depth'], random_state=42)

        elif model_type == "LightGBM":
            model_params['n_estimators'] = st.slider("Number of Trees", min_value=10, max_value=500, value=100)
            model_params['learning_rate'] = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1)
            model_params['max_depth'] = st.slider("Max Depth of Trees", min_value=1, max_value=20, value=3)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
            if task_type == "Classification":
                model = lgb.LGBMClassifier(n_estimators=model_params['n_estimators'], learning_rate=model_params['learning_rate'], max_depth=model_params['max_depth'], random_state=42)
            else:
                model = lgb.LGBMRegressor(n_estimators=model_params['n_estimators'], learning_rate=model_params['learning_rate'], max_depth=model_params['max_depth'], random_state=42)

        elif model_type == "CatBoost":
            model = create_catboost_model(task_type)
            param_grid = {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1, 0.2]
            }

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

        perform_tuning = st.checkbox("Perform Hyperparameter Tuning", value=False)
        
        if st.button("Train Model"):
            st.write(f"Training {model_type} for {task_type} task...")
            with st.spinner("Training in progress..."):
                try:
                    if task_type == "Classification":
                        model, y_pred, accuracy, fig_cm = train_model(model, st.session_state.X_train, st.session_state.y_train, st.session_state.X_test, st.session_state.y_test, task_type, perform_tuning, param_grid)
                    else:
                        model, y_pred, mse = train_model(model, st.session_state.X_train, st.session_state.y_train, st.session_state.X_test, st.session_state.y_test, task_type, perform_tuning, param_grid)

                    plots = []
                    descriptions = []
                    dataframes = []
                    dataframe_descriptions = []
                    text_sections = []

                    if task_type == "Classification":
                        st.write(f"Accuracy: {accuracy}")
                        st.write("Classification Report:")
                        class_report = classification_report(st.session_state.y_test, y_pred)
                        st.text(class_report)
                        st.write("Confusion Matrix:")
                        st.plotly_chart(fig_cm)
                        text_sections.append(("Classification Report", class_report))
                        text_sections.append(("Accuracy", f"Accuracy: {accuracy}"))
                        plots.append(fig_cm)
                        descriptions.append("Confusion Matrix")
                    else:
                        st.write(f"Mean Squared Error: {mse}")
                        text_sections.append(("Mean Squared Error", f"Mean Squared Error: {mse}"))

                    # Save the model
                    model_save_path = os.path.join(MODEL_SAVE_PATH, f"{model_type}_{task_type}_model.pkl")
                    joblib.dump(model, model_save_path)
                    st.write(f"Model saved to {model_save_path}")

                    # Feature importance
                    importance_df = pd.DataFrame({
                        'Feature': st.session_state.indicator_columns,
                        'Importance': [0] * len(st.session_state.indicator_columns)
                    })
                    if model_type not in ["Neural Network", "RNN (LSTM)", "RNN (GRU)", "CNN", "CatBoost"] and hasattr(model, 'feature_importances_'):
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

                    # Display SHAP explanations for interpretability
                    if model_type in ["Neural Network", "RNN (LSTM)", "RNN (GRU)", "CNN"]:
                        display_shap_explanations(underlying_model, st.session_state.X_train, st.session_state.X_test)

                    st.subheader("Optimal Win Ranges")
                    top_n = st.selectbox("Select Top N Indicators", [3, 5, 10, len(st.session_state.indicator_columns)], index=2)
                    if not importance_df.empty:
                        selected_features = importance_df.head(top_n)['Feature']
                    else:
                        selected_features = st.session_state.indicator_columns[:top_n]

                    # Calculate optimal win ranges for long, short, and both trades
                    optimal_ranges_long = calculate_optimal_win_ranges(st.session_state.data, features=selected_features, trade_type='LE')
                    optimal_ranges_short = calculate_optimal_win_ranges(st.session_state.data, features=selected_features, trade_type='SE')
                    optimal_ranges_both = calculate_optimal_win_ranges(st.session_state.data, features=selected_features)

                    st.session_state.optimal_ranges_long = optimal_ranges_long
                    st.session_state.optimal_ranges_short = optimal_ranges_short
                    st.session_state.optimal_ranges_both = optimal_ranges_both

                    # Plot KDE distributions for Long, Short, and Both trades
                    st.subheader("KDE Distribution for Long Trades")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        long_plots, long_descriptions, long_feature_info = plot_kde_distribution(st.session_state.data, 'LE', optimal_ranges_long)
                        for fig in long_plots:
                            st.plotly_chart(fig)
                        #display_feature_info(long_feature_info)

                    st.subheader("KDE Distribution for Short Trades")
                    with col2:
                        short_plots, short_descriptions, short_feature_info = plot_kde_distribution(st.session_state.data, 'SE', optimal_ranges_short)
                        for fig in short_plots:
                            st.plotly_chart(fig)
                        #display_feature_info(short_feature_info)

                    st.subheader("KDE Distribution for Both Long and Short Trades")
                    with col3:
                        both_plots, both_descriptions, both_feature_info = plot_kde_distribution(st.session_state.data, '', optimal_ranges_both)
                        for fig in both_plots:
                            st.plotly_chart(fig)
                        #display_feature_info(both_feature_info)

                    # Summarize optimal win ranges for each trade type
                    optimal_win_ranges_summary_long = summarize_optimal_win_ranges(optimal_ranges_long, 'LE')
                    optimal_win_ranges_summary_short = summarize_optimal_win_ranges(optimal_ranges_short, 'SE')
                    optimal_win_ranges_summary_both = summarize_optimal_win_ranges(optimal_ranges_both, '')

                    st.write("Optimal Win Ranges Summary for Long Trades")
                    st.dataframe(optimal_win_ranges_summary_long)
                    st.write("Optimal Win Ranges Summary for Short Trades")
                    st.dataframe(optimal_win_ranges_summary_short)
                    st.write("Optimal Win Ranges Summary for Both Long and Short Trades")
                    st.dataframe(optimal_win_ranges_summary_both)

                    # Save optimal win ranges summary to CSV
                    output_path_long = os.path.join(PDF_SAVE_PATH, f'win_ranges_summary/optimal_win_ranges_summary_long_{model_type}.csv')
                    optimal_win_ranges_summary_long.to_csv(output_path_long, index=False)
                    output_path_short = os.path.join(PDF_SAVE_PATH, f'win_ranges_summary/optimal_win_ranges_summary_short_{model_type}.csv')
                    optimal_win_ranges_summary_short.to_csv(output_path_short, index=False)
                    output_path_both = os.path.join(PDF_SAVE_PATH, f'win_ranges_summary/optimal_win_ranges_summary_both_{model_type}.csv')
                    optimal_win_ranges_summary_both.to_csv(output_path_both, index=False)

                    st.write(f"Saved optimal win ranges summary to {output_path_long}, {output_path_short}, and {output_path_both}")

                    dataframes.append(optimal_win_ranges_summary_long)
                    dataframe_descriptions.append("Optimal Win Ranges Summary for Long Trades")
                    dataframes.append(optimal_win_ranges_summary_short)
                    dataframe_descriptions.append("Optimal Win Ranges Summary for Short Trades")
                    dataframes.append(optimal_win_ranges_summary_both)
                    dataframe_descriptions.append("Optimal Win Ranges Summary for Both Long and Short Trades")

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
                        win_data = st.session_state.data[st.session_state.data['result'] == 1][selected_indicator].dropna()
                        loss_data = st.session_state.data[st.session_state.data['result'] == 0][selected_indicator].dropna()

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
                        for range_item in optimal_ranges_both:
                            if range_item['feature'] == selected_indicator:
                                for start, end in range_item['optimal_win_ranges']:
                                    fig.add_vrect(x0=start, x1=end, fillcolor="blue", opacity=0.3, line_width=0)

                        fig.update_layout(title_text=f'KDE Plot with Optimal Win Ranges for {selected_indicator}', xaxis_title=selected_indicator, yaxis_title='Density', width=800, height=400)
                        st.plotly_chart(fig)
                        plots.append(fig)
                        descriptions.append(f'KDE Plot with Optimal Win Ranges for {selected_indicator}')

                        st.write("Loss Mitigation Analysis")
                        loss_conditions = st.session_state.data[st.session_state.data['result'] == 0][st.session_state.indicator_columns].describe().transpose()
                        st.write(loss_conditions)
                        fig_loss_cond = px.bar(loss_conditions, x=loss_conditions.index, y="mean", labels={'x': 'Indicators', 'y': 'Mean Value'}, title='Mean Indicator Values for Losses')
                        st.plotly_chart(fig_loss_cond)
                        plots.append(fig_loss_cond)
                        descriptions.append("Mean Indicator Values for Losses")
                        dataframes.append(loss_conditions)
                        dataframe_descriptions.append("Loss Mitigation Analysis")

                    pdf_filename = os.path.join(PDF_SAVE_PATH, f'{model_type}_{task_type}_complete_analysis.pdf')
                    save_all_to_pdf(pdf_filename, text_sections, dataframes, dataframe_descriptions, plots, descriptions)

                    # Add download button for PDF
                    with open(pdf_filename, "rb") as file:
                        st.download_button(
                            label="Download PDF",
                            data=file,
                            file_name=f'{model_type}_{task_type}_complete_analysis.pdf',
                            mime='application/octet-stream',
                        )

                except Exception as e:
                    st.error(f"Error during model training: {e}")

    if st.session_state.current_step == "eda":
        if st.button("Restart Exploration"):
            reset_session_state()

if __name__ == "__main__":
    run_advanced_model_exploration()
