# advanced_model_exploration.py
import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
import xgboost as xgb
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
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

    data['result'] = data['result'].apply(lambda x: 1 if x == 'win' else 0)

    X = data[indicator_columns]
    y = data['result']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.write("Data preprocessing completed.")
    return X_train_scaled, X_test_scaled, y_train, y_test, indicator_columns

# Define Keras model
def create_nn_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def run_advanced_model_exploration():
    st.title("Advanced Model Exploration")

    if "base_dir" not in st.session_state:
        st.session_state.base_dir = "."

    base_dir = st.text_input("Base Directory", value=st.session_state.base_dir)

    if st.button("Load Data"):
        st.write("Loading data...")
        try:
            data = load_data(base_dir)
            X_train, X_test, y_train, y_test, indicator_columns = preprocess_data(data)
            st.session_state.data = data
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.indicator_columns = indicator_columns
            st.success("Data loaded and preprocessed successfully.")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return

    if "data" in st.session_state:
        st.write("Select model and hyperparameters for exploration")

        model_type = st.selectbox("Select Model Type", ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "Neural Network", "Stacking Ensemble"])

        if model_type == "Random Forest":
            st.subheader("Random Forest Parameters")
            n_estimators = st.slider("Number of Trees", min_value=10, max_value=500, value=100)
            max_depth = st.slider("Max Depth of Trees", min_value=1, max_value=20, value=10)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        
        elif model_type == "Gradient Boosting":
            st.subheader("Gradient Boosting Parameters")
            n_estimators = st.slider("Number of Trees", min_value=10, max_value=500, value=100)
            learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1)
            max_depth = st.slider("Max Depth of Trees", min_value=1, max_value=20, value=3)
            model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)

        elif model_type == "XGBoost":
            st.subheader("XGBoost Parameters")
            n_estimators = st.slider("Number of Trees", min_value=10, max_value=500, value=100)
            learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1)
            max_depth = st.slider("Max Depth of Trees", min_value=1, max_value=20, value=3)
            model = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, use_label_encoder=False, eval_metric='logloss', random_state=42)

        elif model_type == "LightGBM":
            st.subheader("LightGBM Parameters")
            n_estimators = st.slider("Number of Trees", min_value=10, max_value=500, value=100)
            learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1)
            max_depth = st.slider("Max Depth of Trees", min_value=1, max_value=20, value=3)
            model = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)

        elif model_type == "Neural Network":
            st.subheader("Neural Network Parameters")
            epochs = st.slider("Number of Epochs", min_value=10, max_value=1000, value=100)
            batch_size = st.slider("Batch Size", min_value=10, max_value=128, value=32)
            model = KerasClassifier(build_fn=create_nn_model, input_dim=X_train.shape[1], epochs=epochs, batch_size=batch_size, verbose=0)
        
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
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"Accuracy: {accuracy}")
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            st.write("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

if __name__ == "__main__":
    run_advanced_model_exploration()
