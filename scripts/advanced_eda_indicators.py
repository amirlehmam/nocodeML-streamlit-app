import pandas as pd
import numpy as np
import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from scipy import stats

def load_data(data_dir):
    indicator_data = pd.read_csv(os.path.join(data_dir, "indicator_data.csv"))
    trade_data = pd.read_csv(os.path.join(data_dir, "trade_data.csv"))
    event_data = pd.read_csv(os.path.join(data_dir, "event_data.csv"))

    indicator_data['time'] = pd.to_datetime(indicator_data['time'])
    trade_data['time'] = pd.to_datetime(trade_data['time'])
    event_data['time'] = pd.to_datetime(event_data['time'])

    return {'indicator_data': indicator_data, 'trade_data': trade_data, 'event_data': event_data}

def preprocess_events(event_data):
    relevant_events = ['Profit', 'Loss']
    event_data = event_data[event_data['event'].str.contains('|'.join(relevant_events))]
    return event_data

def identify_trade_results(trade_data, event_data):
    relevant_trades = ['SE1', 'SE2', 'SE3', 'SE4', 'SE5', 'LE1', 'LE2', 'LE3', 'LE4', 'LE5', 'Parabolic stop', 'SX', 'LX']
    trade_data = trade_data[trade_data['event'].isin(relevant_trades)]
    
    trade_event_data = pd.merge_asof(trade_data.sort_values('time'), event_data.sort_values('time'), on='time', direction='backward', suffixes=('', '_event'))

    def classify_trade(row):
        if pd.notna(row['event_event']):
            if 'Profit' in (row['event_event']):
                return 'win'
            elif 'Loss' in (row['event_event']):
                return 'loss'
        return 'unknown'
    
    trade_event_data['result'] = trade_event_data.apply(classify_trade, axis=1)
    return trade_event_data

def preprocess_indicator_data(indicator_data):
    indicator_data_agg = indicator_data.groupby(['time', 'indicator_name']).mean().reset_index()
    return indicator_data_agg

def merge_with_indicators(trade_event_data, indicator_data):
    indicator_data = preprocess_indicator_data(indicator_data)
    
    market_value_indicators = indicator_data[indicator_data['indicator_value'] > 10000]
    other_indicators = indicator_data[indicator_data['indicator_value'] <= 10000]
    
    indicator_pivot = market_value_indicators.pivot(index='time', columns='indicator_name', values='indicator_value').reset_index()
    binary_indicator_pivot = market_value_indicators.pivot(index='time', columns='indicator_name', values='binary_indicator').reset_index()
    percent_away_pivot = market_value_indicators.pivot(index='time', columns='indicator_name', values='percent_away').reset_index()
    
    merged_data = pd.merge_asof(trade_event_data.sort_values('time'), indicator_pivot.sort_values('time'), on='time', direction='nearest')
    merged_data = pd.merge_asof(merged_data, binary_indicator_pivot.sort_values('time'), on='time', direction='nearest', suffixes=('', '_binary'))
    merged_data = pd.merge_asof(merged_data, percent_away_pivot.sort_values('time'), on='time', direction='nearest', suffixes=('', '_percent_away'))

    market_value_columns = market_value_indicators['indicator_name'].unique()
    merged_data.drop(columns=market_value_columns, inplace=True)

    other_indicator_pivot = other_indicators.pivot(index='time', columns='indicator_name', values='indicator_value').reset_index()
    merged_data = pd.merge_asof(merged_data, other_indicator_pivot.sort_values('time'), on='time', direction='nearest', suffixes=('', '_other'))

    return merged_data

def load_and_prepare_data(data_dir):
    data = load_data(data_dir)
    trade_event_data = identify_trade_results(data['trade_data'], data['event_data'])
    merged_data = merge_with_indicators(trade_event_data, data['indicator_data'])

    merged_data = merged_data[merged_data['result'].isin(['win', 'loss'])]
    merged_data['result'] = merged_data['result'].map({'win': 0, 'loss': 1})

    features = [col for col in merged_data.columns if col not in ['time', 'event', 'qty', 'price', 'event_event', 'qty_event', 'amount', 'result']]
    
    X = merged_data[features]
    X = X.dropna(axis=1, how='all')
    
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    y = merged_data['result']

    return train_test_split(X_imputed, y, test_size=0.3, random_state=42), merged_data

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': lgb.LGBMClassifier(random_state=42),
    }

    results = {}
    feature_importances = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else 'N/A',
            'classification_report': classification_report(y_test, y_pred)
        }

        importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.abs(model.coef_[0])
        feature_importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
        
        feature_importances[name] = feature_importance_df

    return models, results, feature_importances

def advanced_eda(data, feature_importances, trade_type, model_name, top_n=None, individual_indicator=None, target='result'):
    if trade_type == 'Long Only':
        data = data[data['event'].str.startswith('LE')]
    elif trade_type == 'Short Only':
        data = data[data['event'].str.startswith('SE')]

    if top_n:
        top_features = feature_importances[model_name]['feature'] if top_n == "ALL" else feature_importances[model_name]['feature'].head(top_n)
    elif individual_indicator:
        top_features = [individual_indicator]
    else:
        return

    top_features = [feature for feature in top_features if feature in data.columns]

    if not top_features:
        st.write("No matching features found in the data.")
        return

    descriptive_stats = data[top_features].describe()
    st.write(f"\nDescriptive Statistics for Top Features ({model_name}):")
    st.write(descriptive_stats)

    corr_matrix = data[top_features].corr()
    fig = px.imshow(corr_matrix, text_auto=True, title=f'Correlation Matrix of Top Features ({trade_type}, {model_name})')
    st.plotly_chart(fig)

    for feature in top_features:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=data[data[target] == 0][feature], name='Win', opacity=0.75))
        fig.add_trace(go.Histogram(x=data[data[target] == 1][feature], name='Loss', opacity=0.75))
        fig.update_layout(barmode='overlay', title=f'Distribution of {feature} ({trade_type}, {model_name})')
        st.plotly_chart(fig)

        fig = px.box(data, x=target, y=feature, title=f'{feature} by {target} ({trade_type}, {model_name})')
        st.plotly_chart(fig)

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

        fig = go.Figure()
        fig.add_trace(go.Violin(x=data[target], y=data[feature], box_visible=True, meanline_visible=True))
        fig.update_layout(title=f'Violin Plot of {feature} ({trade_type}, {model_name})')
        st.plotly_chart(fig)

    fig = px.scatter_matrix(data[top_features + [target]], dimensions=top_features, color=target)
    fig.update_layout(title=f'Scatter Matrix ({trade_type}, {model_name})')
    st.plotly_chart(fig)


def run_advanced_eda_indicators():
    st.subheader("Advanced EDAs=TEST on Indicators 2")

    if "base_dir" not in st.session_state:
        st.session_state.base_dir = "."

    base_dir = st.text_input("Base Directory", value=st.session_state.base_dir)
    data_dir = os.path.join(base_dir, "data/processed")

    if st.button("Load Data"):
        st.session_state.base_dir = base_dir

        if not os.path.exists(data_dir):
            st.write("Data directory not found. Please check the directory path.")
            return

        try:
            (X_train, X_test, y_train, y_test), merged_data = load_and_prepare_data(data_dir)
            models, results, feature_importances = train_and_evaluate_models(X_train, X_test, y_train, y_test)

            st.session_state.merged_data = merged_data
            st.session_state.feature_importances = feature_importances
            st.session_state.models = models
            st.session_state.results = results
            st.success("Data loaded successfully.")
        except Exception as e:
            st.write(f"Error loading data: {e}")
            return

    if "merged_data" in st.session_state:
        trade_type = st.selectbox("Select Trade Type", sorted(["Long Only", "Short Only", "Long & Short"]))
        model_name = st.selectbox("Select Model", sorted(["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"]))
        top_n = st.selectbox("Select Top N Indicators", sorted([None, 10, 5, 3, "ALL"], key=lambda x: (str(x) if x is not None else "")))
        individual_indicator = st.selectbox("Select Individual Indicator", sorted(st.session_state.feature_importances[model_name]['feature']))

        if st.button("Run EDA"):
            advanced_eda(st.session_state.merged_data, st.session_state.feature_importances, trade_type=trade_type, model_name=model_name, top_n=top_n, individual_indicator=individual_indicator)
