import pandas as pd
import numpy as np
import os
import re
import logging
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO)

# Database connection details
DB_CONFIG = {
    'dbname': 'defaultdb',
    'user': 'doadmin',
    'password': 'YOUR_PASSWORD',  # Replace 'YOUR_PASSWORD' with your actual password
    'host': 'nocodemldb-do-user-16993120-0.c.db.ondigitalocean.com',
    'port': 25060,
    'sslmode': 'require'
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)

def parse_data(lines):
    trade_data = []
    indicator_data = []
    event_data = []
    account_data = []
    signal_data = []

    for line in lines:
        parts = line.strip().split(';')
        if len(parts) < 2:
            continue

        timestamp = parts[0]
        event_type = parts[1]

        if event_type == 'Indicator':
            # Parse indicators
            indicators = parts[9:]  # Assuming indicators start from the 10th column
            for i in range(0, len(indicators), 2):
                if i + 1 < len(indicators):
                    indicator_name = indicators[i]
                    indicator_value = indicators[i + 1].replace(',', '.')
                    try:
                        indicator_value = float(indicator_value)
                    except ValueError:
                        indicator_value = np.nan
                    indicator_data.append([timestamp, indicator_name, indicator_value])
        elif event_type == 'Signal':
            # Parse signals if any
            signals = parts[9:]
            signal_data.append([timestamp] + signals)
        elif event_type in ('LE1', 'LE2', 'LE3', 'LE4', 'LE5', 'LX', 'SE1', 'SE2', 'SE3', 'SE4', 'SE5', 'SX', 'Profit target', 'Parabolic stop', 'Stop-Loss', 'Exit'):
            # Parse trade events
            qty = parts[2]
            price = parts[3].replace(',', '.')
            trade_id = parts[4] if len(parts) > 4 else None
            trade_data.append([timestamp, event_type, qty, price, trade_id])
        elif event_type in ('Commission', 'Strategy', 'Account', 'Profit', 'Loss'):
            # Parse account updates
            amount = parts[-1].replace(',', '.').replace('$', '')
            account_data.append([timestamp, event_type, amount])
        else:
            # Other events
            event_data.append([timestamp, event_type] + parts[2:])

    # Convert to DataFrames
    trade_df = pd.DataFrame(trade_data, columns=['Timestamp', 'Event', 'Qty', 'Price', 'Trade_ID'])
    indicator_df = pd.DataFrame(indicator_data, columns=['Timestamp', 'Indicator_Name', 'Indicator_Value'])
    account_df = pd.DataFrame(account_data, columns=['Timestamp', 'Event', 'Amount'])
    event_df = pd.DataFrame(event_data)
    signal_df = pd.DataFrame(signal_data)

    # Convert data types
    trade_df['Timestamp'] = pd.to_datetime(trade_df['Timestamp'])
    trade_df['Qty'] = pd.to_numeric(trade_df['Qty'], errors='coerce')
    trade_df['Price'] = pd.to_numeric(trade_df['Price'], errors='coerce')

    indicator_df['Timestamp'] = pd.to_datetime(indicator_df['Timestamp'])
    indicator_df['Indicator_Value'] = pd.to_numeric(indicator_df['Indicator_Value'], errors='coerce')

    account_df['Timestamp'] = pd.to_datetime(account_df['Timestamp'])
    account_df['Amount'] = pd.to_numeric(account_df['Amount'], errors='coerce')

    return trade_df, indicator_df, account_df, event_df, signal_df

def merge_data(trade_df, indicator_df):
    # Pivot indicators to have one row per timestamp
    indicator_pivot = indicator_df.pivot(index='Timestamp', columns='Indicator_Name', values='Indicator_Value').reset_index()

    # Merge trade data with indicators
    merged_df = pd.merge(trade_df, indicator_pivot, on='Timestamp', how='left')

    return merged_df

def create_lag_features(df, indicator_columns, max_lag=3):
    for col in indicator_columns:
        for lag in range(1, max_lag + 1):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

def create_rolling_features(df, indicator_columns, windows=[3, 5, 10]):
    for col in indicator_columns:
        for window in windows:
            df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window).mean()
            df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window).std()
    return df

def create_interaction_features(df, indicator_columns):
    from itertools import combinations
    for col1, col2 in combinations(indicator_columns, 2):
        df[f'{col1}_mul_{col2}'] = df[col1] * df[col2]
        df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-5)
    return df

def create_time_features(df):
    df['hour'] = df['Timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df

def handle_missing_values(df):
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df

def advanced_data_ingestion(file_content):
    # Read the file content
    lines = file_content.decode('utf-8').splitlines()

    # Parse data
    trade_df, indicator_df, account_df, event_df, signal_df = parse_data(lines)

    # Merge data
    merged_df = merge_data(trade_df, indicator_df)

    # List of indicator columns
    indicator_columns = indicator_df['Indicator_Name'].unique().tolist()

    # Feature Engineering
    merged_df = create_lag_features(merged_df, indicator_columns)
    merged_df = create_rolling_features(merged_df, indicator_columns)
    merged_df = create_interaction_features(merged_df, indicator_columns)
    merged_df = create_time_features(merged_df)
    merged_df = handle_missing_values(merged_df)

    # Additional feature engineering can be added here

    return merged_df

def add_missing_columns(table_name, required_columns):
    conn = get_db_connection()
    cur = conn.cursor()

    # Get existing columns
    cur.execute(f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = '{table_name}';
    """)
    existing_columns = {row[0] for row in cur.fetchall()}

    # Identify missing columns
    missing_columns = required_columns - existing_columns

    # Add missing columns
    for column in missing_columns:
        cur.execute(f"""
            ALTER TABLE {table_name}
            ADD COLUMN "{column}" TEXT;
        """)
        logging.info(f"Added missing column: {column}")

    conn.commit()
    cur.close()
    conn.close()

def save_advanced_data_to_db(df, table_name='enhanced_trade_data'):
    conn = get_db_connection()
    cur = conn.cursor()

    # Sanitize column names
    df.columns = [re.sub(r'\W|^(?=\d)', '_', col).lstrip('_') for col in df.columns]

    # Ensure table exists and has the correct columns
    add_missing_columns(table_name, set(df.columns))

    # Insert data
    columns = df.columns.tolist()
    values = [tuple(x) for x in df.to_numpy()]
    insert_sql = f"""
        INSERT INTO {table_name} ({', '.join(f'"{col}"' for col in columns)})
        VALUES %s
    """
    execute_values(cur, insert_sql, values)

    conn.commit()
    cur.close()
    conn.close()

def run_advanced_data_ingestion():
    st.title("Advanced Data Ingestion and Feature Engineering")

    # File upload
    uploaded_file = st.file_uploader("Upload your data file", type=["csv"])
    if uploaded_file is not None:
        # Read and process the file
        file_content = uploaded_file.getvalue()
        enhanced_df = advanced_data_ingestion(file_content)

        st.success("Data successfully ingested and enhanced with advanced features.")
        st.write("Sample of the enhanced data:")
        st.dataframe(enhanced_df.head())

        # Save to database
        save_advanced_data_to_db(enhanced_df)
        st.success("Enhanced data saved to the database.")

if __name__ == "__main__":
    run_advanced_data_ingestion()
