import pandas as pd
import os
import streamlit as st

def get_file_path(base_dir, relative_path):
    return os.path.join(base_dir, relative_path)

def clean_and_parse_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        lines = file.readlines()
    
    header = lines[0].strip().split(';')
    data_lines = lines[1:]

    trade_data = []
    indicator_data = []
    event_data = []
    signal_data = []

    for line in data_lines:
        parts = line.strip().split(';')
        if len(parts) < 2:
            print(f"Error parsing line: {line}")
            continue
        
        timestamp = parts[0]
        event_type = parts[1]

        if event_type == 'Indicator':
            indicators = parts[7:]
            for i in range(0, len(indicators), 2):
                if i + 1 < len(indicators):
                    indicator_name = indicators[i]
                    indicator_value = indicators[i + 1].replace(',', '.')
                    if indicator_name and indicator_value:  # Ensure both name and value are not empty
                        indicator_data.append([timestamp, indicator_name, float(indicator_value)])
        elif event_type == 'Signal':
            signals = parts[8:]
            row = [timestamp, 'Signal']
            for i in range(0, len(signals), 2):
                if i + 1 < len(signals):
                    signal_name = signals[i]
                    signal_value = signals[i + 1].replace(',', '.')
                    row.extend([signal_name, signal_value])
            signal_data.append(row)
        elif event_type in ('LE1','LE2','LE3','LE4','LE5','LX','SE1', 'SE2', 'SE3', "SE4", 'SE5', 'SX', 'Profit target', 'Parabolic stop'):
            trade_data.append([timestamp, event_type, parts[2], parts[3].replace(',', '.')])
        else:
            event = parts[1]
            amount = parts[-1].replace(',', '.')
            event_data.append([timestamp, event, amount])
    
    trade_columns = ['time', 'event', 'qty', 'price']
    trade_df = pd.DataFrame(trade_data, columns=trade_columns)
    
    indicator_df = pd.DataFrame(indicator_data, columns=['time', 'indicator_name', 'indicator_value'])
    
    event_columns = ['time', 'event', 'amount']
    event_df = pd.DataFrame(event_data, columns=event_columns)

    max_signal_length = max(len(row) for row in signal_data)
    signal_columns = ['time', 'event'] + [f'signal_name{i//2+1}' if i % 2 == 0 else 'value' for i in range(max_signal_length - 2)]
    signal_df = pd.DataFrame(signal_data, columns=signal_columns)

    return {
        'trade_data': trade_df, 
        'indicator_data': indicator_df, 
        'event_data': event_df, 
        'signal_data': signal_df
    }

def calculate_indicators(data):
    trade_df = data['trade_data']
    indicator_df = data['indicator_data']

    market_value_indicators = indicator_df[indicator_df['indicator_value'] > 10000]

    merged_df = pd.merge(market_value_indicators, trade_df[['time', 'price']], on='time', how='left')

    merged_df['price'] = pd.to_numeric(merged_df['price'], errors='coerce')
    merged_df['indicator_value'] = pd.to_numeric(merged_df['indicator_value'], errors='coerce')

    merged_df = merged_df.dropna(subset=['price', 'indicator_value'])

    merged_df['binary_indicator'] = (merged_df['price'] > merged_df['indicator_value']).astype(int)
    merged_df['percent_away'] = ((merged_df['price'] - merged_df['indicator_value']) / merged_df['indicator_value']) * 100

    non_market_value_indicators = indicator_df[indicator_df['indicator_value'] <= 10000].copy()
    non_market_value_indicators.loc[:, 'binary_indicator'] = None
    non_market_value_indicators.loc[:, 'percent_away'] = None

    merged_df = merged_df.dropna(axis=1, how='all')
    non_market_value_indicators = non_market_value_indicators.dropna(axis=1, how='all')

    final_indicator_df = pd.concat([merged_df, non_market_value_indicators])

    final_indicator_df = final_indicator_df.sort_values(by='time').reset_index(drop=True)

    return final_indicator_df

def save_dataframes(data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    trade_path = os.path.join(output_dir, "trade_data.csv")
    data['trade_data'].to_csv(trade_path, index=False)
    print(f"Saved trade data to {trade_path}")

    indicator_path = os.path.join(output_dir, "indicator_data.csv")
    data['indicator_data'].to_csv(indicator_path, index=False)
    print(f"Saved indicator data to {indicator_path}")

    event_path = os.path.join(output_dir, "event_data.csv")
    data['event_data'].to_csv(event_path, index=False)
    print(f"Saved event data to {event_path}")

    signal_path = os.path.join(output_dir, "signal_data.csv")
    data['signal_data'].to_csv(signal_path, index=False)
    print(f"Saved signal data to {signal_path}")

def parse_parameters(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    parameters = []
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(': ', 1)
            parameters.append([key, value.replace(',', '.')])

    parameters_df = pd.DataFrame(parameters, columns=['Parameter', 'Value'])
    return parameters_df

def save_parameters(parameters_df, output_path):
    parameters_df.to_csv(output_path, index=False)
    print(f"Saved parameters to {output_path}")

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
    relevant_trades = ['LE1','LE2','LE3','LE4','LE5','LX','SE1', 'SE2', 'SE3', 'SE4', 'SE5', 'SX']
    trade_data = trade_data[trade_data['event'].isin(relevant_trades)]
    
    trade_event_data = pd.merge_asof(trade_data.sort_values('time'), event_data.sort_values('time'), on='time', direction='forward', suffixes=('', '_event'))

    def classify_trade(row):
        if pd.notna(row['event_event']):
            if 'Profit' in row['event_event']:
                return 'win'
            elif 'Loss' in row['event_event']:
                return 'loss'
        return 'unknown'
    
    trade_event_data['result'] = trade_event_data.apply(classify_trade, axis=1)
    
    trade_event_data = trade_event_data[trade_event_data['result'] != 'unknown']
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

def verify_trade_parsing(data_dir, output_dir):
    data = load_data(data_dir)
    trade_data = data['trade_data']
    event_data = preprocess_events(data['event_data'])
    indicator_data = data['indicator_data']

    trade_event_data = identify_trade_results(trade_data, event_data)

    merged_data = merge_with_indicators(trade_event_data, indicator_data)

    result_distribution = merged_data['result'].value_counts()

    st.write(f"\nDistribution of trade results:\n{result_distribution}")

    sample_size = min(10, len(merged_data))  # Adjust sample size
    if sample_size > 0:
        sample_classified_trades = merged_data.sample(n=sample_size)
        print(f"\nSample of classified trades (showing {sample_size}):")
        print(sample_classified_trades)
    else:
        print("\nNo classified trades available for sampling.")

    output_file = os.path.join(output_dir, "merged_trade_indicator_event.csv")
    merged_data.to_csv(output_file, index=False)
    st.write(f"Saved merged data to {output_file}")

def run_data_ingestion_preparation():
    st.subheader("Data Ingestion and Preparation")
    if "base_dir" not in st.session_state:
        st.session_state.base_dir = "."

    base_dir = st.text_input("Base Directory", value=st.session_state.base_dir)
    data_output_dir = get_file_path(base_dir, "data/processed")
    raw_data_dir = get_file_path(base_dir, "data/raw")

    uploaded_file = st.file_uploader("Choose a data file", type=["csv"])
    if uploaded_file is not None:
        raw_file_path = os.path.join(raw_data_dir, uploaded_file.name)
        with open(raw_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File {uploaded_file.name} uploaded successfully to {raw_file_path}")

    file_dropdown = st.selectbox("Select a raw data file", os.listdir(raw_data_dir) if os.path.exists(raw_data_dir) else [])
    if file_dropdown:
        file_path = get_file_path(raw_data_dir, file_dropdown)
        data = clean_and_parse_data(file_path)
        st.write(data['trade_data'].head(15))
        st.write(data['indicator_data'].head(15))
        st.write(data['event_data'].head(15))
        st.write(data['signal_data'].head(15))
        
        data['indicator_data'] = calculate_indicators(data)
        save_dataframes(data, data_output_dir)
        
        st.success("Data successfully parsed and saved.")

    uploaded_param_file = st.file_uploader("Choose a parameter file", key="params", type=["csv", "txt"])
    if uploaded_param_file is not None:
        param_file_path = os.path.join(raw_data_dir, "params", uploaded_param_file.name)
        with open(param_file_path, "wb") as f:
            f.write(uploaded_param_file.getbuffer())
        st.success(f"Parameter file {uploaded_param_file.name} uploaded successfully to {param_file_path}")

        parameters_df = parse_parameters(param_file_path)
        st.write(parameters_df)
        
        save_parameters(parameters_df, os.path.join(data_output_dir, "parameters.csv"))
        
        st.success("Parameters successfully parsed and saved.")
    
    if st.button("Verify Trade Parsing"):
        verify_trade_parsing(data_output_dir, data_output_dir)
