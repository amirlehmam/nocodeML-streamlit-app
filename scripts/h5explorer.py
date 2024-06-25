import pandas as pd
import numpy as np
import h5py

# Function to load and preprocess the CSV file
def load_and_preprocess_csv(filepath):
    # Load the CSV file
    df = pd.read_csv(filepath, delimiter=';', header=None, engine='python', on_bad_lines='skip')

    # Define the base columns
    base_columns = ['Type', 'MarketDataType', 'Timestamp', 'Offset', 'Operation', 'OrderBookPosition', 'MarketMaker', 'Price', 'Volume']
    extra_columns = [f'Extra{i}' for i in range(len(df.columns) - len(base_columns))]
    df.columns = base_columns + extra_columns

    # Parse timestamps
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y%m%d%H%M%S')

    # Clean and convert price columns
    for column in df.columns:
        if column not in base_columns:
            df[column] = df[column].astype(str).str.replace(',', '.').astype(float, errors='coerce')
    
    return df

# Function to filter valid price data
def filter_valid_prices(df):
    valid_prices = pd.DataFrame()

    for column in df.columns:
        if column not in ['Type', 'MarketDataType', 'Timestamp', 'Offset', 'Operation', 'OrderBookPosition', 'MarketMaker', 'Volume']:
            # Ensure the column is converted to numeric values
            df[column] = pd.to_numeric(df[column], errors='coerce')
            valid_price_indices = (df[column] >= 17000) & (df[column] <= 19000)
            if valid_price_indices.any():
                temp_df = df[valid_price_indices].copy()
                temp_df['Price'] = df[column][valid_price_indices]
                valid_prices = pd.concat([valid_prices, temp_df], ignore_index=True)
    
    return valid_prices

# Function to read and display the contents of the HDF5 file
def read_hdf5_file(filepath):
    with h5py.File(filepath, 'r') as f:
        print("Datasets in the HDF5 file:")
        for name in f:
            print(name)

        timestamps = f['L1/timestamp'][:]
        prices = f['L1/price'][:]
        volumes = f['L1/volume'][:]

        # Convert timestamps to datetime for better readability
        dates = pd.to_datetime(timestamps, unit='ns')

        # Create a DataFrame for easy inspection
        df_hdf5 = pd.DataFrame({
            'Timestamp': dates,
            'Price': prices,
            'Volume': volumes
        })

        return df_hdf5

# Paths to the files
csv_filepath = 'C:/Users/Administrator/Desktop/sample.csv'
hdf5_filepath = 'C:/Users/Administrator/Desktop/sample_fixed.h5'

# Load and preprocess the CSV file
df_csv = load_and_preprocess_csv(csv_filepath)

# Filter valid price data
valid_prices = filter_valid_prices(df_csv)

# Save to HDF5
with h5py.File(hdf5_filepath, 'w') as f:
    f.create_dataset('L1/timestamp', data=valid_prices['Timestamp'].astype('int64').values)
    f.create_dataset('L1/price', data=valid_prices['Price'].values)
    if 'Volume' in valid_prices:
        f.create_dataset('L1/volume', data=valid_prices['Volume'].fillna(0).values)

print("Data saved to HDF5.")

# Read and display the HDF5 file contents
df_hdf5 = read_hdf5_file(hdf5_filepath)
print(df_hdf5.head(90))
