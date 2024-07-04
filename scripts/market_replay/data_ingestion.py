import pandas as pd
import h5py
import os
import psycopg2
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing

# PostgreSQL database credentials
db_credentials = {
    "dbname": "defaultdb",
    "user": "doadmin",
    "password": "AVNS_hnzmIdBmiO7aj5nylWW",
    "host": "nocodemldb-do-user-16993120-0.c.db.ondigitalocean.com",
    "port": 25060,
    "sslmode": "require"
}

def connect_db():
    conn = psycopg2.connect(**db_credentials)
    return conn

def store_h5_in_db(file_path):
    start_time = time.time()
    
    conn = connect_db()
    cursor = conn.cursor()
    
    with open(file_path, 'rb') as file:
        binary_data = file.read()

    filename = os.path.basename(file_path)
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS h5_files (
        filename TEXT PRIMARY KEY,
        data BYTEA
    );
    ''')

    cursor.execute('''
    INSERT INTO h5_files (filename, data) VALUES (%s, %s)
    ON CONFLICT (filename) DO UPDATE SET data = EXCLUDED.data;
    ''', (filename, binary_data))

    conn.commit()
    cursor.close()
    conn.close()

    print(f"Storing {file_path} in DB took {time.time() - start_time:.2f} seconds")

def process_file(file_path, output_directory):
    start_time = time.time()
    df = pd.read_csv(file_path, delimiter=';', header=None)

    # Separate L1 and L2 records
    df_l1 = df[df[0] == 'L1'].copy()
    df_l2 = df[df[0] == 'L2'].copy()

    # Define column names for L1 and L2
    l1_columns = [
        'RecordType', 'MarketDataType', 'Timestamp', 'Offset', 'Price', 
        'Volume', 'Empty1', 'Empty2', 'Empty3'
    ]
    l2_columns = [
        'RecordType', 'MarketDataType', 'Timestamp', 'Offset', 'Operation', 
        'Position', 'MarketMaker', 'Price', 'Volume'
    ]

    # Log the shape of the DataFrames before assigning columns
    print(f"Processing {file_path}: df_l1 shape = {df_l1.shape}, df_l2 shape = {df_l2.shape}")

    # Check if the number of columns matches the expected number
    if df_l1.shape[1] != len(l1_columns):
        print(f"Error: {file_path} has {df_l1.shape[1]} columns in L1, expected {len(l1_columns)}. Skipping this file.")
        return f"Error processing {file_path}"
    if df_l2.shape[1] != len(l2_columns):
        print(f"Error: {file_path} has {df_l2.shape[1]} columns in L2, expected {len(l2_columns)}. Skipping this file.")
        return f"Error processing {file_path}"

    # Assign column names
    df_l1.columns = l1_columns
    df_l2.columns = l2_columns

    # Replace commas with dots in Price and Volume columns
    df_l1['Price'] = df_l1['Price'].str.replace(',', '.').astype(float)
    df_l1['Volume'] = df_l1['Volume'].str.replace(',', '.').astype(float)
    df_l2['Price'] = df_l2['Price'].str.replace(',', '.').astype(float)
    df_l2['Volume'] = df_l2['Volume'].str.replace(',', '.').astype(float)

    # Convert columns to appropriate data types using .loc to avoid SettingWithCopyWarning
    df_l1['Timestamp'] = pd.to_datetime(df_l1['Timestamp'], format='%Y%m%d%H%M%S%f', errors='coerce')
    df_l1['Offset'] = df_l1['Offset'].astype(int)
    df_l1['Price'] = pd.to_numeric(df_l1['Price'], errors='coerce')
    df_l1['Volume'] = pd.to_numeric(df_l1['Volume'], errors='coerce')

    df_l2['Timestamp'] = pd.to_datetime(df_l2['Timestamp'], format='%Y%m%d%H%M%S%f', errors='coerce')
    df_l2['Offset'] = df_l2['Offset'].astype(int)
    df_l2['Operation'] = df_l2['Operation'].astype(int)
    df_l2['Position'] = df_l2['Position'].astype(int)
    df_l2['MarketMaker'] = pd.to_numeric(df_l2['MarketMaker'], errors='coerce').fillna(0).astype(int)
    df_l2['Price'] = pd.to_numeric(df_l2['Price'], errors='coerce')
    df_l2['Volume'] = pd.to_numeric(df_l2['Volume'], errors='coerce')

    # Convert Timestamp to string for HDF5 storage
    df_l1['Timestamp'] = df_l1['Timestamp'].astype(str)
    df_l2['Timestamp'] = df_l2['Timestamp'].astype(str)

    # Create HDF5 file named after the CSV file
    filename = os.path.basename(file_path)
    hdf5_file = os.path.join(output_directory, f"{filename.split('.')[0]}.h5")
    if os.path.exists(hdf5_file):
        os.remove(hdf5_file)

    with h5py.File(hdf5_file, 'w') as f:
        # Store L1 data
        grp_l1 = f.create_group('L1')
        for col in df_l1.columns:
            grp_l1.create_dataset(col, data=df_l1[col].values.astype('S'), compression="gzip", compression_opts=9)  # Store as bytes with compression

        # Store L2 data
        grp_l2 = f.create_group('L2')
        for col in df_l2.columns:
            grp_l2.create_dataset(col, data=df_l2[col].values.astype('S'), compression="gzip", compression_opts=9)  # Store as bytes with compression

    # Store the .h5 file in the database
    store_h5_in_db(hdf5_file)

    # Remove the local .h5 file after storing it in the database
    os.remove(hdf5_file)

    process_time = time.time() - start_time
    print(f"{filename} processed in {process_time:.2f} seconds")
    return f"{filename} processed in {process_time:.2f} seconds"

def main():
    # Directory containing the CSV files
    input_directory = 'C:/Users/Administrator/Documents/NinjaTrader 8/db/replay/temp_preprocessed/'

    # Output directory for temporary HDF5 files
    output_directory = 'C:/Users/Administrator/Desktop/nocodeML-streamlit-app/scripts/market_replay/temp/'

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Get list of CSV files
    csv_files = [os.path.join(input_directory, filename) for filename in os.listdir(input_directory) if filename.endswith('.csv')]

    # Determine the number of workers
    num_workers = min(multiprocessing.cpu_count(), len(csv_files))

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file, file, output_directory): file for file in csv_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            print(future.result())

if __name__ == "__main__":
    main()
