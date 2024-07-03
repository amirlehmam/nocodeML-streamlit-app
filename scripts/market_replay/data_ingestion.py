import pandas as pd
import h5py
import os
import psycopg2
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

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

def process_file(file_path, output_directory):
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

    # Assign column names
    df_l1.columns = l1_columns
    df_l2.columns = l2_columns

    # Replace commas with dots in Price and Volume columns
    df_l1['Price'] = df_l1['Price'].astype(str).str.replace(',', '.')
    df_l1['Volume'] = df_l1['Volume'].astype(str).str.replace(',', '.')
    df_l2['Price'] = df_l2['Price'].astype(str).str.replace(',', '.')
    df_l2['Volume'] = df_l2['Volume'].astype(str).str.replace(',', '.')

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

    return f"{filename} processed"

def main():
    # Directory containing the CSV files
    input_directory = 'C:/Users/Administrator/Documents/NinjaTrader 8/db/replay/temp_preprocessed/test/'

    # Output directory for temporary HDF5 files
    output_directory = 'C:/Users/Administrator/Desktop/nocodeML-streamlit-app/scripts/market_replay/temp/'

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Get list of CSV files
    csv_files = [os.path.join(input_directory, filename) for filename in os.listdir(input_directory) if filename.endswith('.csv')]

    # Process files in parallel
    with ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(process_file, csv_files, [output_directory]*len(csv_files)), total=len(csv_files), desc="Processing files"):
            print(result)

if __name__ == "__main__":
    main()
