import pandas as pd
import h5py
import os

# Read CSV file
file_path = 'C:/Users/Administrator/Documents/NinjaTrader 8/db/replay/temp_preprocessed/20240509.csv'
df = pd.read_csv(file_path, delimiter=';', header=None)

# Debug: Print first few rows of the CSV data
print("Initial CSV data:")
print(df.head(20))

# Separate L1 and L2 records
df_l1 = df[df[0] == 'L1'].copy()
df_l2 = df[df[0] == 'L2'].copy()

# Debug: Print first few rows of L1 and L2 records
print("L1 records:")
print(df_l1.head(20))
print("L2 records:")
print(df_l2.head(20))

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
df_l1.loc[:, 'Timestamp'] = pd.to_datetime(df_l1['Timestamp'], format='%Y%m%d%H%M%S%f')
df_l1.loc[:, 'Offset'] = df_l1['Offset'].astype(int)
df_l1.loc[:, 'Price'] = pd.to_numeric(df_l1['Price'], errors='coerce')
df_l1.loc[:, 'Volume'] = pd.to_numeric(df_l1['Volume'], errors='coerce')

df_l2.loc[:, 'Timestamp'] = pd.to_datetime(df_l2['Timestamp'], format='%Y%m%d%H%M%S%f')
df_l2.loc[:, 'Offset'] = df_l2['Offset'].astype(int)
df_l2.loc[:, 'Operation'] = df_l2['Operation'].astype(int)
df_l2.loc[:, 'Position'] = df_l2['Position'].astype(int)
df_l2.loc[:, 'MarketMaker'] = pd.to_numeric(df_l2['MarketMaker'], errors='coerce').fillna(0).astype(int)
df_l2.loc[:, 'Price'] = pd.to_numeric(df_l2['Price'], errors='coerce')
df_l2.loc[:, 'Volume'] = pd.to_numeric(df_l2['Volume'], errors='coerce')

# Debug: Print first few rows after data type conversion
print("L1 records after conversion:")
print(df_l1.head(20))
print("L2 records after conversion:")
print(df_l2.head(20))

# Convert Timestamp to string for HDF5 storage
df_l1.loc[:, 'Timestamp'] = df_l1['Timestamp'].astype(str)
df_l2.loc[:, 'Timestamp'] = df_l2['Timestamp'].astype(str)

# Debug: Print first few rows after timestamp conversion
print("L1 records after timestamp conversion:")
print(df_l1.head(20))
print("L2 records after timestamp conversion:")
print(df_l2.head(20))

# Create HDF5 file and store data
hdf5_file = 'market_replay_data.h5'
if os.path.exists(hdf5_file):
    os.remove(hdf5_file)

with h5py.File(hdf5_file, 'w') as f:
    # Store L1 data
    grp_l1 = f.create_group('L1')
    for col in df_l1.columns:
        grp_l1.create_dataset(col, data=df_l1[col].values.astype('S'))  # Store as bytes

    # Store L2 data
    grp_l2 = f.create_group('L2')
    for col in df_l2.columns:
        grp_l2.create_dataset(col, data=df_l2[col].values.astype('S'))  # Store as bytes

print(f"Data stored in {hdf5_file}")

# Debug: Verify the stored data
with h5py.File(hdf5_file, 'r') as f:
    print("L1 data from HDF5:")
    for col in df_l1.columns:
        print(f"{col}: {f['L1'][col][:20]}")
    
    print("L2 data from HDF5:")
    for col in df_l2.columns:
        print(f"{col}: {f['L2'][col][:20]}")
