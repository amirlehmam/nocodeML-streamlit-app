import csv
import h5py
from tqdm import tqdm

def parse_csv_to_dict(filepath):
    data = {
        'L1': [],
        'L2': []
    }
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        total_lines = sum(1 for row in reader)  # Calculate total lines for progress bar
        f.seek(0)  # Reset file pointer to beginning
        
        for row in tqdm(reader, total=total_lines, desc="Parsing CSV"):
            if row[0] == 'L1':
                record = {
                    'type': row[1],
                    'timestamp': row[2],
                    'offset': int(row[3]),
                    'price': float(row[4].replace(',', '.')),
                    'volume': float(row[5].replace(',', '.'))  # Changed to float
                }
                data['L1'].append(record)
            elif row[0] == 'L2':
                record = {
                    'type': row[1],
                    'timestamp': row[2],
                    'offset': int(row[3]),
                    'operation': int(row[4]),
                    'position': int(row[5]),
                    'market_maker_id': row[6],
                    'price': float(row[7].replace(',', '.')),
                    'volume': float(row[8].replace(',', '.'))  # Changed to float
                }
                data['L2'].append(record)
    return data

def convert_to_hdf5(data, output_filepath):
    with h5py.File(output_filepath, 'w') as f:
        l1_group = f.create_group('L1')
        l2_group = f.create_group('L2')

        for key in tqdm(['timestamp', 'offset', 'price', 'volume'], desc="Converting L1 Data"):
            l1_group.create_dataset(key, data=[record[key] for record in data['L1']])

        for key in tqdm(['timestamp', 'offset', 'operation', 'position', 'market_maker_id', 'price', 'volume'], desc="Converting L2 Data"):
            l2_group.create_dataset(key, data=[record[key] for record in data['L2']])

# Example usage
csv_filepath = 'C:/Users/Administrator/Documents/NinjaTrader 8/db/replay/temp_preprocessed/20240301.csv'
output_filepath = 'C:/Users/Administrator/Documents/NinjaTrader 8/db/replay/temp_preprocessed/hdf5/20240301.h5'

parsed_data = parse_csv_to_dict(csv_filepath)
convert_to_hdf5(parsed_data, output_filepath)
