import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from joblib import Parallel, delayed

def process_row(row):
    # Convert date and time offset to datetime object with sub-second granularity
    timestamp = pd.to_datetime(row['Date'], format='%Y%m%d%H%M%S') + pd.to_timedelta(int(row['TimeOffset']) * 100, unit='ns')
    
    last_price = ''
    bid_price = ''
    ask_price = ''
    volume = ''
    
    if row['Type'] == 'L1':
        if row['OrderType'] == 0:  # Ask
            ask_price = str(row['Price']).replace(',', '.')
            volume = row['Volume']
        elif row['OrderType'] == 1:  # Bid
            bid_price = str(row['Price']).replace(',', '.')
            volume = row['Volume']
        elif row['OrderType'] == 2:  # Last
            last_price = str(row['Price']).replace(',', '.')
            volume = row['Volume']
    elif row['Type'] == 'L2':
        if row['OrderType'] == 0:  # Ask
            ask_price = str(row['L2Price']).replace(',', '.')
        elif row['OrderType'] == 1:  # Bid
            bid_price = str(row['L2Price']).replace(',', '.')
    
    return [timestamp.strftime('%Y%m%d %H%M%S %f'), last_price, bid_price, ask_price, volume]

def convert_nrd_to_tick(input_file, output_file):
    # Read the input CSV file
    df = pd.read_csv(input_file, delimiter=';', header=None, 
                     names=['Type', 'OrderType', 'Date', 'TimeOffset', 'Price', 'Volume', 'Unused', 'L2Price', 'L2Volume'])

    # Prepare the data for multiprocessing
    data = df.to_dict(orient='records')

    # Use joblib to process rows in parallel
    results = Parallel(n_jobs=-1)(delayed(process_row)(row) for row in tqdm(data, desc="Processing rows"))

    # Filter out rows with no prices to avoid incomplete data
    tick_data = [tick for tick in results if any(tick[1:4])]

    # Save the tick data to output file
    with open(output_file, 'w') as f:
        for tick in tick_data:
            f.write(';'.join(map(str, tick)) + '\n')

# Example usage
if __name__ == "__main__":
    input_file = '20221024.csv'
    output_file = 'historical_tick_data.csv'
    convert_nrd_to_tick(input_file, output_file)