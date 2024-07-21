import pandas as pd
from datetime import datetime, timedelta
from joblib import Parallel, delayed

# Function to parse and format individual records
def parse_and_format(record):
    parts = record.split(';')
    if parts[0] == 'L1':
        # L1 Record
        record_type, bid_ask, timestamp, offset, price, volume = parts
        parsed_record = {
            'type': 'L1',
            'bid_ask': int(bid_ask),
            'timestamp': timestamp,
            'offset': int(offset),
            'price': float(price.replace(',', '.')),
            'volume': int(volume)
        }
    elif parts[0] == 'L2':
        # L2 Record
        record_type, bid_ask, timestamp, offset, operation, order_pos, mm_id, price, volume = parts
        parsed_record = {
            'type': 'L2',
            'bid_ask': int(bid_ask),
            'timestamp': timestamp,
            'offset': int(offset),
            'operation': int(operation),
            'order_pos': int(order_pos),
            'mm_id': mm_id,
            'price': float(price.replace(',', '.')),
            'volume': int(volume)
        }
    return parsed_record

# Function to convert timestamp and offset to required format
def format_timestamp(ts, offset):
    dt = datetime.strptime(ts, '%Y%m%d%H%M%S')
    dt += timedelta(microseconds=offset // 10)
    formatted_timestamp = dt.strftime('%Y%m%d %H%M%S') + f' {offset % 10000000:07d}'
    return formatted_timestamp

# Function to format parsed records
def format_record(record):
    formatted_timestamp = format_timestamp(record['timestamp'], record['offset'])
    return f"{formatted_timestamp};{record['price']:.2f};{record['volume']}"

# Function to process a chunk of data
def process_chunk(chunk):
    records = chunk.split('\n')
    parsed_records = [parse_and_format(record) for record in records if record.strip()]
    formatted_records = [format_record(record) for record in parsed_records]
    return formatted_records

# Function to process the entire file
def process_file(file_path, output_prefix):
    # Read file in chunks
    chunks = []
    with open(file_path, 'r') as file:
        chunk_size = 10000  # Adjust based on memory and performance requirements
        chunk = []
        for line in file:
            chunk.append(line.strip())
            if len(chunk) >= chunk_size:
                chunks.append('\n'.join(chunk))
                chunk = []
        if chunk:
            chunks.append('\n'.join(chunk))
    
    # Process chunks in parallel
    results = Parallel(n_jobs=-1)(delayed(process_chunk)(chunk) for chunk in chunks)
    
    # Flatten results
    flat_results = [item for sublist in results for item in sublist]
    
    # Write to output file
    output_file = f"{output_prefix}.Last.txt"
    
    with open(output_file, 'w') as file:
        file.write('\n'.join(flat_results))
    
    print(f"Data processed and saved to {output_file}")

# Define input file and output prefix
input_file = "20221024.csv"
output_prefix = "NQ 12-22"

# Process the file
process_file(input_file, output_prefix)
