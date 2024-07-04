import pandas as pd
import numpy as np
import mplfinance as mpf
from matplotlib import animation
import h5py
import psycopg2
import tempfile
from tqdm import tqdm

# Assuming renkodf.py is in the same directory
from renkodf import RenkoWS

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

def fetch_available_dates():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT filename FROM h5_files;')
    rows = cursor.fetchall()
    conn.close()
    
    dates = [row[0].replace('.h5', '') for row in rows]
    return sorted(dates)

def load_h5_from_db(filename):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT data FROM h5_files WHERE filename = %s;', (filename,))
    binary_data = cursor.fetchone()[0]
    conn.close()
    
    temp_dir = tempfile.gettempdir()
    file_path = f"{temp_dir}/{filename}.h5"
    
    with open(file_path, "wb") as f:
        f.write(binary_data)
    
    return file_path

def load_data_from_db(start_date, end_date):
    conn = connect_db()
    cursor = conn.cursor()
    
    query = '''
    SELECT filename, data FROM h5_files WHERE filename >= %s AND filename <= %s;
    '''
    cursor.execute(query, (start_date, end_date))
    rows = cursor.fetchall()
    conn.close()
    
    data_frames = []
    
    for filename, binary_data in tqdm(rows, desc="Loading data from DB"):
        temp_dir = tempfile.gettempdir()
        file_path = f"{temp_dir}/{filename}"
        
        with open(file_path, "wb") as f:
            f.write(binary_data)
        
        with h5py.File(file_path, "r") as f:
            timestamps = [t.decode('utf-8') for t in f['L2/Timestamp'][:]]
            timestamps = pd.to_datetime(timestamps, errors='coerce')
            prices = f['L2/Price'][:].astype(float)
            df = pd.DataFrame({"datetime": timestamps, "close": prices})
            df.dropna(subset=["datetime"], inplace=True)
            data_frames.append(df)
    
    combined_df = pd.concat(data_frames)
    combined_df.sort_values(by="datetime", inplace=True)
    
    return combined_df

def animate(ival, df_ticks, renko_chart, ax1, ax2, step, start_date, end_date, my_style):
    start_idx = ival * step
    end_idx = min(start_idx + step, len(df_ticks))

    if start_idx >= len(df_ticks):
        print('No more data to plot')
        return

    for i in range(start_idx, end_idx):
        timestamp = df_ticks['datetime'].iat[i]
        price = df_ticks['close'].iat[i]
        renko_chart.add_prices(timestamp.value // 10**6, price)  # Convert to milliseconds

    df_wicks = renko_chart.renko_animate('wicks', max_len=10000, keep=5000)

    ax1.clear()
    ax2.clear()

    title = f"NQ: {start_date} to {end_date}"
    mpf.plot(df_wicks, type='candle', ax=ax1, volume=ax2, axtitle='6AM to NY Close (10:30PM)', style=my_style)

def main():
    available_dates = fetch_available_dates()
    
    print("Available dates:")
    for date in available_dates:
        print(date)
    
    start_date = input("Enter the start date (YYYYMMDD): ")
    end_date = input("Enter the end date (YYYYMMDD): ")
    
    speeds = {
        '1x': 1,
        '5x': 5,
        '10x': 10,
        '25x': 25,
        '50x': 50,
        '100x': 100,
        '500x': 500,
        '1000x': 1000,
        '5000x': 5000,
        '10000x': 10000,
        '50000x': 50000
    }

    print("Select the speed:")
    for key in speeds:
        print(f"{key}: {speeds[key]}x")

    speed_choice = input("Enter the speed (e.g., 1x, 5x, 10x, etc.): ")

    if speed_choice in speeds:
        step = speeds[speed_choice]
    else:
        print("Invalid choice. Defaulting to 1x speed.")
        step = 1

    df_l2 = load_data_from_db(start_date, end_date)
    initial_timestamp = df_l2['datetime'].iat[0].value // 10**6  # Convert to milliseconds
    initial_price = df_l2['close'].iat[0]

    brick_size = 3  # Adjust based on NinjaTrader settings
    brick_threshold = 5  # Adjust based on NinjaTrader settings

    renko_chart = RenkoWS(initial_timestamp, initial_price, brick_size=brick_size, brick_threshold=brick_threshold)

    # Define custom style
    my_style = mpf.make_mpf_style(base_mpf_style='charles', 
                                  marketcolors=mpf.make_marketcolors(up='g', down='r', inherit=True))

    fig, axes = mpf.plot(renko_chart.initial_df, returnfig=True, volume=True,
                         figsize=(16, 9), panel_ratios=(2, 1),
                         title=f"NQ: {start_date} to {end_date}", type='candle', style=my_style)
    ax1 = axes[0]
    ax2 = axes[2]

    ani = animation.FuncAnimation(fig, animate, fargs=(df_l2, renko_chart, ax1, ax2, step, start_date, end_date, my_style), interval=1)
    mpf.show()

if __name__ == "__main__":
    main()
