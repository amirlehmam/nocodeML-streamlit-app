import pandas as pd
import numpy as np
import mplfinance as mpf
from matplotlib import animation
import h5py
import psycopg2
import tempfile
from tqdm import tqdm
from datetime import datetime, timedelta

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
    result = cursor.fetchone()
    conn.close()
    
    if result is None:
        return None
    
    binary_data = result[0]
    temp_dir = tempfile.gettempdir()
    file_path = f"{temp_dir}/{filename}.h5"
    
    with open(file_path, "wb") as f:
        f.write(binary_data)
    
    return file_path

def process_h5(file_path):
    with h5py.File(file_path, "r") as f:
        timestamps = [t.decode('utf-8') for t in f['L2/Timestamp'][:]]
        timestamps = pd.to_datetime(timestamps, errors='coerce')
        prices = f['L2/Price'][:].astype(float)
        df = pd.DataFrame({"datetime": timestamps, "close": prices})
        df.dropna(subset=["datetime"], inplace=True)
        return df

def load_data_for_date(date):
    filename = date.strftime('%Y%m%d') + ".h5"
    file_path = load_h5_from_db(filename)
    if file_path:
        df = process_h5(file_path)
        return df
    return None

def data_generator(dates, step):
    for date in dates:
        df_ticks = load_data_for_date(date)
        if df_ticks is not None:
            for start_idx in range(0, len(df_ticks), step):
                yield df_ticks.iloc[start_idx:start_idx + step]
        else:
            print(f"No data for {date.strftime('%Y-%m-%d')}")

def animate(df_ticks, renko_chart, ax1, ax2, start_date, end_date, my_style):
    for i in range(len(df_ticks)):
        timestamp = df_ticks['datetime'].iat[i]
        price = df_ticks['close'].iat[i]
        renko_chart.add_prices(timestamp.value // 10**6, price)  # Convert to milliseconds

    df_wicks = renko_chart.renko_animate('wicks', max_len=10000, keep=5000)

    ax1.clear()
    ax2.clear()

    title = f"NQ: {start_date} to {end_date}"
    mpf.plot(df_wicks, type='candle', ax=ax1, volume=ax2, axtitle='6AM to NY Close (10:30PM)', style=my_style)
    print(f"Animating data from {df_ticks['datetime'].iat[0].strftime('%Y-%m-%d')}")

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

    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Initializing Renko chart with first date's data
    df_l2 = load_data_for_date(dates[0])
    if df_l2 is None:
        print(f"No data available for {dates[0].strftime('%Y-%m-%d')}")
        return

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

    data = data_generator(dates, step)
    ani = animation.FuncAnimation(fig, animate, fargs=(renko_chart, ax1, ax2, start_date, end_date, my_style), frames=data, interval=100, repeat=False)
    
    print("Starting animation")
    mpf.show()
    print("Animation done")

if __name__ == "__main__":
    main()
