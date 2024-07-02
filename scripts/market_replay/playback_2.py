import pandas as pd
import numpy as np
import mplfinance as mpf
from matplotlib import animation
import h5py

# Assuming renkodf.py is in the same directory
from renkodf import RenkoWS

def load_data(hdf5_file):
    with h5py.File(hdf5_file, 'r') as f:
        dataset_name = "L2/Timestamp"
        timestamps = f[dataset_name][:]
        prices = f["L2/Price"][:]
        volumes = f["L2/Volume"][:]

    # Convert byte strings to normal strings before converting to datetime
    timestamps = [ts.decode('utf-8') for ts in timestamps]
    timestamps = pd.to_datetime(timestamps, errors='coerce')

    data = {
        "datetime": timestamps,
        "close": prices.astype(float)
    }
    df_l2 = pd.DataFrame(data)

    # Drop rows with invalid timestamps
    df_l2.dropna(subset=["datetime"], inplace=True)

    # Filter data for the specified time range (6 AM to 10:30 PM GMT+1)
    start_time = pd.to_datetime('06:00:00').time()
    end_time = pd.to_datetime('22:30:00').time()
    df_l2 = df_l2[(df_l2['datetime'].dt.time >= start_time) & (df_l2['datetime'].dt.time <= end_time)]

    return df_l2

def animate(ival, df_ticks, renko_chart, ax1, ax2, step):
    start_idx = ival * step
    end_idx = min(start_idx + step, len(df_ticks))

    if start_idx >= len(df_ticks):
        print('No more data to plot')
        return

    for i in range(start_idx, end_idx):
        timestamp = df_ticks['datetime'].iat[i]
        price = df_ticks['close'].iat[i]
        renko_chart.add_prices(timestamp.value // 10**6, price)  # Convert to milliseconds

    df_wicks = renko_chart.renko_animate('normal', max_len=10000, keep=5000)

    ax1.clear()
    ax2.clear()

    mpf.plot(df_wicks, type='candle', ax=ax1, volume=ax2, axtitle='renko: wicks')

def main():
    hdf5_file = 'C:/Users/Administrator/Desktop/nocodeML-streamlit-app/scripts/market_replay/data/market_replay_data.h5'
    brick_size = 3  # Adjust based on NinjaTrader settings
    brick_threshold = 5  # Adjust based on NinjaTrader settings

    df_l2 = load_data(hdf5_file)
    initial_timestamp = df_l2['datetime'].iat[0].value // 10**6  # Convert to milliseconds
    initial_price = df_l2['close'].iat[0]

    renko_chart = RenkoWS(initial_timestamp, initial_price, brick_size=brick_size, brick_threshold=brick_threshold)

    fig, axes = mpf.plot(renko_chart.initial_df, returnfig=True, volume=True,
                         figsize=(11, 8), panel_ratios=(2, 1),
                         title='\nRenko Animation', type='candle', style='charles')
    ax1 = axes[0]
    ax2 = axes[2]

    # Get the speed from the user
    speeds = {
        '1x': 1,
        '5x': 5,
        '10x': 10,
        '25x': 25,
        '50x': 50,
        '100x': 100,
        '500x': 500,
        '1000x': 1000,
        '5000x': 5000
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

    ani = animation.FuncAnimation(fig, animate, fargs=(df_l2, renko_chart, ax1, ax2, step), interval=1)
    mpf.show()

if __name__ == "__main__":
    main()
