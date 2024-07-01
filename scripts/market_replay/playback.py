import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def generate_renko(df, brick_size, brick_threshold):
    df['Renko'] = np.nan
    df['Direction'] = 0
    df['Renko_Close'] = 0

    previous_close = df['Price'].iloc[0]
    renko_close = previous_close
    direction = 0  # 1 for up, -1 for down

    for i in range(1, len(df)):
        price = df['Price'].iloc[i]
        if direction == 0:
            if price >= previous_close + brick_size * brick_threshold:
                direction = 1
                renko_close = previous_close + brick_size
                df.loc[i, 'Renko'] = renko_close
                df.loc[i, 'Direction'] = direction
                df.loc[i, 'Renko_Close'] = renko_close
            elif price <= previous_close - brick_size * brick_threshold:
                direction = -1
                renko_close = previous_close - brick_size
                df.loc[i, 'Renko'] = renko_close
                df.loc[i, 'Direction'] = direction
                df.loc[i, 'Renko_Close'] = renko_close
        elif direction == 1:
            while price >= renko_close + brick_size:
                renko_close += brick_size
                df.loc[i, 'Renko'] = renko_close
                df.loc[i, 'Renko_Close'] = renko_close
            if price <= renko_close - brick_size * brick_threshold:
                direction = -1
                renko_close -= brick_size
                df.loc[i, 'Renko'] = renko_close
                df.loc[i, 'Direction'] = direction
                df.loc[i, 'Renko_Close'] = renko_close
        elif direction == -1:
            while price <= renko_close - brick_size:
                renko_close -= brick_size
                df.loc[i, 'Renko'] = renko_close
                df.loc[i, 'Renko_Close'] = renko_close
            if price >= renko_close + brick_size * brick_threshold:
                direction = 1
                renko_close += brick_size
                df.loc[i, 'Renko'] = renko_close
                df.loc[i, 'Direction'] = direction
                df.loc[i, 'Renko_Close'] = renko_close
        previous_close = price

    return df.dropna(subset=['Renko'])

# Load data from HDF5
hdf5_file = 'C:/Users/Administrator/Desktop/nocodeML-streamlit-app/scripts/market_replay/data/market_replay_data.h5'
with h5py.File(hdf5_file, 'r') as f:
    l2_price = f['L2/Price'][:]
    l2_timestamp = f['L2/Timestamp'][:]

# Convert to DataFrame
df_l2 = pd.DataFrame({
    'Timestamp': pd.to_datetime(l2_timestamp.astype(str).astype(str)),
    'Price': l2_price.astype(float)
})

# Debug: Print first few rows of the DataFrame
print(df_l2.head(20))

# Generate Renko bars
brick_size = 30  # Define the brick size for Renko bars
brick_threshold = 5  # Define the brick threshold for Renko bars
renko_df = generate_renko(df_l2, brick_size, brick_threshold)

# Debug: Print first few rows of the Renko DataFrame
print(renko_df.head(20))

# Save Renko data to HDF5
with h5py.File(hdf5_file, 'a') as f:
    if 'Renko' in f:
        del f['Renko']  # Remove existing Renko group if it exists
    renko_group = f.create_group('Renko')
    renko_group.create_dataset('Timestamp', data=renko_df['Timestamp'].astype(str).values.astype('S'))
    renko_group.create_dataset('Renko_Close', data=renko_df['Renko_Close'].values)

print("Renko bars generated and saved to HDF5")

def playback_renko(renko_df, speed=1):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    timestamps = renko_df['Timestamp']
    renko_closes = renko_df['Renko_Close']

    l, = plt.plot_date(timestamps[:1], renko_closes[:1], '-')

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Time', 0, len(timestamps) - 1, valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        l.set_data(timestamps[:idx+1], renko_closes[:idx+1])
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

# Load Renko data from HDF5
with h5py.File(hdf5_file, 'r') as f:
    renko_timestamp = f['Renko/Timestamp'][:]
    renko_close = f['Renko/Renko_Close'][:]

# Convert to DataFrame
renko_df = pd.DataFrame({
    'Timestamp': pd.to_datetime(renko_timestamp.astype(str)),
    'Renko_Close': renko_close
})

# Debug: Print first few rows of the Renko DataFrame for playback
print(renko_df.head(20))

# Run the playback
playback_renko(renko_df)
