import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from tqdm import tqdm

# Step 1: Read the CSV file and identify the price column dynamically
csv_filepath = 'C:/Users/Administrator/Documents/NinjaTrader 8/db/replay/temp_preprocessed/20240301.csv'

# Load the CSV file
df_csv = pd.read_csv(csv_filepath, delimiter=';', header=None, engine='python', on_bad_lines='skip')

# Define the base columns
base_columns = ['Type', 'MarketDataType', 'Timestamp', 'Offset', 'Operation', 'OrderBookPosition', 'MarketMaker', 'Price', 'Volume']
extra_columns = [f'Extra{i}' for i in range(len(df_csv.columns) - len(base_columns))]
df_csv.columns = base_columns + extra_columns

# Parse timestamps
df_csv['Timestamp'] = pd.to_datetime(df_csv['Timestamp'], format='%Y%m%d%H%M%S')

# Identify and filter valid price data
valid_prices = pd.DataFrame()

# Debugging: Print the first few rows of the DataFrame to inspect its structure
print("Initial DataFrame structure:")
print(df_csv.head(10))
print(df_csv.columns)

# Define a function to clean and convert potential price columns
def clean_and_convert(column):
    df_csv[column] = df_csv[column].astype(str).str.replace(',', '.')
    return pd.to_numeric(df_csv[column], errors='coerce')

# Iterate over all columns to find potential price data columns
for column in df_csv.columns:
    if column not in base_columns:
        price_data = clean_and_convert(column)
        valid_price_indices = (price_data >= 17000) & (price_data <= 19000)
        if valid_price_indices.any():
            temp_df = df_csv[valid_price_indices].copy()
            temp_df['Price'] = price_data[valid_price_indices]
            valid_prices = pd.concat([valid_prices, temp_df])

# Check if 'Operation' column contains price data for 'L1' records
operation_as_price = clean_and_convert('Operation')
valid_operation_indices = (operation_as_price >= 17000) & (operation_as_price <= 19000) & (df_csv['Type'] == 'L1')

if valid_operation_indices.any():
    temp_df = df_csv[valid_operation_indices].copy()
    temp_df['Price'] = operation_as_price[valid_operation_indices]
    valid_prices = pd.concat([valid_prices, temp_df])

# Ensure 'Timestamp' column is retained in valid_prices
if not valid_prices.empty:
    valid_prices['Timestamp'] = df_csv.loc[valid_prices.index, 'Timestamp']

# Debugging statements to show the structure of valid_prices DataFrame
print("valid_prices DataFrame structure:")
print(valid_prices.head(10))
print(valid_prices.columns)

# Step 2: Convert the CSV data to HDF5 format
hdf5_filepath = 'C:/Users/Administrator/Documents/NinjaTrader 8/db/replay/temp_preprocessed/20240301_fixed.h5'
with h5py.File(hdf5_filepath, 'w') as f:
    if 'Price' in valid_prices:
        f.create_dataset('L1/timestamp', data=valid_prices['Timestamp'].astype('int64').values)
        f.create_dataset('L1/price', data=valid_prices['Price'].values)
        if 'Volume' in valid_prices:
            f.create_dataset('L1/volume', data=valid_prices['Volume'].fillna(0).values)  # Handle missing volume data
    else:
        print("Error: 'Price' column not found in valid_prices DataFrame.")

# Debugging statement
print("Data saved to HDF5.")

# Step 3: Read and filter the HDF5 data, and Step 4: Generate the Renko chart
class Renko:
    def __init__(self, hdf5_filepath):
        try:
            with h5py.File(hdf5_filepath, 'r') as f:
                timestamps = f['L1/timestamp'][:]
                prices = f['L1/price'][:]
                volumes = f['L1/volume'][:]

                # Convert timestamps to datetime
                dates = pd.to_datetime(timestamps, unit='ns', errors='coerce')

                # Filter out invalid timestamps
                valid_indices = ~dates.isna()
                dates = dates[valid_indices]
                prices = prices[valid_indices]
                volumes = volumes[valid_indices]

                # Filter prices within a realistic range (17k to 19k in this case)
                valid_price_indices = (prices >= 17000) & (prices <= 19000)
                prices = prices[valid_price_indices]
                dates = dates[valid_price_indices]

                # Create a DataFrame and drop any rows with NaT (Not a Time) in the date column
                df = pd.DataFrame({'Price': prices, 'date': dates}).dropna()

                # Debugging statement
                print("Filtered Data with Prices:")
                print(df.head(10))

                self.df = df
                self.close = df['Price'].values

        except FileNotFoundError:
            raise FileNotFoundError(f"{hdf5_filepath}\n\nDoes not exist.")
        
    def set_brick_size(self, brick_size=30, brick_threshold=5):
        """ Setting brick size """
        self.brick_size = brick_size
        self.brick_threshold = brick_threshold
        return self.brick_size

    def _apply_renko(self, i):
        """ Determine if there are any new bricks to paint with current price """
        num_bricks = 0
        gap = (self.close[i] - self.renko['price'][-1]) // self.brick_size
        direction = np.sign(gap)
        if direction == 0:
            return
        if (gap > 0 and self.renko['direction'][-1] >= 0) or (gap < 0 and self.renko['direction'][-1] <= 0):
            num_bricks = gap
        elif np.abs(gap) >= self.brick_threshold:
            num_bricks = gap - self.brick_threshold * direction
            self._update_renko(i, direction, self.brick_threshold)

        for brick in range(abs(int(num_bricks))):
            self._update_renko(i, direction)

    def _update_renko(self, i, direction, brick_multiplier=1):
        """ Append price and new block to renko dict """
        renko_price = self.renko['price'][-1] + (direction * brick_multiplier * self.brick_size)
        self.renko['index'].append(i)
        self.renko['price'].append(renko_price)
        self.renko['direction'].append(direction)
        self.renko['date'].append(self.df['date'].iat[i])

    def build(self):
        """ Create Renko data """
        if self.df.empty:
            raise ValueError("DataFrame is empty after filtering. Check the filtering conditions.")
        
        units = self.df['Price'].iat[0] // self.brick_size
        start_price = units * self.brick_size

        self.renko = {'index': [0], 'date': [self.df['date'].iat[0]], 'price': [start_price], 'direction': [0]}
        for i in range(1, len(self.close)):
            self._apply_renko(i)
        return self.renko

    def plot(self, speed=0.1):
        prices = self.renko['price']
        directions = self.renko['direction']
        dates = self.renko['date']
        brick_size = self.brick_size

        # Debugging output
        print(f"Prices: {prices}")
        print(f"Directions: {directions}")
        print(f"Brick size: {brick_size}")

        fig, ax = plt.subplots(1, figsize=(10, 5))
        fig.suptitle(f"Renko Chart (brick size = {round(brick_size, 2)})", fontsize=20)
        ax.set_ylabel("Price ($)")
        plt.rc('axes', labelsize=20)
        plt.rc('font', size=16)

        x_min = 0
        x_max = 50  # Initial x-axis limit
        y_min = min(prices) - 2 * brick_size
        y_max = max(prices) + 2 * brick_size

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        for x, (price, direction, date) in enumerate(zip(prices, directions, dates)):
            if direction == 1:
                facecolor = 'g'
                y = price - brick_size
            else:
                facecolor = 'r'
                y = price

            ax.add_patch(patches.Rectangle((x + 1, y), height=brick_size, width=1, facecolor=facecolor))
            
            if x + 1 >= x_max:  # Extend x-axis limit dynamically
                x_max += 50
                ax.set_xlim(x_min, x_max)

            if price < y_min + 2 * brick_size:
                y_min = price - 2 * brick_size
                ax.set_ylim(y_min, y_max)
            
            if price > y_max - 2 * brick_size:
                y_max = price + 2 * brick_size
                ax.set_ylim(y_min, y_max)

            plt.pause(speed)

        # Convert x-ticks to dates
        x_ticks = ax.get_xticks()
        x_labels = [dates[int(tick)-1] if 0 <= int(tick)-1 < len(dates) else '' for tick in x_ticks]
        ax.set_xticklabels(x_labels, rotation=45, ha='right')

        plt.show(block=True)  # Ensure the plot window stays open

# Usage example
if __name__ == "__main__":
    hdf5_filepath = 'C:/Users/Administrator/Documents/NinjaTrader 8/db/replay/temp_preprocessed/20240301_fixed.h5'
    renko_chart = Renko(hdf5_filepath)
    renko_chart.set_brick_size(brick_size=30, brick_threshold=5)
    renko_data = renko_chart.build()
    renko_chart.plot(speed=0.1)
