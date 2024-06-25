import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Function to load and preprocess the CSV file
def load_and_preprocess_csv(filepath):
    df = pd.read_csv(filepath, delimiter=';', header=None, engine='python', on_bad_lines='skip')

    base_columns = ['Type', 'MarketDataType', 'Timestamp', 'Offset', 'Operation', 'OrderBookPosition', 'MarketMaker', 'Price', 'Volume']
    extra_columns = [f'Extra{i}' for i in range(len(df.columns) - len(base_columns))]
    df.columns = base_columns + extra_columns

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y%m%d%H%M%S')

    for column in df.columns:
        if column not in base_columns:
            df[column] = df[column].astype(str).str.replace(',', '.').astype(float, errors='coerce')
    
    return df

# Function to filter valid price data
def filter_valid_prices(df):
    valid_prices = pd.DataFrame()

    for column in df.columns:
        if column not in ['Type', 'MarketDataType', 'Timestamp', 'Offset', 'Operation', 'OrderBookPosition', 'MarketMaker', 'Volume']:
            df[column] = pd.to_numeric(df[column], errors='coerce')
            valid_price_indices = (df[column] >= 18250) & (df[column] <= 18650)  # Adjusted price range based on actual data
            if valid_price_indices.any():
                temp_df = df[valid_price_indices].copy()
                temp_df['Price'] = df[column][valid_price_indices]
                valid_prices = pd.concat([valid_prices, temp_df], ignore_index=True)
    
    return valid_prices

# Function to read and display the contents of the HDF5 file
def read_hdf5_file(filepath):
    with h5py.File(filepath, 'r') as f:
        print("Datasets in the HDF5 file:")
        for name in f:
            print(name)

        timestamps = f['L1/timestamp'][:]
        prices = f['L1/price'][:]
        volumes = f['L1/volume'][:]

        dates = pd.to_datetime(timestamps, unit='ns')

        df_hdf5 = pd.DataFrame({
            'Timestamp': dates,
            'Price': prices,
            'Volume': volumes
        })

        return df_hdf5

# Renko chart class
class Renko:
    def __init__(self, hdf5_filepath):
        try:
            with h5py.File(hdf5_filepath, 'r') as f:
                timestamps = f['L1/timestamp'][:]
                prices = f['L1/price'][:]
                volumes = f['L1/volume'][:]

                dates = pd.to_datetime(timestamps, unit='ns', errors='coerce')

                valid_indices = ~dates.isna()
                dates = dates[valid_indices]
                prices = prices[valid_indices]
                volumes = volumes[valid_indices]

                valid_price_indices = (prices >= 18250) & (prices <= 18650)
                prices = prices[valid_price_indices]
                dates = dates[valid_price_indices]

                df = pd.DataFrame({'Price': prices, 'date': dates}).dropna()

                print("Filtered Data with Prices:")
                print(df.head(10))

                self.df = df
                self.close = df['Price'].values

        except FileNotFoundError:
            raise FileNotFoundError(f"{hdf5_filepath}\n\nDoes not exist.")
        
    def set_brick_size(self, brick_size=30, brick_threshold=5):
        self.brick_size = brick_size
        self.brick_threshold = brick_threshold
        return self.brick_size

    def _apply_renko(self, i):
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
        renko_price = self.renko['price'][-1] + (direction * brick_multiplier * self.brick_size)
        self.renko['index'].append(i)
        self.renko['price'].append(renko_price)
        self.renko['direction'].append(direction)
        self.renko['date'].append(self.df['date'].iat[i])

    def build(self):
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

        print(f"Prices: {prices}")
        print(f"Directions: {directions}")
        print(f"Brick size: {brick_size}")

        fig, ax = plt.subplots(1, figsize=(10, 5))
        fig.suptitle(f"Renko Chart (brick size = {round(brick_size, 2)})", fontsize=20)
        ax.set_ylabel("Price ($)")
        plt.rc('axes', labelsize=20)
        plt.rc('font', size=16)

        x_min = 0
        x_max = 50
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
            
            if x + 1 >= x_max:
                x_max += 50
                ax.set_xlim(x_min, x_max)

            if price < y_min + 2 * brick_size:
                y_min = price - 2 * brick_size
                ax.set_ylim(y_min, y_max)
            
            if price > y_max - 2 * brick_size:
                y_max = price + 2 * brick_size
                ax.set_ylim(y_min, y_max)

            plt.pause(speed)

        x_ticks = ax.get_xticks()
        x_labels = [dates[int(tick)-1] if 0 <= int(tick)-1 < len(dates) else '' for tick in x_ticks]
        ax.set_xticklabels(x_labels, rotation=45, ha='right')

        plt.show(block=True)

# Main code
if __name__ == "__main__":
    csv_filepath = 'C:/Users/Administrator/Documents/NinjaTrader 8/db/replay/temp_preprocessed/20240301.csv'
    hdf5_filepath = 'C:/Users/Administrator/Documents/NinjaTrader 8/db/replay/temp_preprocessed/20240301_fixed.h5'

    df_csv = load_and_preprocess_csv(csv_filepath)
    valid_prices = filter_valid_prices(df_csv)

    with h5py.File(hdf5_filepath, 'w') as f:
        f.create_dataset('L1/timestamp', data=valid_prices['Timestamp'].astype('int64').values)
        f.create_dataset('L1/price', data=valid_prices['Price'].values)
        if 'Volume' in valid_prices:
            f.create_dataset('L1/volume', data=valid_prices['Volume'].fillna(0).values)

    print("Data saved to HDF5.")
    df_hdf5 = read_hdf5_file(hdf5_filepath)
    print(df_hdf5.head(10))

    renko_chart = Renko(hdf5_filepath)
    renko_chart.set_brick_size(brick_size=30, brick_threshold=5)
    renko_data = renko_chart.build()
    renko_chart.plot(speed=0.1)
