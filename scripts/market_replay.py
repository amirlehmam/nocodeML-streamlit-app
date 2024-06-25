import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_and_preprocess_csv(filepath):
    df = pd.read_csv(filepath, delimiter=';', header=None, engine='python', on_bad_lines='skip')

    # Separate L1 and L2 records
    df_L1 = df[df[0] == 'L1'].copy()
    df_L2 = df[df[0] == 'L2'].copy()

    # Define column names for L1 and L2 based on the provided format
    L1_columns = ['Type', 'MarketDataType', 'Timestamp', 'Offset', 'Price', 'Volume']
    L2_columns = ['Type', 'MarketDataType', 'Timestamp', 'Offset', 'Operation', 'OrderBookPosition', 'MarketMaker', 'Price', 'Volume']
    
    df_L1.columns = L1_columns + ['Extra'] * (len(df_L1.columns) - len(L1_columns))
    df_L2.columns = L2_columns + ['Extra'] * (len(df_L2.columns) - len(L2_columns))

    # Parse timestamps for L1
    df_L1['Timestamp'] = pd.to_datetime(df_L1['Timestamp'], format='%Y%m%d%H%M%S')

    # Convert price and volume columns to float
    df_L1['Price'] = pd.to_numeric(df_L1['Price'].astype(str).str.replace(',', '.'), errors='coerce')
    df_L1['Volume'] = pd.to_numeric(df_L1['Volume'].astype(str).str.replace(',', '.'), errors='coerce')

    return df_L1, df_L2

def filter_valid_prices(df):
    valid_prices = df[(df['Price'] >= 18250) & (df['Price'] <= 18650)].copy()
    return valid_prices

def save_to_hdf5(df, filepath):
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('L1/timestamp', data=df['Timestamp'].astype('int64').values)
        f.create_dataset('L1/price', data=df['Price'].values)
        if 'Volume' in df:
            f.create_dataset('L1/volume', data=df['Volume'].fillna(0).values)

def read_hdf5_file(filepath):
    with h5py.File(filepath, 'r') as f:
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

if __name__ == "__main__":
    csv_filepath = 'C:/Users/Administrator/Documents/NinjaTrader 8/db/replay/temp_preprocessed/20240301.csv'
    hdf5_filepath = 'C:/Users/Administrator/Documents/NinjaTrader 8/db/replay/temp_preprocessed/20240301_fixed.h5'

    df_L1, df_L2 = load_and_preprocess_csv(csv_filepath)
    valid_prices_L1 = filter_valid_prices(df_L1)

    save_to_hdf5(valid_prices_L1, hdf5_filepath)
    df_hdf5 = read_hdf5_file(hdf5_filepath)
    print(df_hdf5.head(10))

    renko_chart = Renko(hdf5_filepath)
    renko_chart.set_brick_size(brick_size=30, brick_threshold=5)
    renko_data = renko_chart.build()
    renko_chart.plot(speed=0.1)
