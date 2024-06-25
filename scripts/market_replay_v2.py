import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import os
from numba import jit
import time

class Renko:
    def __init__(self, df=None, filename=None, interval=None):
        if filename:
            try:
                df = pd.read_csv(filename, delimiter=';', header=None, engine='python', on_bad_lines='skip')
                print("Raw Data:")
                print(df.head(10))

                # Define the base columns
                base_columns = ['Type', 'MarketDataType', 'Timestamp', 'Offset', 'Operation', 'OrderBookPosition', 'MarketMaker', 'Price', 'Volume']
                extra_columns = [f'Extra{i}' for i in range(len(df.columns) - len(base_columns))]
                df.columns = base_columns + extra_columns

                # Parse timestamps
                df['date'] = pd.to_datetime(df['Timestamp'], format='%Y%m%d%H%M%S')

                # Initialize a list to collect valid price values
                price_values = []
                timestamps = []

                # Function to check and add valid prices to the list
                def check_and_add_price(row):
                    for column in ['Operation', 'Price'] + extra_columns:
                        price_str = row[column]
                        if isinstance(price_str, str):
                            price_str = price_str.replace(',', '.')
                        try:
                            price = float(price_str)
                            if 16000 <= price <= 20000:
                                price_values.append(price)
                                timestamps.append(row['date'])
                        except ValueError:
                            pass

                # Iterate through the rows and apply the function
                df.apply(check_and_add_price, axis=1)

                # Create a DataFrame from the collected valid price values
                df_filtered = pd.DataFrame({'Price': price_values, 'date': timestamps})

                # Debugging statement
                print("Filtered Data with Prices:")
                print(df_filtered.head(10))

            except FileNotFoundError:
                raise FileNotFoundError(f"{filename}\n\nDoes not exist.")
        elif df is None:
            raise ValueError("DataFrame or filename must be provided.")

        self.df = df_filtered
        self.close = df_filtered['Price'].values

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
    filename = "/mnt/data/20240305.csv"  # Update with the correct path to your CSV file
    renko_chart = Renko(filename=filename)
    renko_chart.set_brick_size(brick_size=30, brick_threshold=5)
    renko_data = renko_chart.build()
    renko_chart.plot(speed=0.1)
