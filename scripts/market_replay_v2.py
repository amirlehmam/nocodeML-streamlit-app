import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

class Renko:
    def __init__(self, df=None, filename=None):
        if filename:
            try:
                df = pd.read_csv(filename, delimiter=';', header=None, engine='python', on_bad_lines='skip')
                print("Raw Data:")
                print(df.head(10))

                # Define the base columns for L1 and L2
                l1_columns = ['Type', 'MarketDataType', 'Timestamp', 'Offset', 'Price', 'Volume', 'NA1', 'NA2', 'NA3']
                l2_columns = ['Type', 'MarketDataType', 'Timestamp', 'Offset', 'Operation', 'OrderBookPosition', 'MarketMaker', 'Price', 'Volume']

                # Separate L1 and L2 records
                l1_records = df[df[0] == 'L1'].copy()
                l2_records = df[df[0] == 'L2'].copy()

                # Assign column names and parse timestamps
                l1_records.columns = l1_columns
                l2_records.columns = l2_columns
                l1_records['date'] = pd.to_datetime(l1_records['Timestamp'], format='%Y%m%d%H%M%S')
                l2_records['date'] = pd.to_datetime(l2_records['Timestamp'], format='%Y%m%d%H%M%S')

                # Ensure unique indices before combining
                l1_records = l1_records.reset_index(drop=True)
                l2_records = l2_records.reset_index(drop=True)

                # Combine L1 and L2 records into a single DataFrame
                df_combined = pd.concat([l1_records, l2_records]).reset_index(drop=True)

                # Initialize lists to collect valid price values
                price_values = []
                timestamps = []

                # Function to check and add valid prices to the list
                def check_and_add_price(row):
                    price_str = str(row['Price']).replace(',', '.')
                    try:
                        price = float(price_str)
                        if 16000 <= price <= 20000:  # Adjust the range as necessary
                            price_values.append(price)
                            timestamps.append(row['date'])
                    except ValueError:
                        pass

                # Iterate through the rows and apply the function
                df_combined.apply(check_and_add_price, axis=1)

                # Create a DataFrame from the collected valid price values
                df_filtered = pd.DataFrame({'Price': price_values, 'date': timestamps})

                # Debugging statement
                print("Filtered Data with Prices:")
                print(df_filtered.head(10))
                print(f"Number of valid price entries: {len(df_filtered)}")

            except FileNotFoundError:
                raise FileNotFoundError(f"{filename}\n\nDoes not exist.")
        elif df is None:
            raise ValueError("DataFrame or filename must be provided.")

        self.df = df_filtered
        self.close = df_filtered['Price'].values
        self.timestamps = df_filtered['date'].values

    def set_brick_size(self, brick_size=30, brick_threshold=5):
        """ Setting brick size and threshold """
        self.brick_size = brick_size
        self.brick_threshold = brick_threshold
        return self.brick_size, self.brick_threshold

    def build_renko(self):
        """ Create Renko data """
        if self.df.empty:
            raise ValueError("DataFrame is empty after filtering. Check the filtering conditions.")
        
        self.renko = {'index': [], 'date': [], 'price': [], 'direction': []}
        
        initial_price = self.close[0]
        self.renko['index'].append(0)
        self.renko['date'].append(self.timestamps[0])
        self.renko['price'].append(initial_price)
        self.renko['direction'].append(0)

        last_price = initial_price

        for i in range(1, len(self.close)):
            current_price = self.close[i]
            gap = current_price - last_price
            direction = np.sign(gap)

            while abs(gap) >= self.brick_size:
                last_price += self.brick_size * direction
                self.renko['index'].append(i)
                self.renko['date'].append(self.timestamps[i])
                self.renko['price'].append(last_price)
                self.renko['direction'].append(direction)
                gap = current_price - last_price
                print(f"Timestamp: {self.timestamps[i]}, Price: {current_price}, Last Renko Price: {last_price}, Gap: {gap}, Direction: {direction}")

        print("Final Renko Data:")
        print(self.renko)
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

        x = 0
        for i, (price, direction, date) in enumerate(zip(prices, directions, dates)):
            if direction == 1:
                facecolor = 'g'
                y = price - brick_size
            else:
                facecolor = 'r'
                y = price

            ax.add_patch(patches.Rectangle((x + 1, y), height=brick_size, width=1, facecolor=facecolor))
            
            x += 1
            if x >= x_max:  # Extend x-axis limit dynamically
                x_max += 50
                ax.set_xlim(x_min, x_max)

            if price < y_min + 2 * brick_size:
                y_min = price - 2 * brick_size
                ax.set_ylim(y_min, y_max)
            
            if price > y_max - 2 * brick_size:
                y_max = price + 2 * brick_size
                ax.set_ylim(y_min, y_max)

            plt.pause(speed)

            # Add dynamic timestamp labels
            x_ticks = ax.get_xticks()
            x_labels = [dates[int(tick)-1] if 0 <= int(tick)-1 < len(dates) else '' for tick in x_ticks]
            ax.set_xticklabels(x_labels, rotation=45, ha='right')

        plt.show(block=True)  # Ensure the plot window stays open


# Usage example
if __name__ == "__main__":
    filename = "C:/Users/Administrator/Documents/NinjaTrader 8/db/replay/temp_preprocessed/20240509.csv"  # Update with the correct path to your CSV file
    renko_chart = Renko(filename=filename)
    renko_chart.set_brick_size(brick_size=30, brick_threshold=5)
    renko_data = renko_chart.build_renko()
    renko_chart.plot(speed=0.1)
