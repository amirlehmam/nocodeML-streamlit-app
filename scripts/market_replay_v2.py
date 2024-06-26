import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Renko:
    def __init__(self, filename, brick_size, threshold):
        self.filename = filename
        self.brick_size = brick_size
        self.threshold = threshold
        self.df = self.read_csv_with_debug()
        self.renko_data = self.calculate_renko()

    def read_csv_with_debug(self):
        try:
            df = pd.read_csv(self.filename, delimiter=';', header=None, engine='python')
            print(f"CSV file read successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
            return df
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return pd.DataFrame()

    def preprocess_data(self, df):
        print("Preprocessing data...")
        l1_records = df[df[0] == 'L1'].copy()
        l2_records = df[df[0] == 'L2'].copy()

        print(f"L1 records: {len(l1_records)}, L2 records: {len(l2_records)}")

        # Convert relevant columns to strings before processing
        l1_records[4] = l1_records[4].astype(str)
        l2_records[8] = l2_records[8].astype(str)

        # Process L1 records
        l1_records['timestamp'] = pd.to_datetime(l1_records[2], format='%Y%m%d%H%M%S')
        l1_records['price'] = l1_records[4].str.replace(',', '.').astype(float)
        l1_records['volume'] = l1_records[5].astype(float)
        l1_records = l1_records[['timestamp', 'price', 'volume']]

        # Process L2 records
        print("L2 records columns:", l2_records.columns)
        l2_records['timestamp'] = pd.to_datetime(l2_records[2], format='%Y%m%d%H%M%S')
        l2_records['price'] = l2_records[8].str.replace(',', '.').astype(float)

        # Check if column 9 exists in l2_records
        if 9 in l2_records.columns:
            l2_records['volume'] = l2_records[9].astype(float)
        else:
            l2_records['volume'] = np.nan  # or handle as needed

        l2_records = l2_records[['timestamp', 'price', 'volume']]

        combined_df = pd.concat([l1_records, l2_records])
        combined_df.sort_values(by='timestamp', inplace=True)
        combined_df.reset_index(drop=True, inplace=True)

        print(f"Combined data prepared with {combined_df.shape[0]} rows.")
        print(combined_df.head())  # Debugging line to show first few rows
        return combined_df

    def calculate_renko(self):
        data = self.preprocess_data(self.df)
        print("Starting Renko calculation...")
        renko_prices = []
        renko_dates = []
        directions = []

        last_renko_price = None
        for i in range(len(data)):
            current_price = data['price'].iloc[i]
            current_date = data['timestamp'].iloc[i]

            if last_renko_price is None:
                last_renko_price = current_price
                renko_prices.append(current_price)
                renko_dates.append(current_date)
                directions.append(0)
                continue

            price_diff = current_price - last_renko_price
            print(f"Current price: {current_price}, Last Renko price: {last_renko_price}, Price difference: {price_diff}")

            if abs(price_diff) >= self.brick_size:
                num_bricks = int(price_diff / self.brick_size)
                for _ in range(abs(num_bricks)):
                    last_renko_price += self.brick_size * np.sign(price_diff)
                    renko_prices.append(last_renko_price)
                    renko_dates.append(current_date)
                    directions.append(np.sign(price_diff))

        if not renko_prices:
            print("No valid data to plot Renko chart.")
            return pd.DataFrame()

        print(f"Renko calculation completed with {len(renko_prices)} bars.")
        return pd.DataFrame({'date': renko_dates, 'price': renko_prices, 'direction': directions})

    def plot_renko(self):
        if self.renko_data.empty:
            print("No data to plot.")
            return

        plt.figure(figsize=(12, 6))
        for i in range(1, len(self.renko_data)):
            if self.renko_data['direction'][i] == 1:
                plt.plot([self.renko_data['date'][i-1], self.renko_data['date'][i]],
                         [self.renko_data['price'][i-1], self.renko_data['price'][i]], color='green')
            else:
                plt.plot([self.renko_data['date'][i-1], self.renko_data['date'][i]],
                         [self.renko_data['price'][i-1], self.renko_data['price'][i]], color='red')
        plt.title(f"Renko Chart (brick size = {self.brick_size})")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.show()


filename = "C:/Users/Administrator/Documents/NinjaTrader 8/db/replay/temp_preprocessed/20240509.csv"
renko_chart = Renko(filename=filename, brick_size=30, threshold=5)
renko_chart.plot_renko()
