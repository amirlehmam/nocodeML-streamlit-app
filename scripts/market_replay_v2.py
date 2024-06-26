import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Renko:
    def __init__(self, filename, brick_size=30):
        self.filename = filename
        self.brick_size = brick_size
        self.df = pd.read_csv(filename, header=None)
        self.df.columns = ['Type', 'MarketDataType', 'Timestamp', 'Offset', 'Operation', 'OrderBookPosition', 'MarketMaker', 'Price', 'Volume']
        self.df = self.preprocess_data(self.df)
        self.renko_data = self.build_renko(self.df, self.brick_size)

    def preprocess_data(self, df):
        l1_columns = ['Type', 'MarketDataType', 'Timestamp', 'Offset', 'Price', 'Volume']
        l2_columns = ['Type', 'MarketDataType', 'Timestamp', 'Offset', 'Operation', 'OrderBookPosition', 'MarketMaker', 'Price', 'Volume']
        l1_records = df[df['Type'] == 'L1'].copy()
        l2_records = df[df['Type'] == 'L2'].copy()

        l1_records.columns = l2_columns  # Use the l2_columns to make sure the columns match

        l1_records['date'] = pd.to_datetime(l1_records['Timestamp'], format='%Y%m%d%H%M%S')
        l2_records['date'] = pd.to_datetime(l2_records['Timestamp'], format='%Y%m%d%H%M%S')

        df_combined = pd.concat([l1_records, l2_records]).reset_index(drop=True)
        df_combined = df_combined.sort_values(by=['date', 'Offset']).reset_index(drop=True)
        df_combined['Price'] = df_combined['Price'].str.replace(',', '').astype(float)
        return df_combined[['date', 'Price']]

    def build_renko(self, df, brick_size):
        df['diff'] = df['Price'].diff()
        df = df.dropna().reset_index(drop=True)

        renko_data = {
            'index': [],
            'date': [],
            'price': [],
            'direction': []
        }

        last_price = df['Price'][0]
        last_brick_price = last_price
        direction = 0  # 1 for up, -1 for down
        brick_threshold = self.brick_size

        for i in range(1, len(df)):
            price = df['Price'][i]
            diff = price - last_brick_price

            if direction == 0:
                if abs(diff) >= brick_threshold:
                    direction = np.sign(diff)
                    last_brick_price += direction * brick_threshold
                    renko_data['index'].append(i)
                    renko_data['date'].append(df['date'][i])
                    renko_data['price'].append(last_brick_price)
                    renko_data['direction'].append(direction)
            elif direction == 1:
                if diff >= brick_threshold:
                    last_brick_price += brick_threshold
                    renko_data['index'].append(i)
                    renko_data['date'].append(df['date'][i])
                    renko_data['price'].append(last_brick_price)
                    renko_data['direction'].append(direction)
                elif diff <= -brick_threshold:
                    last_brick_price -= brick_threshold
                    direction = -1
                    renko_data['index'].append(i)
                    renko_data['date'].append(df['date'][i])
                    renko_data['price'].append(last_brick_price)
                    renko_data['direction'].append(direction)
            elif direction == -1:
                if diff <= -brick_threshold:
                    last_brick_price -= brick_threshold
                    renko_data['index'].append(i)
                    renko_data['date'].append(df['date'][i])
                    renko_data['price'].append(last_brick_price)
                    renko_data['direction'].append(direction)
                elif diff >= brick_threshold:
                    last_brick_price += brick_threshold
                    direction = 1
                    renko_data['index'].append(i)
                    renko_data['date'].append(df['date'][i])
                    renko_data['price'].append(last_brick_price)
                    renko_data['direction'].append(direction)

        return renko_data

    def plot_renko(self):
        plt.figure(figsize=(10, 5))
        prices = self.renko_data['price']
        dates = self.renko_data['date']
        directions = self.renko_data['direction']

        for i in range(1, len(prices)):
            if directions[i] == 1:
                plt.plot([dates[i-1], dates[i]], [prices[i-1], prices[i]], color='green', linewidth=2)
            else:
                plt.plot([dates[i-1], dates[i]], [prices[i-1], prices[i]], color='red', linewidth=2)

        plt.title(f'Renko Chart (brick size = {self.brick_size})')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.show()

if __name__ == "__main__":
    filename = "C:/Users/Administrator/Documents/NinjaTrader 8/db/replay/temp_preprocessed/20240509.csv"
    renko_chart = Renko(filename=filename, brick_size=30)
    print(f"Number of valid price entries: {len(renko_chart.df)}")
    for i in range(len(renko_chart.renko_data['date'])):
        print(f"Timestamp: {renko_chart.renko_data['date'][i]}, Price: {renko_chart.renko_data['price'][i]}, Direction: {renko_chart.renko_data['direction'][i]}")
    renko_chart.plot_renko()
