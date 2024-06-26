import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Renko:
    def __init__(self, filename, brick_size, threshold):
        self.df = self.load_and_preprocess_data(filename)
        self.brick_size = brick_size
        self.threshold = threshold
        self.renko_data = self.calculate_renko()

    def load_and_preprocess_data(self, filename):
        try:
            df = pd.read_csv(filename, sep=';', header=None, on_bad_lines='skip')
            df.columns = ['Type', 'MarketDataType', 'Timestamp', 'Offset', 'BidAsk', 'Operation', 'Position', 'Price', 'Volume']
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y%m%d%H%M%S')
            df['Price'] = df['Price'].str.replace(',', '.').astype(float)
            return df
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return pd.DataFrame()

    def calculate_renko(self):
        renko_prices = []
        renko_dates = []
        directions = []
        current_trend = 0  # 1 for up, -1 for down
        last_price = None

        print("Starting Renko calculation...")
        for index, row in self.df.iterrows():
            price = row['Price']
            date = row['Timestamp']
            
            if last_price is None:
                last_price = price
                continue
            
            price_diff = price - last_price
            if abs(price_diff) >= self.brick_size:
                num_bricks = int(abs(price_diff) // self.brick_size)
                direction = np.sign(price_diff)
                
                for _ in range(num_bricks):
                    last_price += direction * self.brick_size
                    renko_prices.append(last_price)
                    renko_dates.append(date)
                    directions.append(direction)
                    print(f"Renko bar added: Date={date}, Price={last_price}, Direction={direction}")

        return pd.DataFrame({'date': renko_dates, 'price': renko_prices, 'direction': directions})

    def plot_renko(self):
        plt.figure(figsize=(10, 6))
        for i in range(1, len(self.renko_data)):
            color = 'green' if self.renko_data['direction'][i] == 1 else 'red'
            plt.plot([self.renko_data['date'][i-1], self.renko_data['date'][i]],
                     [self.renko_data['price'][i-1], self.renko_data['price'][i]],
                     color=color, linewidth=2)
        plt.title(f'Renko Chart (brick size = {self.brick_size})')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    filename = 'C:/Users/Administrator/Documents/NinjaTrader 8/db/replay/temp_preprocessed/20240509.csv'
    renko_chart = Renko(filename=filename, brick_size=30, threshold=5)
    renko_chart.plot_renko()
