import dask.dataframe as dd
import pandas as pd
import numpy as np
from tqdm import tqdm
from numba import jit
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

def load_market_data(file_path):
    print("Loading market data...")
    dtype = {0: 'object', 1: 'object', 2: 'object', 3: 'int64', 4: 'object', 5: 'int64', 6: 'object', 7: 'object', 8: 'object'}
    data = dd.read_csv(file_path, delimiter=';', header=None, dtype=dtype)
    data = data.compute()
    print("Market data loaded.")
    
    l1_data = data[data[0] == 'L1'].copy()
    l2_data = data[data[0] == 'L2'].copy()

    l1_columns = ['Type', 'MarketDataType', 'Timestamp', 'Offset', 'Price', 'Volume']
    l2_columns = ['Type', 'MarketDataType', 'Timestamp', 'Offset', 'Operation', 'Position', 'MarketMaker', 'Price', 'Volume']

    if len(l1_data.columns) == len(l1_columns):
        l1_data.columns = l1_columns
    else:
        l1_data = l1_data.iloc[:, :len(l1_columns)]
        l1_data.columns = l1_columns

    if len(l2_data.columns) == len(l2_columns):
        l2_data.columns = l2_columns
    else:
        l2_data = l2_data.iloc[:, :len(l2_columns)]
        l2_data.columns = l2_columns

    return l1_data, l2_data

@jit(nopython=True)
def calculate_renko_bars_numba(prices, brick_size):
    renko_bars = []
    uptrend = True
    open_price = prices[0]
    high_price = open_price
    low_price = open_price
    close_price = open_price

    for i in range(1, len(prices)):
        price = prices[i]

        if uptrend:
            if price >= close_price + brick_size:
                renko_bars.append((open_price, high_price, low_price, close_price + brick_size, True))
                open_price = close_price + brick_size
                close_price = open_price
                high_price = open_price
                low_price = open_price
            elif price <= close_price - 2 * brick_size:
                uptrend = False
                renko_bars.append((open_price, high_price, low_price, close_price - brick_size, False))
                open_price = close_price - brick_size
                close_price = open_price
                high_price = open_price
                low_price = open_price
            else:
                high_price = max(high_price, price)
                low_price = min(low_price, price)
        else:
            if price <= close_price - brick_size:
                renko_bars.append((open_price, high_price, low_price, close_price - brick_size, False))
                open_price = close_price - brick_size
                close_price = open_price
                high_price = open_price
                low_price = open_price
            elif price >= close_price + 2 * brick_size:
                uptrend = True
                renko_bars.append((open_price, high_price, low_price, close_price + brick_size, True))
                open_price = close_price + brick_size
                close_price = open_price
                high_price = open_price
                low_price = open_price
            else:
                high_price = max(high_price, price)
                low_price = min(low_price, price)

    return renko_bars

def calculate_renko_bars(data, brick_size):
    print("Calculating Renko bars...")
    data['Date'] = pd.to_datetime(data['Timestamp'], format='%Y%m%d%H%M%S')
    data['Price'] = data['Price'].str.replace(',', '.').astype(float)

    prices = data['Price'].values

    renko_bars = calculate_renko_bars_numba(prices, brick_size)

    renko_df = pd.DataFrame(renko_bars, columns=['open', 'high', 'low', 'close', 'uptrend'])
    renko_df['date'] = data['Date'].iloc[:len(renko_df)].values

    if renko_df.empty:
        print("No Renko bars calculated.")
    else:
        print("Renko bars calculated.")
    return renko_df

def plot_renko_ohlc(renko_df, speed):
    print("Plotting Renko OHLC bars...")

    fig, ax = plt.subplots()
    ax.set_title('Renko OHLC Chart')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True)

    def update_chart(frame):
        ax.clear()
        ax.set_title('Renko OHLC Chart')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.grid(True)

        current_data = renko_df.iloc[:frame]
        for i in range(len(current_data)):
            color = 'green' if current_data['uptrend'].iloc[i] else 'red'
            rect = Rectangle((i, current_data['low'].iloc[i]), 
                             0.8, 
                             current_data['high'].iloc[i] - current_data['low'].iloc[i], 
                             color=color, alpha=0.7)
            ax.add_patch(rect)
            ax.plot([i, i], 
                    [current_data['low'].iloc[i], current_data['high'].iloc[i]], 
                    color=color)

        ax.set_xlim(0, len(current_data))
        ax.set_xticks(range(0, len(current_data), max(1, len(current_data)//10)))
        ax.set_xticklabels([current_data['date'].iloc[x].strftime('%Y-%m-%d %H:%M:%S') for x in ax.get_xticks()], rotation=45)

    ani = animation.FuncAnimation(fig, update_chart, frames=range(1, len(renko_df) + 1), interval=speed, repeat=False)
    plt.show()
    print("Renko OHLC bars plotted.")

if __name__ == "__main__":
    print("Script started.")
    l1_data, l2_data = load_market_data('C:/Users/Administrator/Documents/NinjaTrader 8/db/replay/temp_preprocessed/20240301.csv')

    combined_data = pd.concat([l1_data, l2_data])
    combined_data.sort_values(by='Timestamp', inplace=True)

    renko_df = calculate_renko_bars(combined_data, brick_size=30)
    if not renko_df.empty:
        print(renko_df.head())

        plot_renko_ohlc(renko_df, speed=1)  # Adjust the speed as needed
    else:
        print("No Renko bars to plot.")
    print("Script completed.")
