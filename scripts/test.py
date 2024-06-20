import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
from numba import jit
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom Functions and Helper Methods
def reverse_array(arr):
    return arr[::-1]

def sum_of_values(arr):
    return np.sum(arr)

def cross_above(series1, series2):
    return (series1.shift(1) < series2.shift(1)) & (series1 > series2)

def cross_below(series1, series2):
    return (series1.shift(1) > series2.shift(1)) & (series1 < series2)

def draw_triangle_up(data, index, color):
    plt.plot(index, data[index], marker='^', color=color, markersize=10)

def draw_triangle_down(data, index, color):
    plt.plot(index, data[index], marker='v', color=color, markersize=10)

class Renko:
    def __init__(self, df=None, filename=None, interval=None):
        if filename:
            try:
                df = pd.read_csv(filename, delimiter=';', header=None, engine='python', on_bad_lines='skip')
                base_columns = ['Type', 'MarketDataType', 'Timestamp', 'Offset', 'Operation', 'OrderBookPosition', 'MarketMaker', 'Price', 'Volume']
                extra_columns = [f'Extra{i}' for i in range(len(df.columns) - len(base_columns))]
                df.columns = base_columns + extra_columns
                df['date'] = pd.to_datetime(df['Timestamp'], format='%Y%m%d%H%M%S')

                price_values = []
                timestamps = []

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

                df.apply(check_and_add_price, axis=1)
                df_filtered = pd.DataFrame({'Price': price_values, 'date': timestamps})

            except FileNotFoundError:
                raise FileNotFoundError(f"{filename}\n\nDoes not exist.")
        elif df is None:
            raise ValueError("DataFrame or filename must be provided.")

        self.df = df_filtered
        self.close = df_filtered['Price'].values

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

class Delta2Strategy:
    def __init__(self, data, starting_capital=300000):
        self.data = data
        self.trade_log = []
        self.pnl = 0.0
        self.starting_capital = starting_capital
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss = pd.Series(np.zeros(len(data)), index=data.index)
        self.take_profit = pd.Series(np.zeros(len(data)), index=data.index)
        self.signals = {'buy_signal': np.zeros(len(data)), 'sell_signal': np.zeros(len(data))}
        self._initialize_parameters()
        self._initialize_indicators()
        self._initialize_fibonacci()

    def _initialize_parameters(self):
        self.bars_required_to_trade = 200
        self.default_quantity = 1

        self.enable_fib_weight_ma_cross = True
        self.fib_weight_ma_period = 10
        self.smoothing_simple_ma_period = 20

    def _initialize_indicators(self):
        self.zlema8 = self.calculate_zlema(8)
        self.zlema62 = self.calculate_zlema(62)
        self.atr = self.calculate_atr(14)
        self.psar = self.calculate_psar()

    def _initialize_fibonacci(self):
        self.fib_weightings = np.zeros(self.fib_weight_ma_period)
        self.fib_weighted_ma = np.zeros(len(self.data))
        self.smoothed_fib_weighted_ma = np.zeros(len(self.data))
        self._calculate_fibonacci_weights()

    def _calculate_fibonacci_weights(self):
        a, b = 1, 1
        for i in range(self.fib_weight_ma_period):
            self.fib_weightings[i] = a
            a, b = b, a + b
        self.fib_weightings = reverse_array(self.fib_weightings)
        self.sum_of_fib_weights = sum_of_values(self.fib_weightings)

    def _calculate_fib_weighted_ma(self):
        weights = np.arange(1, self.fib_weight_ma_period + 1)
        self.data['fib_weighted_ma'] = self.data['Price'].rolling(window=self.fib_weight_ma_period).apply(
            lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
        self.data['smoothed_fib_weighted_ma'] = self.data['fib_weighted_ma'].rolling(window=self.smoothing_simple_ma_period).mean()

    def calculate_indicators(self):
        self._calculate_fib_weighted_ma()

    def calculate_zlema(self, period):
        return self.data['Price'].ewm(span=period).mean()

    def calculate_atr(self, period):
        tr = pd.Series(np.maximum.reduce([self.data['High'] - self.data['Low'], 
                                          (self.data['High'] - self.data['Price'].shift()).abs(), 
                                          (self.data['Low'] - self.data['Price'].shift()).abs()]))
        self.data['atr'] = tr.rolling(window=period).mean()
        return self.data['atr']

    def calculate_psar(self):
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Price']

        af = 0.02
        max_af = 0.2
        psar = np.zeros(len(close))
        bull = True
        ep = low[0]
        hp = high[0]
        lp = low[0]

        for i in range(1, len(close)):
            psar[i] = psar[i - 1] + af * (hp if bull else lp - psar[i - 1])
            reverse = False

            if bull:
                if low[i] < psar[i]:
                    bull = False
                    psar[i] = hp
                    lp = low[i]
                    af = 0.02
                    reverse = True
            else:
                if high[i] > psar[i]:
                    bull = True
                    psar[i] = lp
                    hp = high[i]
                    af = 0.02
                    reverse = True

            if not reverse:
                if bull:
                    if high[i] > hp:
                        hp = high[i]
                        af = min(af + 0.02, max_af)
                    if i > 1 and low[i - 1] < psar[i]:
                        psar[i] = low[i - 1]
                    if i > 2 and low[i - 2] < psar[i]:
                        psar[i] = low[i - 2]
                else:
                    if low[i] < lp:
                        lp = low[i]
                        af = min(af + 0.02, max_af)
                    if i > 1 and high[i - 1] > psar[i]:
                        psar[i] = high[i - 1]
                    if i > 2 and high[i - 2] > psar[i]:
                        psar[i] = high[i - 2]

        self.data['PSAR'] = psar
        return psar

    def _bar_update(self, i):
        if i < self.bars_required_to_trade:
            return

        if self.enable_fib_weight_ma_cross:
            self._calculate_fib_weighted_ma()

        self._generate_signals(i)
        self._manage_positions(i)

    def _generate_signals(self, i):
        if self.enable_fib_weight_ma_cross:
            self.signals['buy_signal'][i] = cross_above(self.data['fib_weighted_ma'], self.data['smoothed_fib_weighted_ma'])[i]
            self.signals['sell_signal'][i] = cross_below(self.data['fib_weighted_ma'], self.data['smoothed_fib_weighted_ma'])[i]

    def _manage_positions(self, i):
        if self.position == 0:
            if self.signals['buy_signal'][i]:
                self._enter_position(i, 'long')
            elif self.signals['sell_signal'][i]:
                self._enter_position(i, 'short')
        elif self.position > 0:
            if self.signals['sell_signal'][i] or self.data['Price'][i] <= self.stop_loss[i] or self.data['Price'][i] >= self.take_profit[i]:
                self._exit_position(i)
        elif self.position < 0:
            if self.signals['buy_signal'][i] or self.data['Price'][i] >= self.stop_loss[i] or self.data['Price'][i] <= self.take_profit[i]:
                self._exit_position(i)

    def _enter_position(self, index, direction):
        if direction == 'long':
            self.position = self.default_quantity
            self.entry_price = self.data['Price'][index]
            self.stop_loss[index] = self.data['PSAR'][index]
            self.take_profit[index] = self.entry_price + 2 * self.data['atr'][index]
        elif direction == 'short':
            self.position = -self.default_quantity
            self.entry_price = self.data['Price'][index]
            self.stop_loss[index] = self.data['PSAR'][index]
            self.take_profit[index] = self.entry_price - 2 * self.data['atr'][index]
        self._log_entry(index, direction)

    def _exit_position(self, index):
        if self.position > 0:
            self.pnl += (self.data['Price'][index] - self.entry_price) * self.default_quantity
        elif self.position < 0:
            self.pnl += (self.entry_price - self.data['Price'][index]) * self.default_quantity
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss[index] = 0.0
        self.take_profit[index] = 0.0
        self._log_exit(index)

    def _log_entry(self, index, direction):
        log_entry = {
            'timestamp': self.data.index[index],
            'action': 'enter',
            'direction': direction,
            'price': self.data['Price'][index],
            'quantity': self.default_quantity,
            'position': self.position,
            'pnl': self.pnl
        }
        self.trade_log.append(log_entry)

    def _log_exit(self, index):
        log_exit = {
            'timestamp': self.data.index[index],
            'action': 'exit',
            'price': self.data['Price'][index],
            'quantity': self.default_quantity,
            'position': self.position,
            'pnl': self.pnl
        }
        self.trade_log.append(log_exit)

    def execute_strategy(self):
        logger.info("Executing strategy")
        if self.enable_fib_weight_ma_cross:
            self._calculate_fib_weighted_ma()

        self.signals['buy_signal'] = cross_above(self.data['fib_weighted_ma'], self.data['smoothed_fib_weighted_ma'])
        self.signals['sell_signal'] = cross_below(self.data['fib_weighted_ma'], self.data['smoothed_fib_weighted_ma'])

        self._manage_positions_vectorized()

    def _manage_positions_vectorized(self):
        buy_indices = np.where(self.signals['buy_signal'])[0]
        sell_indices = np.where(self.signals['sell_signal'])[0]

        # Vectorized buy logic
        for index in buy_indices:
            self._enter_position(index, 'long')

        # Vectorized sell logic
        for index in sell_indices:
            self._enter_position(index, 'short')

        # Vectorized exit logic
        for i in range(len(self.data)):
            if self.position > 0:
                if self.signals['sell_signal'][i] or self.data['Price'][i] <= self.stop_loss[i] or self.data['Price'][i] >= self.take_profit[i]:
                    self._exit_position(i)
            elif self.position < 0:
                if self.signals['buy_signal'][i] or self.data['Price'][i] >= self.stop_loss[i] or self.data['Price'][i] <= self.take_profit[i]:
                    self._exit_position(i)


if __name__ == "__main__":
    filename = "C:/Users/Administrator/Documents/NinjaTrader 8/db/replay/temp_preprocessed/20240305.csv"
    renko_chart = Renko(filename=filename)
    renko_chart.set_brick_size(brick_size=30, brick_threshold=5)
    renko_data = renko_chart.build()
    
    strategy = Delta2Strategy(renko_chart.df)
    strategy.execute_strategy()

    for log in strategy.trade_log:
        print(log)
    
    renko_chart.plot(speed=0.1)
