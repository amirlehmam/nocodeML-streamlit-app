import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import os
from numba import jit
import time

# Custom Functions and Helper Methods
def reverse_array(arr):
    return arr[::-1]

def sum_of_values(arr):
    return np.sum(arr)

def is_rising(arr):
    return arr[-1] > arr[-2]

def is_falling(arr):
    return arr[-1] < arr[-2]

def cross_above(series1, series2):
    return series1.shift(1) < series2.shift(1) and series1 > series2

def cross_below(series1, series2):
    return series1.shift(1) > series2.shift(1) and series1 < series2

def draw_triangle_up(data, index, color):
    plt.plot(index, data[index], marker='^', color=color, markersize=10)

def draw_triangle_down(data, index, color):
    plt.plot(index, data[index], marker='v', color=color, markersize=10)

class Delta2Strategy:
    def __init__(self, data):
        self.data = data
        self.trade_log = []
        self.pnl = 0.0
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.signals = {'buy_signal': np.zeros(len(data)), 'sell_signal': np.zeros(len(data))}
        self._initialize_parameters()
        self._initialize_indicators()
        self._initialize_fibonacci()

    def _initialize_parameters(self):
        self.description = "VERSION FIB weight MA entry Raw indicator print Real Time Logger"
        self.name = "Δ_2"
        self.calculate_on_bar_close = True
        self.default_quantity = 1
        self.entries_per_direction = 1
        self.entry_handling = 'UniqueEntries'
        self.exit_on_session_close = True
        self.exit_on_session_close_seconds = 3600
        self.include_commission = True
        self.bars_required_to_trade = 200

        self.enable_cci = True
        self.enable_awesome = True
        self.enable_rsi = True
        self.enable_dynamic_momentum = True
        self.sunday = True
        self.monday = True
        self.tuesday = True
        self.wednesday = True
        self.thursday = True
        self.friday = True
        self.enable_time_filter1 = False
        self.start_time1 = '06:30:00'
        self.stop_time1 = '08:30:00'
        self.enable_time_filter2 = False
        self.start_time2 = '09:30:00'
        self.stop_time2 = '11:55:00'
        self.enable_time_filter3 = False
        self.start_time3 = '14:00:00'
        self.stop_time3 = '16:30:00'
        self.enable_long = True
        self.enable_short = True
        self.stay_in_trade_flat_first = False
        self.enable_limit_orders_overrides_market = False
        self.enter_limit_outside_of_signal = False
        self.ticks_away_from_signal = 30
        self.chart_execution_mark = False

        self.qty_entry1 = 1
        self.entry2 = True
        self.qty_entry2 = 1
        self.entry3 = False
        self.qty_entry3 = 1
        self.entry4 = False
        self.qty_entry4 = 1
        self.entry5 = False
        self.qty_entry5 = 1

        self.all_entries_sl = 90.0
        self.tp1 = True
        self.target1 = 88.0
        self.tp2 = True
        self.target2 = 144.0
        self.tp3 = True
        self.target3 = 166.0
        self.tp4 = True
        self.target4 = 233.0
        self.tp5 = True
        self.target5 = 377.0
        self.simulated_stop_fixed = True
        self.simulated_stop_psar = True
        self.psar_acceleration = 0.02
        self.psar_max_acceleration = 0.2
        self.psar_acceleration_step = 0.02
        self.print_trade_log = False
        self.print_indicator_values = False
        self.print_to_csv = False
        self.file_name = "Test"
        self.enable_debug = False
        self.enable_fib_weight_ma_cross = True
        self.show_fib_weight_mas = False
        self.fib_weight_ma_period = 10
        self.smoothing_simple_ma_period = 20
        self.exit_on_fib_counter_cross = False
        self.enable_macd_cross = False
        self.show_macd = False
        self.macd_fast_ema = 12
        self.macd_slow_ema = 26
        self.ema_signal_line = 9
        self.enable_distance_from_vwap = False
        self.enable_distance_from_bolli500_mid = False
        self.show_cp_day = False
        self.show_gann_lines = False

    def _initialize_indicators(self):
        self.zlema8 = self.calculate_zlema(8)
        self.ma_envelopes14 = self.calculate_ma_envelopes(14)
        self.sma1000 = self.calculate_sma(1000)
        self.zlema62 = self.calculate_zlema(62)
        self.ema382 = self.calculate_ema(382)
        self.ema236 = self.calculate_ema(236)
        self.volume = self.data['volume']
        self.disparity_index = self.calculate_disparity_index(25)
        self.atr_filter = self.calculate_atr(14)
        self.macd = self.calculate_macd(self.macd_fast_ema, self.macd_slow_ema, self.ema_signal_line)
        self.adx = self.calculate_adx(14)
        self.vortex = self.calculate_vortex(14)
        self.polarized_fractal_efficiency = self.calculate_pfe(14, 10)
        self.williams_r = self.calculate_williams_r(14)
        self.momentum = self.calculate_momentum(14)
        self.cci = self.calculate_cci(14)
        self.rsi = self.calculate_rsi(14, 3)
        self.trix = self.calculate_trix(14, 3)
        self.chaikin = self.calculate_chaikin(3, 10)
        self.fisher_transform = self.calculate_fisher_transform(10)
        self.zlema233 = self.calculate_zlema(233)
        self.ema618 = self.calculate_ema(618)
        self.sma144 = self.calculate_sma(144)
        self.ma_envelopes3 = self.calculate_ma_envelopes(3)
        self.obv = self.calculate_obv()
        self.rind = self.calculate_rind(3, 10)
        self.bolli500 = self.calculate_bolli(500)
        self.zlema13 = self.calculate_zlema(13)
        self.stoch_fast = self.calculate_stochastics_fast(3, 14)
        self.cmo = self.calculate_cmo(14)
        self.rss = self.calculate_rss(10, 40, 5)
        self.apz = self.calculate_apz(20)
        self.linreg = self.calculate_linreg(14)
        self.dynamic_momentum = self.calculate_dynamic_momentum(3)

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
        self.data['fib_weighted_ma'] = self.data['close'].rolling(window=self.fib_weight_ma_period).apply(
            lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
        self.data['smoothed_fib_weighted_ma'] = self.data['fib_weighted_ma'].rolling(window=self.smoothing_simple_ma_period).mean()

    def calculate_indicators(self):
        self._calculate_fib_weighted_ma()
        # Additional indicator calculations

    def calculate_zlema(self, period):
        return self.data['close'].ewm(span=period).mean()

    def calculate_ema(self, period):
        return self.data['close'].ewm(span=period).mean()

    def calculate_sma(self, period):
        return self.data['close'].rolling(window=period).mean()

    def calculate_ma_envelopes(self, period):
        return (self.data['close'].rolling(window=period).mean(),
                self.data['close'].rolling(window=period).mean() * 1.05,
                self.data['close'].rolling(window=period).mean() * 0.95)

    def calculate_disparity_index(self, period):
        return (self.data['close'] - self.data['close'].rolling(window=period).mean()) / self.data['close'].rolling(window=period).mean()

    def calculate_atr(self, period):
        tr = self.data['close'].diff().abs()
        return tr.rolling(window=period).mean()

    def calculate_macd(self, fast_period, slow_period, signal_period):
        ema_fast = self.data['close'].ewm(span=fast_period).mean()
        ema_slow = self.data['close'].ewm(span=slow_period).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        return macd_line, signal_line, macd_line - signal_line

    def calculate_adx(self, period):
        dm_plus = self.data['close'].diff().clip(lower=0)
        dm_minus = self.data['close'].diff().clip(upper=0).abs()
        tr = self.data['close'].diff().abs()
        dx = (abs(dm_plus - dm_minus) / (dm_plus + dm_minus)).rolling(window=period).mean()
        return dx.ewm(span=period).mean()

    def calculate_vortex(self, period):
        vm_plus = self.data['close'] - self.data['close'].shift()
        vm_minus = self.data['close'] - self.data['close'].shift()
        tr = self.data['close'].diff().abs()
        return vm_plus.rolling(window=period).sum() / tr.rolling(window=period).sum(), vm_minus.rolling(window=period).sum() / tr.rolling(window=period).sum()

    def calculate_pfe(self, period, smoothing_period):
        delta = self.data['close'].diff(period).abs()
        price_change = (self.data['close'] - self.data['close'].shift(period)).abs()
        smoothing = self.data['close'].rolling(window=smoothing_period).mean()
        return price_change / delta

    def calculate_williams_r(self, period):
        high_max = self.data['close'].rolling(window=period).max()
        low_min = self.data['close'].rolling(window=period).min()
        return (high_max - self.data['close']) / (high_max - low_min) * -100

    def calculate_momentum(self, period):
        return self.data['close'].diff(period)

    def calculate_cci(self, period):
        typical_price = self.data['close']
        return (typical_price - typical_price.rolling(window=period).mean()) / (0.015 * typical_price.rolling(window=period).std())

    def calculate_rsi(self, period, smoothing_period):
        delta = self.data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_trix(self, period, signal_period):
        trix = self.data['close'].ewm(span=period).mean()
        trix_signal = trix.ewm(span=signal_period).mean()
        return trix, trix_signal

    def calculate_chaikin(self, fast_period, slow_period):
        ad = (2 * self.data['close'] - self.data['close'] - self.data['close']) / (self.data['close'] - self.data['close']) * self.data['volume']
        chaikin_osc = ad.ewm(span=fast_period).mean() - ad.ewm(span=slow_period).mean()
        return chaikin_osc

    def calculate_fisher_transform(self, period):
        high = self.data['close'].rolling(window=period).max()
        low = self.data['close'].rolling(window=period).min()
        value = 0.33 * 2 * ((self.data['close'] - low) / (high - low) - 0.5) + 0.67 * self.data['close'].shift()
        return value

    def calculate_obv(self):
        return (np.sign(self.data['close'].diff()) * self.data['volume']).cumsum()

    def calculate_rind(self, short_period, long_period):
        return (self.data['close'].ewm(span=short_period).mean() - self.data['close'].ewm(span=long_period).mean())

    def calculate_bolli(self, period):
        middle_band = self.data['close'].rolling(window=period).mean()
        upper_band = middle_band + 2 * self.data['close'].rolling(window=period).std()
        lower_band = middle_band - 2 * self.data['close'].rolling(window=period).std()
        return upper_band, middle_band, lower_band

    def calculate_stochastics_fast(self, period_k, period_d):
        min_low = self.data['close'].rolling(window=period_k).min()
        max_high = self.data['close'].rolling(window=period_k).max()
        fast_k = 100 * (self.data['close'] - min_low) / (max_high - min_low)
        fast_d = fast_k.rolling(window=period_d).mean()
        return fast_k, fast_d

    def calculate_cmo(self, period):
        return self.data['close'].diff(period)

    def calculate_rss(self, short_period, long_period, smooth_period):
        rss = (self.data['close'].ewm(span=short_period).mean() - self.data['close'].ewm(span=long_period).mean()).rolling(window=smooth_period).mean()
        return rss

    def calculate_apz(self, period):
        return self.data['close'].rolling(window=period).mean() + 2 * self.data['close'].rolling(window=period).std()

    def calculate_linreg(self, period):
        return self.data['close'].rolling(window=period).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0])

    def calculate_dynamic_momentum(self, period):
        return self.data['close'].diff(period)

    def _bar_update(self):
        if len(self.data) < self.bars_required_to_trade:
            return

        if self.enable_fib_weight_ma_cross:
            self._calculate_fib_weighted_ma()

        self._generate_signals()
        self._manage_positions()
        self._log_trade()

    def _generate_signals(self):
        self.signals['buy_signal'] = self._buy_signal()
        self.signals['sell_signal'] = self._sell_signal()

    def _buy_signal(self):
        buy_signal = np.zeros(len(self.data))
        if self.enable_fib_weight_ma_cross:
            buy_signal = cross_above(self.data['fib_weighted_ma'], self.data['smoothed_fib_weighted_ma'])
        return buy_signal

    def _sell_signal(self):
        sell_signal = np.zeros(len(self.data))
        if self.enable_fib_weight_ma_cross:
            sell_signal = cross_below(self.data['fib_weighted_ma'], self.data['smoothed_fib_weighted_ma'])
        return sell_signal

    def _manage_positions(self):
        for i in range(len(self.data)):
            if self.position == 0 and self.signals['buy_signal'][i]:
                self._enter_position(i, 'long')
            elif self.position > 0 and self.signals['sell_signal'][i]:
                self._exit_position(i)
            elif self.position < 0 and self.signals['buy_signal'][i]:
                self._exit_position(i)
            if self.position != 0:
                self._apply_money_management(i)

    def _enter_position(self, index, direction):
        if direction == 'long':
            self.position = self.default_quantity
            self.entry_price = self.data['close'][index]
            self.stop_loss = self.data['close'][index] - 2 * self.data['atr'][index] if 'atr' in self.data.columns else self.entry_price * 0.98
            self.take_profit = self.entry_price + 2 * self.data['atr'][index] if 'atr' in self.data.columns else self.entry_price * 1.02
        elif direction == 'short':
            self.position = -self.default_quantity
            self.entry_price = self.data['close'][index]
            self.stop_loss = self.data['close'][index] + 2 * self.data['atr'][index] if 'atr' in self.data.columns else self.entry_price * 1.02
            self.take_profit = self.entry_price - 2 * self.data['atr'][index] if 'atr' in self.data.columns else self.entry_price * 0.98
        self._log_entry(index, direction)

    def _exit_position(self, index):
        if self.position > 0:
            self.pnl += self.data['close'][index] - self.entry_price
        elif self.position < 0:
            self.pnl += self.entry_price - self.data['close'][index]
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self._log_exit(index)

    def _apply_money_management(self, index):
        if self.position > 0:
            if self.data['close'][index] <= self.stop_loss or self.data['close'][index] >= self.take_profit:
                self._exit_position(index)
        elif self.position < 0:
            if self.data['close'][index] >= self.stop_loss or self.data['close'][index] <= self.take_profit:
                self._exit_position(index)

    def _log_entry(self, index, direction):
        log_entry = {
            'timestamp': self.data.index[index],
            'action': 'enter',
            'direction': direction,
            'price': self.data['close'][index],
            'quantity': self.default_quantity,
            'position': self.position,
            'pnl': self.pnl
        }
        self.trade_log.append(log_entry)

    def _log_exit(self, index):
        log_exit = {
            'timestamp': self.data.index[index],
            'action': 'exit',
            'price': self.data['close'][index],
            'quantity': self.default_quantity,
            'position': self.position,
            'pnl': self.pnl
        }
        self.trade_log.append(log_exit)

    def _log_trade(self):
        if self.print_trade_log:
            for log in self.trade_log:
                print(log)

    def execute_strategy(self):
        for i in range(len(self.data)):
            self._bar_update()

    def plot_trades(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['close'], label='Close Price')
        plt.plot(self.data['smoothed_fib_weighted_ma'], label='Smoothed Fib Weighted MA')
        for log in self.trade_log:
            if log['action'] == 'enter':
                if log['direction'] == 'long':
                    draw_triangle_up(self.data['close'], log['timestamp'], 'green')
                elif log['direction'] == 'short':
                    draw_triangle_down(self.data['close'], log['timestamp'], 'red')
            elif log['action'] == 'exit':
                draw_triangle_down(self.data['close'], log['timestamp'], 'blue')
        plt.legend()
        plt.show()

    def plot_performance(self):
        df = pd.DataFrame(self.trade_log)
        if df.empty:
            print("No trades to display performance.")
            return
        df.set_index('timestamp', inplace=True)
        df['cumulative_pnl'] = df['pnl'].cumsum()
        df['cumulative_pnl'].plot(figsize=(12, 6), title='Cumulative PnL')
        plt.show()

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

    def plot(self, strategy, speed=0.1):
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

            if date in strategy.trade_log:
                for log in strategy.trade_log[date]:
                    if log['action'] == 'enter':
                        if log['direction'] == 'long':
                            draw_triangle_up(prices, x + 1, 'blue')
                        elif log['direction'] == 'short':
                            draw_triangle_down(prices, x + 1, 'blue')
                    elif log['action'] == 'exit':
                        draw_triangle_down(prices, x + 1, 'red')

            plt.pause(speed)

        # Convert x-ticks to dates
        x_ticks = ax.get_xticks()
        x_labels = [dates[int(tick)-1] if 0 <= int(tick)-1 < len(dates) else '' for tick in x_ticks]
        ax.set_xticklabels(x_labels, rotation=45, ha='right')

        plt.show(block=True)  # Ensure the plot window stays open

# Usage example
if __name__ == "__main__":
    filename = "C:/Users/Administrator/Documents/NinjaTrader 8/db/replay/temp_preprocessed/20240305.csv"
    renko_chart = Renko(filename=filename)
    renko_chart.set_brick_size(brick_size=30, brick_threshold=5)
    renko_data = renko_chart.build()

    # Prepare data for strategy
    renko_df = pd.DataFrame(renko_data)
    renko_df.set_index('date', inplace=True)
    renko_df['close'] = renko_df['price']
    renko_df['volume'] = 1  # Dummy volume
    renko_df['high'] = renko_df['close']
    renko_df['low'] = renko_df['close']

    strategy = Delta2Strategy(renko_df)
    strategy.calculate_indicators()
    strategy.execute_strategy()

    renko_chart.plot(strategy, speed=0.1)
    strategy.plot_performance()
