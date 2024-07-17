import pandas as pd
import numpy as np
import mplfinance as mpf
from matplotlib import animation
import h5py
import psycopg2
import tempfile
from datetime import datetime, timedelta

# Assuming renkodf.py is in the same directory
from renkodf import RenkoWS

# PostgreSQL database credentials
db_credentials = {
    "dbname": "defaultdb",
    "user": "doadmin",
    "password": "AVNS_hnzmIdBmiO7aj5nylWW",
    "host": "nocodemldb-do-user-16993120-0.c.db.ondigitalocean.com",
    "port": 25060,
    "sslmode": "require"
}

def connect_db():
    print("Connecting to database...")
    conn = psycopg2.connect(**db_credentials)
    print("Connected to database.")
    return conn

def fetch_available_dates(conn):
    cursor = conn.cursor()
    print("Fetching available dates...")
    cursor.execute('SELECT filename FROM h5_files;')
    rows = cursor.fetchall()
    
    dates = [row[0].replace('.h5', '') for row in rows]
    print("Fetched available dates.")
    return sorted(dates)

def load_h5_from_db(conn, filename):
    cursor = conn.cursor()
    print(f"Loading H5 file from database: {filename}")
    cursor.execute('SELECT data FROM h5_files WHERE filename = %s;', (filename,))
    result = cursor.fetchone()
    
    if result is None:
        print(f"File {filename} not found in database.")
        return None
    
    binary_data = result[0]
    temp_dir = tempfile.gettempdir()
    file_path = f"{temp_dir}/{filename}.h5"
    
    with open(file_path, "wb") as f:
        f.write(binary_data)
    
    print(f"Loaded H5 file from database: {filename}")
    return file_path

def process_h5(file_path):
    print(f"Processing H5 file: {file_path}")
    with h5py.File(file_path, "r") as f:
        timestamps = [t.decode('utf-8') for t in f['L2/Timestamp'][:]]
        timestamps = pd.to_datetime(timestamps, errors='coerce')
        prices = f['L2/Price'][:].astype(float)
        df = pd.DataFrame({"datetime": timestamps, "close": prices})
        df.dropna(subset=["datetime"], inplace=True)
        print(f"Processed H5 file: {file_path}")
        return df

def load_data_for_date(conn, date):
    filename = date.strftime('%Y%m%d') + ".h5"
    file_path = load_h5_from_db(conn, filename)
    if file_path:
        df = process_h5(file_path)
        return df
    return None

def add_high_low(df):
    df['High'] = df['close'] + (np.random.rand(len(df)) * 10)  # Simulate 'High' as close + random value
    df['Low'] = df['close'] - (np.random.rand(len(df)) * 10)   # Simulate 'Low' as close - random value
    df['Close'] = df['close']  # Ensure 'Close' column is present and correctly named
    return df

def data_generator(conn, dates, step):
    for date in dates:
        print(f"Loading data for date: {date.strftime('%Y-%m-%d')}")
        df_ticks = load_data_for_date(conn, date)
        if df_ticks is not None:
            df_ticks = add_high_low(df_ticks)  # Add 'High', 'Low', and 'Close' columns
            for start_idx in range(0, len(df_ticks), step):
                yield df_ticks.iloc[start_idx:start_idx + step]
        else:
            print(f"No data for {date.strftime('%Y-%m-%d')}")

class Delta2Strategy:
    def __init__(self, data, trade_type="BOTH", starting_capital=300000):
        self.data = data
        self.trade_type = trade_type
        self.trade_log = []
        self.pnl = 0.0
        self.starting_capital = starting_capital
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss = pd.Series(np.zeros(len(data)), index=data.index)
        self.take_profit = pd.Series(np.zeros(len(data)), index=data.index)
        self.signals = {'buy_signal': np.zeros(len(data)), 'sell_signal': np.zeros(len(data))}
        self.cooldown = pd.Series(np.zeros(len(data)), index=data.index)  # Cooldown period after a trade
        self.min_holding_period = 5  # Minimum number of bars to hold a position
        self.holding_period = pd.Series(np.zeros(len(data)), index=data.index)  # Tracks holding period of current position
        self._initialize_parameters()
        self._initialize_fibonacci_weightings()
        self._calculate_psar()

    def _initialize_parameters(self):
        self.bars_required_to_trade = 20
        self.default_quantity = 1
        self.enable_fib_weight_ma_cross = True
        self.fib_weight_ma_period = 9
        self.smoothing_simple_ma_period = 20
        self.acceleration = 0.0162  
        self.max_acceleration = 0.162
        self.acceleration_step = 0.0162
        self.cooldown_period = 1  # Number of bars to wait before considering a new trade

    def _initialize_fibonacci_weightings(self):
        self.fib_weightings = self._calculate_fibonacci_weights(self.fib_weight_ma_period)
        self.fib_weighted_ma = pd.Series(np.zeros(len(self.data)), index=self.data.index)
        self.smoothed_fib_weighted_ma = pd.Series(np.zeros(len(self.data)), index=self.data.index)

    def _calculate_fibonacci_weights(self, period):
        fib_weights = np.zeros(period)
        a, b = 1, 1
        for i in range(period):
            fib_weights[i] = a
            a, b = b, a + b
        fib_weights = fib_weights[::-1]  # Reverse to apply the largest weight to the most recent bar
        fib_weights /= np.sum(fib_weights)  # Normalize weights
        return fib_weights

    def _calculate_fib_weighted_ma(self):
        close_prices = self.data['Close']
        fib_ma = np.zeros(len(close_prices))
        
        for i in range(len(close_prices)):
            if i >= self.fib_weight_ma_period - 1:
                fib_ma[i] = np.dot(close_prices.iloc[i - self.fib_weight_ma_period + 1:i + 1], self.fib_weightings)
        
        self.fib_weighted_ma = pd.Series(fib_ma, index=self.data.index)
        self.smoothed_fib_weighted_ma = self.fib_weighted_ma.rolling(window=self.smoothing_simple_ma_period).mean()

    def _calculate_psar(self):
        if self.data.empty:
            return

        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']

        af = self.acceleration
        max_af = self.max_acceleration
        psar = np.zeros(len(close))
        bull = True
        ep = low.iloc[0]
        hp = high.iloc[0]
        lp = low.iloc[0]

        for i in range(1, len(close)):
            psar[i] = psar[i - 1] + af * (hp if bull else lp - psar[i - 1])
            reverse = False

            if bull:
                if low.iloc[i] < psar[i]:
                    bull = False
                    psar[i] = hp
                    lp = low.iloc[i]
                    af = self.acceleration
                    reverse = True
            else:
                if high.iloc[i] > psar[i]:
                    bull = True
                    psar[i] = lp
                    hp = high.iloc[i]
                    af = self.acceleration
                    reverse = True

            if not reverse:
                if bull:
                    if high.iloc[i] > hp:
                        hp = high.iloc[i]
                        af = min(af + self.acceleration_step, max_af)
                    if i > 1 and low.iloc[i - 1] < psar[i]:
                        psar[i] = low.iloc[i - 1]
                    if i > 2 and low.iloc[i - 2] < psar[i]:
                        psar[i] = low.iloc[i - 2]
                else:
                    if low.iloc[i] < lp:
                        lp = low.iloc[i]
                        af = min(af + self.acceleration_step, max_af)
                    if i > 1 and high.iloc[i - 1] > psar[i]:
                        psar[i] = high.iloc[i - 1]
                    if i > 2 and high.iloc[i - 2] > psar[i]:
                        psar[i] = high.iloc[i - 2]

        self.data['PSAR'] = pd.Series(psar, index=self.data.index)
        print("PSAR Indicator Head:")
        print(self.data[['PSAR']].head())

    def _bar_update(self, i):
        self._generate_signals(i)
        self._manage_positions(i)

    def _generate_signals(self, i):
        if self.trade_type in ["BOTH", "LONG ONLY"]:
            self.signals['buy_signal'][i] = True
        if self.trade_type in ["BOTH", "SHORT ONLY"]:
            self.signals['sell_signal'][i] = True

    def _manage_positions(self, i):
        if self.position == 0:
            if not self.cooldown.iloc[i]:
                if self.signals['buy_signal'][i] and self.trade_type in ["BOTH", "LONG ONLY"]:
                    self._enter_position(i, 'long')
                elif self.signals['sell_signal'][i] and self.trade_type in ["BOTH", "SHORT ONLY"]:
                    self._enter_position(i, 'short')
        elif self.position > 0:
            self.holding_period.iloc[i] += 1
            if self.holding_period.iloc[i] >= self.min_holding_period and (self.signals['sell_signal'][i] or self.data['Close'].iloc[i] <= self.stop_loss.iloc[i]):
                self._exit_position(i)
                self.cooldown.iloc[i] = self.cooldown_period
        elif self.position < 0:
            self.holding_period.iloc[i] += 1
            if self.holding_period.iloc[i] >= self.min_holding_period and (self.signals['buy_signal'][i] or self.data['Close'].iloc[i] >= self.stop_loss.iloc[i]):
                self._exit_position(i)
                self.cooldown.iloc[i] = self.cooldown_period

        # Update cooldown period
        if self.cooldown.iloc[i] > 0:
            self.cooldown.iloc[i + 1:i + self.cooldown_period + 1] = self.cooldown.iloc[i] - 1

    def _enter_position(self, index, direction):
        print(f"Entering {direction} position at index {index}")
        if direction == 'long':
            self.position = self.default_quantity
            self.entry_price = self.data['Close'].iloc[index]
            self.stop_loss.iloc[index] = self.data['PSAR'].iloc[index]  # Use PSAR as stop loss
            self.take_profit.iloc[index] = self.entry_price + 90  # Adjust take profit level if needed
        elif direction == 'short':
            self.position = -self.default_quantity
            self.entry_price = self.data['Close'].iloc[index]
            self.stop_loss.iloc[index] = self.data['PSAR'].iloc[index]  # Use PSAR as stop loss
            self.take_profit.iloc[index] = self.entry_price - 90  # Adjust take profit level if needed
        self.holding_period.iloc[index] = 0
        self._log_entry(index, direction)

    def _exit_position(self, index):
        print(f"Exiting position at index {index}")
        if self.position > 0:
            self.pnl += (self.data['Close'].iloc[index] - self.entry_price) * self.default_quantity * 20
        elif self.position < 0:
            self.pnl += (self.entry_price - self.data['Close'].iloc[index]) * self.default_quantity * 20
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss.iloc[index] = 0.0
        self.take_profit.iloc[index] = 0.0
        self.holding_period.iloc[index] = 0
        self._log_exit(index)

    def _log_entry(self, index, direction):
        log_entry = {
            'timestamp': self.data.index[index],
            'action': 'enter',
            'direction': direction,
            'price': self.data['Close'].iloc[index],
            'quantity': self.default_quantity,
            'position': self.position,
            'pnl': self.pnl
        }
        self.trade_log.append(log_entry)
        print(f"Trade Entered: {log_entry}")

    def _log_exit(self, index):
        log_exit = {
            'timestamp': self.data.index[index],
            'action': 'exit',
            'price': self.data['Close'].iloc[index],
            'quantity': self.default_quantity,
            'position': self.position,
            'pnl': self.pnl
        }
        self.trade_log.append(log_exit)
        print(f"Trade Exited: {log_exit}")

    def execute_strategy(self):
        if self.data.empty:
            return
        print("DataFrame columns before executing strategy:")
        print(self.data.columns)
        self._calculate_fib_weighted_ma()  # Calculate the Fibonacci-weighted moving averages
        for i in range(len(self.data)):
            self._bar_update(i)

    def detailed_analysis(self):
        trades_df = pd.DataFrame(self.trade_log)
        if trades_df.empty:
            print("No trades were made.")
            return

        total_trades = len(trades_df[trades_df['action'] == 'enter'])
        winning_trades = trades_df[(trades_df['action'] == 'exit') & (trades_df['pnl'].diff() > 0)]
        losing_trades = trades_df[(trades_df['action'] == 'exit') & (trades_df['pnl'].diff() <= 0)]

        total_pnl = trades_df['pnl'].iloc[-1] - trades_df['pnl'].iloc[0] + self.starting_capital
        average_pnl_per_trade = trades_df['pnl'].diff().mean()
        max_drawdown = (trades_df['pnl'].cummax() - trades_df['pnl']).max()
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

        print("\nDetailed Trade Analysis:")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total PnL: {total_pnl:.2f}")
        print(f"Average PnL per Trade: {average_pnl_per_trade:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2f}")

        print("\nTrade Log:")
        print(trades_df)

    def save_trade_log(self, filename):
        trades_df = pd.DataFrame(self.trade_log)
        trades_df.to_csv(filename, index=False)
        print(f"Trade log saved to {filename}")

def animate(frame, renko_chart, ax1, ax2, start_date, end_date, my_style, strategy):
    df_ticks = frame
    if df_ticks is None or df_ticks.empty:
        return
    
    for i in range(len(df_ticks)):
        timestamp = df_ticks['datetime'].iat[i]
        price = df_ticks['Close'].iat[i]  # Changed 'close' to 'Close'
        renko_chart.add_prices(timestamp.value // 10**6, price)  # Convert to milliseconds
        
        # Execute strategy on new data
        strategy.data = df_ticks.iloc[:i+1]
        strategy.execute_strategy()
        
        # Plot trades
        for trade in strategy.trade_log[-2:]:  # Only plot the last 2 trades to avoid clutter
            if trade['action'] == 'enter':
                ax1.annotate(f"E", (timestamp, trade['price']), textcoords="offset points", xytext=(0,10), ha='center', color='green' if trade['direction'] == 'long' else 'red')
            if trade['action'] == 'exit':
                ax1.annotate(f"X", (timestamp, trade['price']), textcoords="offset points", xytext=(0,-10), ha='center', color='green' if trade['position'] > 0 else 'red')

    df_wicks = renko_chart.renko_animate('wicks', max_len=10000, keep=5000)
    
    ax1.clear()
    ax2.clear()

    title = f"NQ: {start_date} to {end_date}"
    mpf.plot(df_wicks, type='candle', ax=ax1, volume=ax2, axtitle='6AM to NY Close (10:30PM)', style=my_style)
    print(f"Animating data from {df_ticks['datetime'].iat[0].strftime('%Y-%m-%d')}")

def main():
    conn = connect_db()
    available_dates = fetch_available_dates(conn)
    
    print("Available dates:")
    for date in available_dates:
        print(date)
    
    start_date = input("Enter the start date (YYYYMMDD): ")
    end_date = input("Enter the end date (YYYYMMDD): ")

    print("Choose trade type:")
    trade_types = ["LONG ONLY", "SHORT ONLY", "BOTH"]
    for i, trade_type in enumerate(trade_types, 1):
        print(f"{i}. {trade_type}")
    trade_choice = int(input("Enter the number of your choice: "))
    if trade_choice not in range(1, 4):
        print("Invalid choice. Defaulting to BOTH.")
        trade_type = "BOTH"
    else:
        trade_type = trade_types[trade_choice - 1]
    
    speeds = {
        '1x': 1,
        '5x': 5,
        '10x': 10,
        '25x': 25,
        '50x': 50,
        '100x': 100,
        '500x': 500,
        '1000x': 1000,
        '5000x': 5000,
        '10000x': 10000,
        '50000x': 50000
    }

    print("Select the speed:")
    for key in speeds:
        print(f"{key}: {speeds[key]}x")

    speed_choice = input("Enter the speed (e.g., 1x, 5x, 10x, etc.): ")

    if speed_choice in speeds:
        step = speeds[speed_choice]
    else:
        print("Invalid choice. Defaulting to 1x speed.")
        step = 1

    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Initializing Renko chart with first date's data
    df_l2 = load_data_for_date(conn, dates[0])
    if df_l2 is None:
        print(f"No data available for {dates[0].strftime('%Y-%m-%d')}")
        return

    df_l2 = add_high_low(df_l2)  # Ensure 'High', 'Low', and 'Close' columns are present
    initial_timestamp = df_l2['datetime'].iat[0].value // 10**6  # Convert to milliseconds
    initial_price = df_l2['Close'].iat[0]  # Changed 'close' to 'Close'

    brick_size = 3  # Adjust based on NinjaTrader settings
    brick_threshold = 5  # Adjust based on NinjaTrader settings

    renko_chart = RenkoWS(initial_timestamp, initial_price, brick_size=brick_size, brick_threshold=brick_threshold)

    # Define custom style
    my_style = mpf.make_mpf_style(base_mpf_style='charles', 
                                  marketcolors=mpf.make_marketcolors(up='g', down='r', inherit=True))

    fig, axes = mpf.plot(renko_chart.initial_df, returnfig=True, volume=True,
                         figsize=(16, 9), panel_ratios=(2, 1),
                         title=f"NQ: {start_date} to {end_date}", type='candle', style=my_style)
    ax1 = axes[0]
    ax2 = axes[2]

    # Initialize the strategy
    delta2_strategy = Delta2Strategy(data=df_l2, trade_type=trade_type)
    
    data = data_generator(conn, dates, step)
    ani = animation.FuncAnimation(fig, animate, fargs=(renko_chart, ax1, ax2, start_date, end_date, my_style, delta2_strategy), frames=data, interval=500, repeat=False, cache_frame_data=False)  # Throttle the animation with interval=500
    
    print("Starting animation")
    mpf.show()
    print("Animation done")

if __name__ == "__main__":
    main()
