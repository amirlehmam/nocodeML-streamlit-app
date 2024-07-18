import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import sv_ttk
import pandas as pd
import numpy as np
import h5py
import psycopg2
import tempfile
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from renkodf import Renko
import multiprocessing

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
    return psycopg2.connect(**db_credentials)

def fetch_available_dates():
    with connect_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute('SELECT filename FROM h5_files;')
            rows = cursor.fetchall()
    return sorted([row[0].replace('.h5', '') for row in rows])

def load_h5_from_db(filename):
    with connect_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute('SELECT data FROM h5_files WHERE filename = %s;', (filename,))
            result = cursor.fetchone()
    
    if result is None:
        return None
    
    binary_data = result[0]
    temp_dir = tempfile.gettempdir()
    file_path = f"{temp_dir}/{filename}.h5"
    
    with open(file_path, "wb") as f:
        f.write(binary_data)
    
    return file_path

def process_h5(file_path, brick_size, brick_threshold):
    with h5py.File(file_path, "r") as f:
        timestamps = pd.to_datetime([t.decode('utf-8') for t in f['L2/Timestamp'][:]], errors='coerce')
        prices = f['L2/Price'][:].astype(float)
        df = pd.DataFrame({"datetime": timestamps, "close": prices}).dropna(subset=["datetime"])
        
        renko = Renko(df, brick_size, brick_threshold=brick_threshold)
        renko_df = renko.renko_df()
        renko_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        return renko_df

def load_data_for_date(date, brick_size, brick_threshold):
    filename = date.strftime('%Y%m%d') + ".h5"
    file_path = load_h5_from_db(filename)
    if file_path:
        return process_h5(file_path, brick_size, brick_threshold)
    return None

def process_date(date, strategy_class, brick_size, brick_threshold, **kwargs):
    print(f"Processing date: {date}")
    data = load_data_for_date(date, brick_size, brick_threshold)
    if data is not None:
        temp_strategy = strategy_class(data, **kwargs)
        trade_log, pnl = temp_strategy.backtest()
        return trade_log, pnl
    else:
        return pd.DataFrame(), 0.0

class Delta2Strategy:
    def __init__(self, data, starting_capital=300000, tp=90, sl=90, fib_ma_period=9, smooth_ma_period=20, psar_acceleration=0.0162, psar_max_acceleration=0.162, psar_step=0.02):
        self.data = data
        self.trade_log = []
        self.pnl = 0.0
        self.starting_capital = starting_capital
        self.position = 0
        self.entry_price = 0.0
        self.tp = tp
        self.sl = sl
        self.fib_weight_ma_period = fib_ma_period
        self.smoothing_simple_ma_period = smooth_ma_period
        self.acceleration = psar_acceleration
        self.max_acceleration = psar_max_acceleration
        self.psar_step = psar_step
        self.stop_loss = pd.Series(np.zeros(len(data)), index=data.index)
        self.take_profit = pd.Series(np.zeros(len(data)), index=data.index)
        self.signals = {'buy_signal': np.zeros(len(data), dtype=bool), 'sell_signal': np.zeros(len(data), dtype=bool)}
        self.cooldown = pd.Series(np.zeros(len(data)), index=data.index)
        self.cooldown_period = 10  # Adjusted cooldown period to prevent rapid trades
        self.last_trade_time = None  # Track the last trade time
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

    def _initialize_fibonacci_weightings(self):
        self.fib_weightings = self._calculate_fibonacci_weights(self.fib_weight_ma_period)
        self.fib_weighted_ma = pd.Series(np.zeros(len(self.data)), index=self.data.index)
        self.smoothed_fib_weighted_ma = self.fib_weighted_ma.rolling(window=self.smoothing_simple_ma_period).mean()

    def _calculate_fibonacci_weights(self, period):
        fib_weights = np.zeros(period)
        a, b = 1, 1
        for i in range(period):
            fib_weights[i] = a
            a, b = b, a + b
        fib_weights = fib_weights[::-1]
        fib_weights /= np.sum(fib_weights)
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
                        af = min(af + self.psar_step, max_af)
                    if i > 1 and low.iloc[i - 1] < psar[i]:
                        psar[i] = low.iloc[i - 1]
                    if i > 2 and low.iloc[i - 2] < psar[i]:
                        psar[i] = low.iloc[i - 2]
                else:
                    if low.iloc[i] < lp:
                        lp = low.iloc[i]
                        af = min(af + self.psar_step, max_af)
                    if i > 1 and high.iloc[i - 1] > psar[i]:
                        psar[i] = high.iloc[i - 1]
                    if i > 2 and high.iloc[i - 2] > psar[i]:
                        psar[i] = high.iloc[i - 2]

        self.data['PSAR'] = pd.Series(psar, index=self.data.index)

    def _bar_update(self, i):
        self._generate_signals(i)
        self._manage_positions(i)

    def _generate_signals(self, i):
        if i >= self.fib_weight_ma_period - 1:
            self.fib_weighted_ma.iloc[i] = np.dot(self.data['Close'].iloc[i - self.fib_weight_ma_period + 1:i + 1], self.fib_weightings)
            if i >= self.fib_weight_ma_period + self.smoothing_simple_ma_period - 2:
                self.smoothed_fib_weighted_ma.iloc[i] = self.fib_weighted_ma.iloc[i - self.smoothing_simple_ma_period + 1:i + 1].mean()
        
        if i >= self.fib_weight_ma_period + self.smoothing_simple_ma_period - 2:
            if self.fib_weighted_ma.iloc[i] > self.smoothed_fib_weighted_ma.iloc[i] and self.fib_weighted_ma.iloc[i - 1] <= self.smoothed_fib_weighted_ma.iloc[i - 1]:
                self.signals['buy_signal'][i] = True
            elif self.fib_weighted_ma.iloc[i] < self.smoothed_fib_weighted_ma.iloc[i] and self.fib_weighted_ma.iloc[i - 1] >= self.smoothed_fib_weighted_ma.iloc[i - 1]:
                self.signals['sell_signal'][i] = True
            else:
                self.signals['buy_signal'][i] = self.signals['sell_signal'][i] = False

    def _manage_positions(self, i):
        current_time = self.data.index[i]
        
        if self.last_trade_time is not None and (current_time - self.last_trade_time).total_seconds() < self.cooldown_period:
            return

        if self.position == 0:
            if self.signals['buy_signal'][i]:
                self._enter_position(i, 'long')
                self.last_trade_time = current_time
            elif self.signals['sell_signal'][i]:
                self._enter_position(i, 'short')
                self.last_trade_time = current_time
        elif self.position > 0:
            self.stop_loss.iloc[i] = max(self.stop_loss.iloc[i-1], self.data['PSAR'].iloc[i] if 'PSAR' in self.data.columns else self.stop_loss.iloc[i-1])
            if self.signals['sell_signal'][i] or self.data['Close'].iloc[i] <= self.stop_loss.iloc[i] or self.data['Close'].iloc[i] >= self.take_profit.iloc[i]:
                self._exit_position(i)
                self.last_trade_time = current_time
        elif self.position < 0:
            self.stop_loss.iloc[i] = min(self.stop_loss.iloc[i-1], self.data['PSAR'].iloc[i] if 'PSAR' in self.data.columns else self.stop_loss.iloc[i-1])
            if self.signals['buy_signal'][i] or self.data['Close'].iloc[i] >= self.stop_loss.iloc[i] or self.data['Close'].iloc[i] <= self.take_profit.iloc[i]:
                self._exit_position(i)
                self.last_trade_time = current_time

    def _enter_position(self, index, direction):
        print(f"Entering {direction} position at index {index}")
        if direction == 'long':
            self.position = self.default_quantity
            self.entry_price = self.data['Close'].iloc[index]
            self.stop_loss.iloc[index] = self.entry_price - self.sl
            self.take_profit.iloc[index] = self.entry_price + self.tp
        elif direction == 'short':
            self.position = -self.default_quantity
            self.entry_price = self.data['Close'].iloc[index]
            self.stop_loss.iloc[index] = self.entry_price + self.sl
            self.take_profit.iloc[index] = self.entry_price - self.tp
        self.trade_log.append((self.data.index[index], 'entry', direction, self.entry_price, self.position))

    def _exit_position(self, index):
        exit_price = self.data['Close'].iloc[index]
        self.pnl += ((exit_price - self.entry_price) * self.position) * 20
        self.trade_log.append((self.data.index[index], 'exit', 'long' if self.position > 0 else 'short', exit_price, self.position))
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss.iloc[index] = self.take_profit.iloc[index] = 0.0

    def backtest(self):
        for i in range(self.bars_required_to_trade, len(self.data)):
            self._bar_update(i)

        trade_df = pd.DataFrame(self.trade_log, columns=['timestamp', 'action', 'direction', 'price', 'quantity'])
        pnl = self.starting_capital + self.pnl
        return trade_df, pnl

def calculate_summary_metrics(trade_df):
    if trade_df.empty:
        return pd.DataFrame({
            'Metric': [], 'Value': []
        })

    gross_profit = trade_df[(trade_df['action'] == 'exit') & (trade_df['direction'] == 'long')]['price'].diff().sum() * 20
    gross_loss = trade_df[(trade_df['action'] == 'exit') & (trade_df['direction'] == 'short')]['price'].diff().sum() * 20
    net_profit_loss = gross_profit - gross_loss
    total_trades = len(trade_df[trade_df['action'] == 'exit'])
    winning_trades = len(trade_df[(trade_df['action'] == 'exit') & (trade_df['price'].diff() > 0)])
    losing_trades = total_trades - winning_trades
    max_drawdown = trade_df['price'].diff().min() * 20
    percent_profitable = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    avg_win = gross_profit / winning_trades if winning_trades > 0 else 0
    avg_loss = gross_loss / losing_trades if losing_trades > 0 else 0
    ratio_avg_win_loss = avg_win / abs(avg_loss) if avg_loss != 0 else 0
    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else 0
    sharpe_ratio = net_profit_loss / trade_df['price'].std() * np.sqrt(len(trade_df)) if trade_df['price'].std() != 0 else 0
    sortino_ratio = sharpe_ratio  # This is a simplification
    avg_trade = net_profit_loss / total_trades if total_trades > 0 else 0

    metrics = {
        'Metric': [
            'Gross Profit ($)', 'Gross Loss ($)', 'Net Profit/Loss ($)', 'Total Trades',
            'Winning Trades', 'Losing Trades', 'Max Drawdown ($)', 'Percent Profitable (%)',
            'Ratio Avg Win / Avg Loss', 'Profit Factor', 'Sharpe Ratio', 'Sortino Ratio',
            'Average Trade ($)', 'Average Winning Trade ($)', 'Average Losing Trade ($)'
        ],
        'Value': [
            gross_profit, gross_loss, net_profit_loss, total_trades,
            winning_trades, losing_trades, max_drawdown, percent_profitable,
            ratio_avg_win_loss, profit_factor, sharpe_ratio, sortino_ratio,
            avg_trade, avg_win, avg_loss
        ]
    }
    
    return pd.DataFrame(metrics)

def backtest_strategy(strategy_class, dates, brick_size, brick_threshold, **kwargs):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for date in dates:
            futures.append(executor.submit(process_date, date, strategy_class, brick_size, brick_threshold, **kwargs))
        
        combined_trade_log = pd.DataFrame()
        for future in concurrent.futures.as_completed(futures):
            try:
                trade_log, pnl = future.result()
                if not trade_log.empty:
                    combined_trade_log = pd.concat([combined_trade_log, trade_log], ignore_index=True)
            except Exception as exc:
                print(f"An error occurred: {exc}")

    combined_trade_log.to_csv("trade_log.csv", index=False)
    summary_metrics = calculate_summary_metrics(combined_trade_log)
    summary_metrics.to_csv("summary_metrics.csv", index=False)
    return combined_trade_log, summary_metrics

def run_backtest():
    try:
        start_date = pd.to_datetime(start_date_var.get())
        end_date = pd.to_datetime(end_date_var.get())
        dates = pd.date_range(start_date, end_date, freq='D')

        kwargs = {
            "brick_size": int(brick_size_entry.get()),
            "brick_threshold": int(brick_threshold_entry.get()),
            "tp": int(take_profit_entry.get()),
            "sl": int(stop_loss_entry.get()),
            "fib_ma_period": int(fib_ma_period_entry.get()),
            "smooth_ma_period": int(smoothed_ma_period_entry.get()),
            "psar_acceleration": float(psar_acceleration_entry.get()),
            "psar_max_acceleration": float(psar_max_acceleration_entry.get()),
            "psar_step": float(psar_step_entry.get())
        }

        combined_trade_log, summary_metrics = backtest_strategy(Delta2Strategy, dates, **kwargs)
        display_trade_log(combined_trade_log)
        display_summary_metrics(summary_metrics)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def display_trade_log(trade_log):
    trade_log_listbox.delete(0, tk.END)  # Clear previous results
    for idx, row in trade_log.iterrows():
        trade_log_listbox.insert(tk.END, f"{row['timestamp']}, {row['action']}, {row['direction']}, {row['price']}, {row['quantity']}")

def display_summary_metrics(summary_metrics):
    summary_listbox.delete(0, tk.END)  # Clear previous results
    for idx, row in summary_metrics.iterrows():
        formatted_value = f"{row['Value']:.2f}"
        summary_listbox.insert(tk.END, f"{row['Metric']}: {formatted_value}")

# Fetch available dates from the database
available_dates = fetch_available_dates()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    root = tk.Tk()
    root.title("Backtest GUI")
    root.geometry("800x600")

    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    ttk.Label(frame, text="Algorithm Parameters", font=("Helvetica", 16)).grid(row=0, column=0, columnspan=2, pady=10)

    # Date selection using Combobox
    ttk.Label(frame, text="Start Date:").grid(row=1, column=0, sticky=tk.W)
    start_date_var = tk.StringVar()
    start_date_combobox = ttk.Combobox(frame, textvariable=start_date_var, values=available_dates)
    start_date_combobox.grid(row=1, column=1, sticky=(tk.W, tk.E))

    ttk.Label(frame, text="End Date:").grid(row=2, column=0, sticky=tk.W)
    end_date_var = tk.StringVar()
    end_date_combobox = ttk.Combobox(frame, textvariable=end_date_var, values=available_dates)
    end_date_combobox.grid(row=2, column=1, sticky=(tk.W, tk.E))

    # Other parameters with default values
    ttk.Label(frame, text="Renko Brick Size:").grid(row=3, column=0, sticky=tk.W)
    brick_size_entry = ttk.Entry(frame)
    brick_size_entry.insert(0, "3")
    brick_size_entry.grid(row=3, column=1, sticky=(tk.W, tk.E))

    ttk.Label(frame, text="Renko Brick Threshold:").grid(row=4, column=0, sticky=tk.W)
    brick_threshold_entry = ttk.Entry(frame)
    brick_threshold_entry.insert(0, "5")
    brick_threshold_entry.grid(row=4, column=1, sticky=(tk.W, tk.E))

    ttk.Label(frame, text="Take Profit:").grid(row=5, column=0, sticky=tk.W)
    take_profit_entry = ttk.Entry(frame)
    take_profit_entry.insert(0, "90")
    take_profit_entry.grid(row=5, column=1, sticky=(tk.W, tk.E))

    ttk.Label(frame, text="Stop Loss:").grid(row=6, column=0, sticky=tk.W)
    stop_loss_entry = ttk.Entry(frame)
    stop_loss_entry.insert(0, "90")
    stop_loss_entry.grid(row=6, column=1, sticky=(tk.W, tk.E))

    ttk.Label(frame, text="Fibonacci MA Period:").grid(row=7, column=0, sticky=tk.W)
    fib_ma_period_entry = ttk.Entry(frame)
    fib_ma_period_entry.insert(0, "9")
    fib_ma_period_entry.grid(row=7, column=1, sticky=(tk.W, tk.E))

    ttk.Label(frame, text="Smoothed MA Period:").grid(row=8, column=0, sticky=tk.W)
    smoothed_ma_period_entry = ttk.Entry(frame)
    smoothed_ma_period_entry.insert(0, "20")
    smoothed_ma_period_entry.grid(row=8, column=1, sticky=(tk.W, tk.E))

    ttk.Label(frame, text="PSAR Acceleration:").grid(row=9, column=0, sticky=tk.W)
    psar_acceleration_entry = ttk.Entry(frame)
    psar_acceleration_entry.insert(0, "0.0162")
    psar_acceleration_entry.grid(row=9, column=1, sticky=(tk.W, tk.E))

    ttk.Label(frame, text="PSAR Max Acceleration:").grid(row=10, column=0, sticky=tk.W)
    psar_max_acceleration_entry = ttk.Entry(frame)
    psar_max_acceleration_entry.insert(0, "0.162")
    psar_max_acceleration_entry.grid(row=10, column=1, sticky=(tk.W, tk.E))

    ttk.Label(frame, text="PSAR Step:").grid(row=11, column=0, sticky=tk.W)
    psar_step_entry = ttk.Entry(frame)
    psar_step_entry.insert(0, "0.02")
    psar_step_entry.grid(row=11, column=1, sticky=(tk.W, tk.E))

    run_button = ttk.Button(frame, text="Run Backtest", command=run_backtest)
    run_button.grid(row=12, column=0, columnspan=2, pady=10)

    ttk.Label(frame, text="Trade Log", font=("Helvetica", 14)).grid(row=13, column=0, pady=10, sticky=tk.W)
    trade_log_listbox = tk.Listbox(frame, height=10, width=50)
    trade_log_listbox.grid(row=14, column=0, pady=5, sticky=(tk.W, tk.E))

    ttk.Label(frame, text="Summary Metrics", font=("Helvetica", 14)).grid(row=13, column=1, pady=10, sticky=tk.W)
    summary_listbox = tk.Listbox(frame, height=10, width=50)
    summary_listbox.grid(row=14, column=1, pady=5, sticky=(tk.W, tk.E))

    # Adjust grid configurations to fit the new layout
    frame.grid_columnconfigure(0, weight=1)
    frame.grid_columnconfigure(1, weight=1)
    frame.grid_rowconfigure(14, weight=1)

    for widget in frame.winfo_children():
        widget.grid_configure(padx=5, pady=5)

    sv_ttk.set_theme("dark")

    root.mainloop()
