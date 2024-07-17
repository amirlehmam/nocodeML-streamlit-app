import pandas as pd
import numpy as np
import h5py
import psycopg2
import tempfile
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from qiskit_aer import Aer
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_machine_learning.algorithms import QSVC
from qiskit_algorithms import VQE
from qiskit.primitives import Sampler, Estimator
from qiskit_algorithms.optimizers import COBYLA
from renkodf import Renko

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

def process_date(date, strategy_class, brick_size, brick_threshold):
    print(f"Processing date: {date}")
    data = load_data_for_date(date, brick_size, brick_threshold)  # Pass brick_size and brick_threshold
    if data is not None:
        temp_strategy = strategy_class(data)
        trade_log, pnl = temp_strategy.backtest()  # Use the correct backtest method
        return trade_log, pnl
    else:
        return None, None

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
        self.signals = {'buy_signal': np.zeros(len(data), dtype=bool), 'sell_signal': np.zeros(len(data), dtype=bool)}
        self.cooldown = pd.Series(np.zeros(len(data)), index=data.index)
        self.cooldown_period = 10  # Adjusted cooldown period to prevent rapid trades
        self.last_trade_time = None  # Track the last trade time
        self._initialize_parameters()
        self._initialize_fibonacci_weightings()
        self._calculate_psar()
        self._initialize_quantum_components()

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

    def _initialize_quantum_components(self):
        self.sampler = Sampler()
        self.estimator = Estimator()
        self.feature_map = ZZFeatureMap(feature_dimension=len(self.data.columns), reps=2)
        self.qsvc = None
        self.vqe = None
        self.variational_circuit = TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')

    def _initialize_qsvc(self):
        self.qsvc = QSVC(quantum_kernel=self.feature_map, estimator=self.estimator)

    def _initialize_vqe(self, optimizer=COBYLA()):
        self.vqe = VQE(ansatz=self.variational_circuit, optimizer=optimizer)

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
            self.stop_loss.iloc[index] = self.entry_price - 90  # Example value for stop loss
            self.take_profit.iloc[index] = self.entry_price + 350  # Example value for take profit
        elif direction == 'short':
            self.position = -self.default_quantity
            self.entry_price = self.data['Close'].iloc[index]
            self.stop_loss.iloc[index] = self.entry_price + 90  # Example value for stop loss
            self.take_profit.iloc[index] = self.entry_price - 350  # Example value for take profit
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

# Assuming the backtest_strategy function is calling process_date
def backtest_strategy(strategy_class, dates, brick_size, brick_threshold):
    all_trade_logs = []
    all_pnls = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for date in dates:
            futures.append(executor.submit(process_date, date, strategy_class, brick_size, brick_threshold))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                trade_log, pnl = future.result()
                if trade_log is not None and pnl is not None:
                    print(f"Trade Log: {trade_log}, PnL: {pnl}")
                    all_trade_logs.append(trade_log)
                    all_pnls.append(pnl)
                else:
                    print(f"No data returned for date: {future}")
            except Exception as exc:
                print(f"An error occurred: {exc}")
    
    # Combine all trade logs and export to CSV
    combined_trade_log = pd.concat(all_trade_logs, ignore_index=True)
    combined_trade_log.to_csv("trade_log.csv", index=False)
    print("Trade log exported to trade_log.csv")
    
    # Calculate and print summary metrics
    total_trades = len(combined_trade_log)
    winning_trades = combined_trade_log[combined_trade_log['direction'] == 'long']
    losing_trades = combined_trade_log[combined_trade_log['direction'] == 'short']
    total_gross_profit = winning_trades[winning_trades['action'] == 'exit']['price'].sum() * 20
    total_gross_loss = losing_trades[losing_trades['action'] == 'exit']['price'].sum() * 20
    net_profit_loss = sum(all_pnls)
    max_drawdown = combined_trade_log['price'].min() * 20
    percent_profitable = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
    ratio_avg_win_loss = total_gross_profit / abs(total_gross_loss) if total_gross_loss != 0 else 0
    profit_factor = total_gross_profit / abs(total_gross_loss) if total_gross_loss != 0 else 0
    sharpe_ratio = net_profit_loss / combined_trade_log['price'].std() if combined_trade_log['price'].std() != 0 else 0
    sortino_ratio = net_profit_loss / combined_trade_log['price'].min() if combined_trade_log['price'].min() != 0 else 0
    avg_trade = net_profit_loss / total_trades if total_trades > 0 else 0
    avg_win_trade = total_gross_profit / len(winning_trades) if len(winning_trades) > 0 else 0
    avg_lose_trade = total_gross_loss / len(losing_trades) if len(losing_trades) > 0 else 0

    summary_metrics = {
        'Gross Profit ($)': total_gross_profit,
        'Gross Loss ($)': total_gross_loss,
        'Net Profit/Loss ($)': net_profit_loss,
        'Total Trades': total_trades,
        'Winning Trades': len(winning_trades),
        'Losing Trades': len(losing_trades),
        'Max Drawdown ($)': max_drawdown,
        'Percent Profitable (%)': percent_profitable,
        'Ratio Avg Win / Avg Loss': ratio_avg_win_loss,
        'Profit Factor': profit_factor,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Average Trade ($)': avg_trade,
        'Average Winning Trade ($)': avg_win_trade,
        'Average Losing Trade ($)': avg_lose_trade
    }

    summary_df = pd.DataFrame(list(summary_metrics.items()), columns=['Metric', 'Value'])
    print("Summary Metrics:")
    print(summary_df)
    
    return combined_trade_log, all_pnls, summary_df

if __name__ == "__main__":
    dates = ['20240513', '20240514', '20240515']  # Example dates
    brick_size = 3  # Example brick size
    brick_threshold = 5  # Example brick threshold
    combined_trade_log, all_pnls, summary_df = backtest_strategy(Delta2Strategy, pd.to_datetime(dates, format='%Y%m%d'), brick_size, brick_threshold)
