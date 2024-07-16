import pandas as pd
import numpy as np
import h5py
import psycopg2
import tempfile
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from qiskit_aer import Aer
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_machine_learning.algorithms import QSVC
from qiskit_algorithms import VQE
from qiskit.primitives import Sampler, Estimator
from qiskit_algorithms.optimizers import COBYLA
from renkodf import Renko
from tqdm import tqdm

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
    conn = psycopg2.connect(**db_credentials)
    return conn

def fetch_available_dates():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT filename FROM h5_files;')
    rows = cursor.fetchall()
    conn.close()
    
    dates = [row[0].replace('.h5', '') for row in rows]
    return sorted(dates)

def load_h5_from_db(filename):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('SELECT data FROM h5_files WHERE filename = %s;', (filename,))
    result = cursor.fetchone()
    conn.close()
    
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
        timestamps = [t.decode('utf-8') for t in f['L2/Timestamp'][:]]
        timestamps = pd.to_datetime(timestamps, errors='coerce')
        prices = f['L2/Price'][:].astype(float)
        df = pd.DataFrame({"datetime": timestamps, "close": prices})
        df.dropna(subset=["datetime"], inplace=True)
        
        print("Initial DataFrame:")
        print(df.head())

        renko = Renko(df, brick_size, brick_threshold=brick_threshold)
        renko_df = renko.renko_df()
        
        renko_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']  # Ensure correct column names
        print("Renko DataFrame:")
        print(renko_df.head())
        
        return renko_df

def load_data_for_date(date, brick_size, brick_threshold):
    filename = date.strftime('%Y%m%d') + ".h5"
    file_path = load_h5_from_db(filename)
    if file_path:
        df = process_h5(file_path, brick_size, brick_threshold)
        return df
    return None

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
        self._initialize_quantum_components()

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
        self.fib_weightings = self.fib_weightings[::-1]
        self.sum_of_fib_weights = np.sum(self.fib_weightings)

    def _calculate_fib_weighted_ma(self):
        close = self.data['Close']
        fib_ma = np.zeros(len(close))
        
        for i in range(len(close)):
            if i >= self.fib_weight_ma_period - 1:
                fib_ma[i] = np.dot(close.iloc[i - self.fib_weight_ma_period + 1:i + 1], self.fib_weightings) / self.sum_of_fib_weights
        
        self.data['fib_weighted_ma'] = fib_ma
        self.data['smoothed_fib_weighted_ma'] = self.data['fib_weighted_ma'].rolling(window=self.smoothing_simple_ma_period).mean()

    def calculate_zlema(self, period):
        return self.data['Close'].ewm(span=period).mean()

    def calculate_atr(self, period):
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']

        tr = pd.Series(np.maximum.reduce([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()]))
        atr = tr.rolling(window=period).mean()
        self.data['atr'] = atr
        print("ATR Indicator Head:")
        print(self.data[['atr']].head())
        return atr

    def calculate_psar(self):
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']

        af = 0.02
        max_af = 0.2
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
                    af = 0.02
                    reverse = True
            else:
                if high.iloc[i] > psar[i]:
                    bull = True
                    psar[i] = lp
                    hp = high.iloc[i]
                    af = 0.02
                    reverse = True

            if not reverse:
                if bull:
                    if high.iloc[i] > hp:
                        hp = high.iloc[i]
                        af = min(af + 0.02, max_af)
                    if i > 1 and low.iloc[i - 1] < psar[i]:
                        psar[i] = low.iloc[i - 1]
                    if i > 2 and low.iloc[i - 2] < psar[i]:
                        psar[i] = low.iloc[i - 2]
                else:
                    if low.iloc[i] < lp:
                        lp = low.iloc[i]
                        af = min(af + 0.02, max_af)
                    if i > 1 and high.iloc[i - 1] > psar[i]:
                        psar[i] = high.iloc[i - 1]
                    if i > 2 and high.iloc[i - 2] > psar[i]:
                        psar[i] = high.iloc[i - 2]

        self.data['PSAR'] = psar
        print("PSAR Indicator Head:")
        print(self.data[['PSAR']].head())
        return psar

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
        self.signals['buy_signal'][i] = self._cross_above(self.data['smoothed_fib_weighted_ma'], self.data['Close'], i)
        self.signals['sell_signal'][i] = self._cross_below(self.data['smoothed_fib_weighted_ma'], self.data['Close'], i)

    def _cross_above(self, series1, series2, i):
        return series1.iloc[i-1] < series2.iloc[i-1] and series1.iloc[i] > series2.iloc[i]

    def _cross_below(self, series1, series2, i):
        return series1.iloc[i-1] > series2.iloc[i-1] and series1.iloc[i] < series2.iloc[i]

    def _manage_positions(self, i):
        if self.position == 0:
            if self.signals['buy_signal'][i]:
                self._enter_position(i, 'long')
            elif self.signals['sell_signal'][i]:
                self._enter_position(i, 'short')
        elif self.position > 0:
            if self.signals['sell_signal'][i] or self.data['Close'].iloc[i] <= self.stop_loss.iloc[i] or self.data['Close'].iloc[i] >= self.take_profit.iloc[i]:
                self._exit_position(i)
        elif self.position < 0:
            if self.signals['buy_signal'][i] or self.data['Close'].iloc[i] >= self.stop_loss.iloc[i] or self.data['Close'].iloc[i] <= self.take_profit.iloc[i]:
                self._exit_position(i)

    def _enter_position(self, index, direction):
        if direction == 'long':
            self.position = self.default_quantity
            self.entry_price = self.data['Close'].iloc[index]
            self.stop_loss.iloc[index] = self.data['PSAR'].iloc[index] if 'PSAR' in self.data.columns else self.entry_price - 2 * (self.data['atr'].iloc[index] if 'atr' in self.data.columns else 0)
            self.take_profit.iloc[index] = self.entry_price + 2 * (self.data['atr'].iloc[index] if 'atr' in self.data.columns else 0)
        elif direction == 'short':
            self.position = -self.default_quantity
            self.entry_price = self.data['Close'].iloc[index]
            self.stop_loss.iloc[index] = self.data['PSAR'].iloc[index] if 'PSAR' in self.data.columns else self.entry_price + 2 * (self.data['atr'].iloc[index] if 'atr' in self.data.columns else 0)
            self.take_profit.iloc[index] = self.entry_price - 2 * (self.data['atr'].iloc[index] if 'atr' in self.data.columns else 0)
        self._log_entry(index, direction)

    def _exit_position(self, index):
        if self.position > 0:
            self.pnl += (self.data['Close'].iloc[index] - self.entry_price) * self.default_quantity
        elif self.position < 0:
            self.pnl += (self.entry_price - self.data['Close'].iloc[index]) * self.default_quantity
        self.position = 0
        self.entry_price = 0.0
        self.stop_loss.iloc[index] = 0.0
        self.take_profit.iloc[index] = 0.0
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

    def execute_strategy(self):
        if self.enable_fib_weight_ma_cross:
            self._calculate_fib_weighted_ma()

        for i in range(len(self.data)):
            self._bar_update(i)

    def detailed_analysis(self):
        trades_df = pd.DataFrame(self.trade_log)
        if trades_df.empty:
            print("No trades were made.")
            return

        total_trades = len(trades_df) // 2
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]

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

def backtest_strategy(strategy, market_replay_data, brick_size, brick_threshold):
    for date in tqdm(market_replay_data, desc="Backtesting"):
        df_ticks = load_data_for_date(date, brick_size, brick_threshold)
        if df_ticks is not None:
            strategy.data = df_ticks
            print("Backtesting DataFrame before execution:")
            print(strategy.data.head())
            print("DataFrame Columns:", strategy.data.columns)  # Check columns
            strategy.execute_strategy()
            print(f"PNL for {date}: {strategy.pnl}")
        else:
            print(f"No data for {date}")

    strategy.detailed_analysis()
    strategy.save_trade_log("trade_log.csv")

# Execute the strategy
if __name__ == "__main__":
    available_dates = fetch_available_dates()
    
    print("Available dates:")
    for date in available_dates:
        print(date)
    
    start_date = input("Enter the start date (YYYYMMDD): ")
    end_date = input("Enter the end date (YYYYMMDD): ")
    
    brick_size = float(input("Enter the brick size: "))
    brick_threshold = int(input("Enter the brick threshold: "))
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    initial_data = load_data_for_date(pd.to_datetime(start_date, format='%Y%m%d'), brick_size, brick_threshold)
    delta2_strategy = Delta2Strategy(data=initial_data)
    
    backtest_strategy(delta2_strategy, pd.to_datetime(dates, format='%Y%m%d'), brick_size, brick_threshold)
