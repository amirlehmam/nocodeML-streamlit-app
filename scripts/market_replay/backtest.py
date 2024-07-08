import pandas as pd
import backtrader as bt
import h5py
import psycopg2
import tempfile
from datetime import datetime
from tqdm import tqdm
import logging
from renkodf import RenkoWS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def process_h5(file_path):
    with h5py.File(file_path, "r") as f:
        timestamps = [t.decode('utf-8') for t in f['L2/Timestamp'][:]]
        timestamps = pd.to_datetime(timestamps, errors='coerce')
        prices = f['L2/Price'][:].astype(float)
        df = pd.DataFrame({"datetime": timestamps, "close": prices})
        df.dropna(subset=["datetime"], inplace=True)
        
        df['open'] = df['close']
        df['high'] = df['close']
        df['low'] = df['close']
        
        return df.set_index('datetime')

def load_data_for_date(date):
    filename = date.strftime('%Y%m%d') + ".h5"
    file_path = load_h5_from_db(filename)
    if file_path:
        df = process_h5(file_path)
        return df
    return None

class RenkoPandas(bt.feeds.PandasData):
    lines = ('renko_open', 'renko_high', 'renko_low', 'renko_close')
    params = (
        ('open', 'renko_open'),
        ('high', 'renko_high'),
        ('low', 'renko_low'),
        ('close', 'renko_close'),
        ('datetime', None),
        ('timeframe', bt.TimeFrame.Ticks),
        ('compression', 1),
    )

class Delta2Strategy(bt.Strategy):
    params = (
        ('fib_weight_ma_period', 10),
        ('smoothing_simple_ma_period', 20),
    )

    def __init__(self):
        self.data_close = self.datas[0].renko_close
        self.data_high = self.datas[0].renko_high
        self.data_low = self.datas[0].renko_low

        # Calculate indicators
        self.zlema8 = bt.indicators.ExponentialMovingAverage(self.data_close, period=8)
        self.zlema62 = bt.indicators.ExponentialMovingAverage(self.data_close, period=62)
        self.atr = bt.indicators.AverageTrueRange(self.datas[0], period=14)
        self.psar = bt.indicators.ParabolicSAR(self.datas[0])

        # Fibonacci weighted MA and smoothed MA
        self.fib_weighted_ma = bt.indicators.WeightedMovingAverage(self.data_close, period=self.params.fib_weight_ma_period)
        self.smoothed_fib_weighted_ma = bt.indicators.SimpleMovingAverage(self.fib_weighted_ma, period=self.params.smoothing_simple_ma_period)

    def next(self):
        if self.position.size == 0:
            if self.fib_weighted_ma > self.smoothed_fib_weighted_ma:
                self.buy()
                self.entry_price = self.data_close[0]
                self.stop_loss = self.psar[0]
                self.take_profit = self.entry_price + 2 * self.atr[0]
            elif self.fib_weighted_ma < self.smoothed_fib_weighted_ma:
                self.sell()
                self.entry_price = self.data_close[0]
                self.stop_loss = self.psar[0]
                self.take_profit = self.entry_price - 2 * self.atr[0]
        else:
            if self.position.size > 0:
                if self.data_close[0] <= self.stop_loss or self.data_close[0] >= self.take_profit:
                    self.close()
            elif self.position.size < 0:
                if self.data_close[0] >= self.stop_loss or self.data_close[0] <= self.take_profit:
                    self.close()

def generate_renko_bars(data, brick_size=3, brick_threshold=5):
    renko_chart = RenkoWS(data.index[0].value // 10**6, data['close'].iloc[0], brick_size, brick_threshold)
    renko_bars = []
    for index, row in data.iterrows():
        renko_chart.add_prices(index.value // 10**6, row['close'])
    for brick in renko_chart.renko_animate('normal', max_len=10000, keep=5000):
        timestamp = int(brick[0]) // 1000  # Ensure timestamp is an integer before division
        renko_bars.append({
            'datetime': datetime.utcfromtimestamp(timestamp),
            'renko_open': float(brick[1]),
            'renko_high': float(brick[2]),
            'renko_low': float(brick[3]),
            'renko_close': float(brick[4])
        })
    return pd.DataFrame(renko_bars).set_index('datetime')

def run_backtest(start_date, end_date):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(Delta2Strategy)

    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    all_renko_data = pd.DataFrame()

    for date in tqdm(dates, desc="Loading data"):
        df = load_data_for_date(date)
        if df is not None:
            renko_data = generate_renko_bars(df)
            all_renko_data = pd.concat([all_renko_data, renko_data])

    data_feed = RenkoPandas(dataname=all_renko_data)
    cerebro.adddata(data_feed)
    cerebro.broker.setcash(300000)
    cerebro.addsizer(bt.sizers.FixedSize, stake=1)
    cerebro.broker.setcommission(commission=0.001)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

def main():
    available_dates = fetch_available_dates()
    
    print("Available dates:")
    for date in available_dates:
        print(date)
    
    start_date = input("Enter the start date (YYYYMMDD): ")
    end_date = input("Enter the end date (YYYYMMDD): ")

    run_backtest(start_date, end_date)

if __name__ == "__main__":
    main()
