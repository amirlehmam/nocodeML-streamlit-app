import gc
import numpy as np
import pandas as pd
import h5py
import mplfinance as mpf
from tqdm import tqdm
from matplotlib import animation

def add_brick_loop(rsd, df, i, renko_multiply, current_direction, current_n_bricks, wick_min_i, wick_max_i, volume_i, custom_columns, brick_size, brick_threshold):
    last_price = rsd["price"][-1]
    renko_price = last_price + (current_direction * renko_multiply * brick_size)
    wick = wick_min_i if current_n_bricks > 0 else wick_max_i

    to_add = [i, df["datetime"].iat[i], renko_price, current_direction, wick, volume_i]
    for name, add in zip(list(rsd.keys()), to_add):
        rsd[name].append(add)
    if custom_columns is not None:
        for name in custom_columns:
            rsd[name].append(df[name].iat[i])

    volume_i = 1
    wick_min_i = renko_price if current_n_bricks > 0 else wick_min_i
    wick_max_i = renko_price if current_n_bricks < 0 else wick_max_i

    return rsd, wick_min_i, wick_max_i, volume_i

def add_prices(rsd, df, brick_size, brick_threshold, custom_columns, show_progress):
    df_len = len(df)
    wick_min_i = df["close"].iat[0]
    wick_max_i = df["close"].iat[0]
    volume_i = 1

    for i in tqdm(range(1, df_len), desc="Calculating Renko Bars", disable=not show_progress):
        df_close = df["close"].iat[i]
        wick_min_i = min(df_close, wick_min_i)
        wick_max_i = max(df_close, wick_max_i)
        volume_i += 1

        last_price = rsd["price"][-1]
        current_n_bricks = (df_close - last_price) / brick_size
        current_direction = np.sign(current_n_bricks)
        if abs(current_n_bricks) < brick_threshold:
            continue
        last_direction = rsd["direction"][-1]
        is_same_direction = ((current_direction > 0 and last_direction >= 0)
                             or (current_direction < 0 and last_direction <= 0))

        total_same_bricks = current_n_bricks if is_same_direction else 0
        if not is_same_direction and abs(current_n_bricks) >= 2:
            rsd, wick_min_i, wick_max_i, volume_i = add_brick_loop(
                rsd, df, i, 2, current_direction, current_n_bricks, wick_min_i, wick_max_i, volume_i, custom_columns, brick_size, brick_threshold
            )
            total_same_bricks = current_n_bricks - 2 * current_direction

        for not_in_use in range(abs(int(total_same_bricks))):
            rsd, wick_min_i, wick_max_i, volume_i = add_brick_loop(
                rsd, df, i, 1, current_direction, current_n_bricks, wick_min_i, wick_max_i, volume_i, custom_columns, brick_size, brick_threshold
            )

    return rsd, wick_min_i, wick_max_i, volume_i

class Renko:
    def __init__(self, df: pd.DataFrame, brick_size: float, brick_threshold: int, add_columns: list = None, show_progress: bool = False):
        if brick_size is None or brick_size <= 0:
            raise ValueError("brick_size cannot be 'None' or '<= 0'")
        if 'datetime' not in df.columns:
            df["datetime"] = df.index
        if 'close' not in df.columns:
            raise ValueError("Column 'close' doesn't exist!")
        if add_columns is not None:
            if not set(add_columns).issubset(df.columns):
                raise ValueError(f"One or more of {add_columns} columns don't exist!")

        self._brick_size = brick_size
        self._brick_threshold = brick_threshold
        self._custom_columns = add_columns
        self._show_progress = show_progress

        first_close = df["close"].iat[0]
        initial_price = (first_close // brick_size) * brick_size
        self._rsd = {
            "origin_index": [0],
            "date": [df["datetime"].iat[0]],
            "price": [initial_price],
            "direction": [0],
            "wick": [initial_price],
            "volume": [1],
        }
        if add_columns is not None:
            for name in add_columns:
                self._rsd.update({
                    name: [df[name].iat[0]]
                })

        self._rsd, self._wick_min_i, self._wick_max_i, self._volume_i = add_prices(
            self._rsd, df, brick_size, brick_threshold, add_columns, show_progress
        )

    def renko_df(self, mode: str = "wicks"):
        _MODE_dict = ['normal', 'wicks', 'nongap', 'reverse-wicks', 'reverse-nongap', 'fake-r-wicks', 'fake-r-nongap']
        if mode not in _MODE_dict:
            raise ValueError(f"Only {_MODE_dict} options are valid.")

        dates = self._rsd["date"]
        prices = self._rsd["price"]
        directions = self._rsd["direction"]
        wicks = self._rsd["wick"]
        volumes = self._rsd["volume"]
        indexes = list(range(len(prices)))
        brick_size = self._brick_size

        df_dict = {
            "datetime": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }
        if self._custom_columns is not None:
            for name in self._custom_columns:
                df_dict.update({
                    name: []
                })

        reverse_rule = mode in ["normal", "wicks", "reverse-wicks", "fake-r-wicks"]
        fake_reverse_rule = mode in ["fake-r-nongap", "fake-r-wicks"]
        same_direction_rule = mode in ["wicks", "nongap"]

        prev_direction = 0
        prev_close = 0
        prev_close_up = 0
        prev_close_down = 0
        for price, direction, date, wick, volume, index in zip(prices, directions, dates, wicks, volumes, indexes):
            if direction != 0:
                df_dict["datetime"].append(date)
                df_dict["close"].append(price)
                df_dict["volume"].append(volume)

            if direction == 1.0:
                df_dict["high"].append(price)
                if self._custom_columns is not None:
                    for name in self._custom_columns:
                        df_dict[name].append(self._rsd[name][index])
                if prev_direction == 1:
                    df_dict["open"].append(wick if mode == "nongap" else prev_close_up)
                    df_dict["low"].append(wick if same_direction_rule else prev_close_up)
                else:
                    if reverse_rule:
                        df_dict["open"].append(prev_close + brick_size)
                    elif mode == "fake-r-nongap":
                        df_dict["open"].append(prev_close_down)
                    else:
                        df_dict["open"].append(wick)

                    if mode == "normal":
                        df_dict["low"].append(prev_close + brick_size)
                    elif fake_reverse_rule:
                        df_dict["low"].append(prev_close_down)
                    else:
                        df_dict["low"].append(wick)
                prev_close_up = price
            elif direction == -1.0:
                df_dict["low"].append(price)
                if self._custom_columns is not None:
                    for name in self._custom_columns:
                        df_dict[name].append(self._rsd[name][index])
                if prev_direction == -1:
                    df_dict["open"].append(wick if mode == "nongap" else prev_close_down)
                    df_dict["high"].append(wick if same_direction_rule else prev_close_down)
                else:
                    if reverse_rule:
                        df_dict["open"].append(prev_close - brick_size)
                    elif mode == "fake-r-nongap":
                        df_dict["open"].append(prev_close_up)
                    else:
                        df_dict["open"].append(wick)

                    if mode == "normal":
                        df_dict["high"].append(prev_close - brick_size)
                    elif fake_reverse_rule:
                        df_dict["high"].append(prev_close_up)
                    else:
                        df_dict["high"].append(wick)
                prev_close_down = price
            else:
                df_dict["datetime"].append(np.NaN)
                df_dict["low"].append(np.NaN)
                df_dict["close"].append(np.NaN)
                df_dict["high"].append(np.NaN)
                df_dict["open"].append(np.NaN)
                df_dict["volume"].append(np.NaN)
                if self._custom_columns is not None:
                    for name in self._custom_columns:
                        df_dict[name].append(np.NaN)

            prev_direction = direction
            prev_close = price

        df = pd.DataFrame(df_dict)
        df.drop(df.head(2).index, inplace=True)
        df.index = pd.DatetimeIndex(df["datetime"])
        df.drop(columns=['datetime'], inplace=True)

        return df

    def renko_animate(self, mode='wicks', max_len=100, keep=50):
        renko_df = self.renko_df(mode)
        renko_df = renko_df.iloc[-max_len:]  # Keep only the last max_len rows
        return renko_df

def process_l2_data(hdf5_file, brick_size, brick_threshold):
    with h5py.File(hdf5_file, 'r') as f:
        dataset_name = "L2/Timestamp"
        timestamps = f[dataset_name][:]
        prices = f["L2/Price"][:]
        volumes = f["L2/Volume"][:]

    # Convert byte strings to normal strings before converting to datetime
    timestamps = [ts.decode('utf-8') for ts in timestamps]
    timestamps = pd.to_datetime(timestamps)

    data = {
        "datetime": timestamps,
        "close": prices.astype(float)
    }
    df_l2 = pd.DataFrame(data)

    renko_chart = Renko(df_l2, brick_size, brick_threshold, show_progress=True)
    renko_df = renko_chart.renko_df()

    try:
        renko_df.to_hdf('renko_bars.h5', key='renko', mode='w')
    except ImportError as e:
        print("Error: The 'tables' package is required to save the dataframe to an HDF5 file.")
        print("Please install it using: pip install tables")
        return renko_chart

    print("Renko bars generated and saved to HDF5")
    return renko_chart

def animate(ival, renko_chart, df_ticks, ax1, ax2):
    if ival >= len(df_ticks):
        print('No more data to plot')
        return

    timestamp = df_ticks['datetime'].iat[ival]
    price = df_ticks['close'].iat[ival]

    renko_chart._rsd, renko_chart._wick_min_i, renko_chart._wick_max_i, renko_chart._volume_i = add_brick_loop(
        renko_chart._rsd, df_ticks, ival, 1, np.sign(price - renko_chart._rsd["price"][-1]), (price - renko_chart._rsd["price"][-1]) / renko_chart._brick_size,
        renko_chart._wick_min_i, renko_chart._wick_max_i, renko_chart._volume_i, renko_chart._custom_columns, renko_chart._brick_size, renko_chart._brick_threshold
    )

    df_wicks = renko_chart.renko_animate('wicks', max_len=100, keep=50)

    ax1.clear()
    ax2.clear()

    mpf.plot(df_wicks, type='candle', ax=ax1, volume=ax2, axtitle='renko: wicks')

def main():
    hdf5_file = 'C:/Users/Administrator/Desktop/nocodeML-streamlit-app/scripts/market_replay/data/market_replay_data.h5'
    brick_size = 5
    brick_threshold = 5
    renko_chart = process_l2_data(hdf5_file, brick_size, brick_threshold)

    # Load tick data for animation
    df_ticks = renko_chart._rsd['date']

    fig, axes = mpf.plot(renko_chart.renko_df(), returnfig=True, volume=True,
                         figsize=(11, 8), panel_ratios=(2, 1),
                         title='\nRenko Animation', type='candle', style='charles')
    ax1 = axes[0]
    ax2 = axes[2]

    ani = animation.FuncAnimation(fig, animate, fargs=(renko_chart, df_ticks, ax1, ax2), interval=80)
    mpf.show()

if __name__ == "__main__":
    main()
