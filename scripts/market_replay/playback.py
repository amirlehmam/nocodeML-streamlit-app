import gc
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

_MODE_dict = ['normal', 'wicks', 'nongap', 'reverse-wicks', 'reverse-nongap', 'fake-r-wicks', 'fake-r-nongap']

class Renko:
    def __init__(self, df: pd.DataFrame, brick_size: float, add_columns: list = None, show_progress: bool = False):
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
        self._custom_columns = add_columns
        self._df_len = len(df["close"])
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

        self._wick_min_i = initial_price
        self._wick_max_i = initial_price
        self._volume_i = 1

        for i in range(1, self._df_len):
            self._add_prices(i, df)

    def _add_prices(self, i, df):
        df_close = df["close"].iat[i]
        self._wick_min_i = min(df_close, self._wick_min_i)
        self._wick_max_i = max(df_close, self._wick_max_i)
        self._volume_i += 1

        last_price = self._rsd["price"][-1]
        current_n_bricks = (df_close - last_price) / self._brick_size
        current_direction = np.sign(current_n_bricks)
        if current_direction == 0:
            return
        last_direction = self._rsd["direction"][-1]
        is_same_direction = ((current_direction > 0 and last_direction >= 0) or (current_direction < 0 and last_direction <= 0))

        total_same_bricks = current_n_bricks if is_same_direction else 0
        if not is_same_direction and abs(current_n_bricks) >= 2:
            self._add_brick_loop(df, i, 2, current_direction, current_n_bricks)
            total_same_bricks = current_n_bricks - 2 * current_direction

        for _ in range(abs(int(total_same_bricks))):
            self._add_brick_loop(df, i, 1, current_direction, current_n_bricks)

        if self._show_progress:
            print(f"\r {round(float((i + 1) / self._df_len * 100), 2)}%", end='')

    def _add_brick_loop(self, df, i, renko_multiply, current_direction, current_n_bricks):
        last_price = self._rsd["price"][-1]
        renko_price = last_price + (current_direction * renko_multiply * self._brick_size)
        wick = self._wick_min_i if current_n_bricks > 0 else self._wick_max_i

        to_add = [i, df["datetime"].iat[i], renko_price, current_direction, wick, self._volume_i]
        for name, add in zip(list(self._rsd.keys()), to_add):
            self._rsd[name].append(add)
        if self._custom_columns is not None:
            for name in self._custom_columns:
                self._rsd[name].append(df[name].iat[i])

        self._volume_i = 1
        self._wick_min_i = renko_price if current_n_bricks > 0 else self._wick_min_i
        self._wick_max_i = renko_price if current_n_bricks < 0 else self._wick_max_i

    def renko_df(self, mode: str = "wicks"):
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

def process_l2_data(hdf5_file, brick_size):
    with h5py.File(hdf5_file, 'r') as f:
        l2_price = f['L2/Price'][:]
        l2_timestamp = f['L2/Timestamp'][:]

    df_l2 = pd.DataFrame({
        'Timestamp': pd.to_datetime(l2_timestamp.astype(str)),
        'close': pd.to_numeric(l2_price, errors='coerce')
    })

    df_l2 = df_l2.dropna(subset=['close'])
    df_l2 = df_l2.groupby('Timestamp').agg({'close': 'mean'}).reset_index()

    renko_chart = Renko(df_l2, brick_size)
    renko_df = renko_chart.renko_df()

    with h5py.File(hdf5_file, 'a') as f:
        if 'Renko' in f:
            del f['Renko']
        renko_group = f.create_group('Renko')
        renko_group.create_dataset('Timestamp', data=renko_df.index.astype(str).values.astype('S'))
        renko_group.create_dataset('Renko_Close', data=renko_df['close'].values)

    print("Renko bars generated and saved to HDF5")

def playback_renko(renko_df, speed=1):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    timestamps = renko_df.index
    renko_closes = renko_df['close']

    l, = plt.plot(timestamps[:1], renko_closes[:1], '-')

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Time', 0, len(timestamps) - 1, valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        l.set_data(timestamps[:idx+1], renko_closes[:idx+1])
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

if __name__ == "__main__":
    hdf5_file = 'C:/Users/Administrator/Desktop/nocodeML-streamlit-app/scripts/market_replay/data/market_replay_data.h5'
    brick_size = 30
    process_l2_data(hdf5_file, brick_size)

    with h5py.File(hdf5_file, 'r') as f:
        renko_timestamp = f['Renko/Timestamp'][:]
        renko_close = f['Renko/Renko_Close'][:]

    renko_df = pd.DataFrame({
        'Timestamp': pd.to_datetime(renko_timestamp.astype(str)),
        'close': renko_close
    })

    renko_df.set_index('Timestamp', inplace=True)

    print("Renko DataFrame for Playback:")
    print(renko_df.head(20))

    playback_renko(renko_df)
