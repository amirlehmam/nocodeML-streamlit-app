import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

_MODE_dict = ['normal', 'wicks', 'nongap', 'reverse-wicks', 'reverse-nongap', 'fake-r-wicks', 'fake-r-nongap']

def add_brick_loop(rsd, df, i, renko_multiply, current_direction, current_n_bricks, wick_min_i, wick_max_i, volume_i, custom_columns, brick_size):
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

def add_prices(rsd, df, i, brick_size, custom_columns):
    df_close = df["close"].iat[i]
    wick_min_i = df_close if df_close < rsd["wick"][-1] else rsd["wick"][-1]
    wick_max_i = df_close if df_close > rsd["wick"][-1] else rsd["wick"][-1]
    volume_i = rsd["volume"][-1] + 1

    last_price = rsd["price"][-1]
    current_n_bricks = (df_close - last_price) / brick_size
    current_direction = np.sign(current_n_bricks)
    if current_direction == 0:
        return rsd, wick_min_i, wick_max_i, volume_i
    last_direction = rsd["direction"][-1]
    is_same_direction = ((current_direction > 0 and last_direction >= 0)
                         or (current_direction < 0 and last_direction <= 0))

    total_same_bricks = current_n_bricks if is_same_direction else 0
    if not is_same_direction and abs(current_n_bricks) >= 2:
        rsd, wick_min_i, wick_max_i, volume_i = add_brick_loop(rsd, df, i, 2, current_direction, current_n_bricks, wick_min_i, wick_max_i, volume_i, custom_columns, brick_size)
        total_same_bricks = current_n_bricks - 2 * current_direction

    for not_in_use in range(abs(int(total_same_bricks))):
        rsd, wick_min_i, wick_max_i, volume_i = add_brick_loop(rsd, df, i, 1, current_direction, current_n_bricks, wick_min_i, wick_max_i, volume_i, custom_columns, brick_size)

    return rsd, wick_min_i, wick_max_i, volume_i

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

        for i in tqdm(range(1, self._df_len), desc="Calculating Renko Bars"):
            self._rsd, self._wick_min_i, self._wick_max_i, self._volume_i = add_prices(self._rsd, df, i, brick_size, self._custom_columns)

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

    def plot(self, speed=0.1):
        prices = self._rsd['price']
        directions = self._rsd['direction']
        dates = self._rsd['date']
        brick_size = self._brick_size

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

def process_l2_data(file_path, brick_size):
    with h5py.File(file_path, "r") as f:
        def print_attrs(name, obj):
            print(name)
        f.visititems(print_attrs)

    dataset_names = ["L2/MarketDataType", "L2/MarketMaker", "L2/Offset", "L2/Operation", "L2/Position", "L2/Price", "L2/RecordType", "L2/Timestamp", "L2/Volume"]

    with h5py.File(file_path, "r") as f:
        l2_data = {name.split('/')[-1]: f[name][:] for name in dataset_names}

    df_l2 = pd.DataFrame(l2_data)
    df_l2["datetime"] = pd.to_datetime(df_l2["Timestamp"].astype(str))
    df_l2["close"] = df_l2["Price"].astype(float)

    renko_chart = Renko(df_l2, brick_size)
    renko_df = renko_chart.renko_df()
    print(f"Renko DataFrame for Playback:\n{renko_df.head()}")

    renko_chart.plot()

if __name__ == "__main__":
    hdf5_file = 'C:/Users/Administrator/Desktop/nocodeML-streamlit-app/scripts/market_replay/data/market_replay_data.h5'
    brick_size = 0.03
    process_l2_data(hdf5_file, brick_size)
