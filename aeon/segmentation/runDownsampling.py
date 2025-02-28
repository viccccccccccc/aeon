import tracemalloc
import math
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from claspy.window_size import suss
from timessb.tssb.utils import visualize_time_series

################################


################################

def downsample_every_nth(ts: pd.Series, n: int) -> pd.Series:
    """
    Downsample the time series by keeping every nth value.

    Parameters:
    ts (pd.Series): The input time series.
    n (int): Interval for keeping values (e.g., every nth value).

    Returns:
    pd.Series: Downsampled time series.
    """
    if n <= 0:
        raise ValueError("n must be greater than 0")
    
    if isinstance(ts, pd.Series):
        # For pandas Series, drop every nth value
        downsampled_ts = ts.drop(ts.index[::n])
        #downsampled_ts = ts.iloc[::n]
    elif isinstance(ts, np.ndarray):
        # For numpy array, use slicing to exclude every nth value
        downsampled_ts = np.delete(ts, np.arange(0, len(ts), n))
        #downsampled_ts = ts[::n]
    else:
        raise TypeError("Input must be a pandas Series or numpy ndarray")
    return downsampled_ts

def downsample_every_nth2(ts: pd.Series, n: int) -> pd.Series:
    """
    Downsample the time series by replacing every n-window with its average.

    Parameters:
    ts (pd.Series): The input time series.
    n (int): Interval for computing the average (window size).

    Returns:
    pd.Series: Downsampled time series with averaged values.
    """
    if n <= 0:
        raise ValueError("n must be greater than 0")

    if isinstance(ts, pd.Series):
        # Compute rolling mean with window size n and take every nth value
        downsampled_ts = ts.rolling(window=n, min_periods=1).mean()[::n]
        downsampled_ts.index = np.arange(len(downsampled_ts))  # Reset index for continuity

    elif isinstance(ts, np.ndarray):
        # Convert to pandas Series to use rolling mean
        ts_series = pd.Series(ts)
        downsampled_ts = ts_series.rolling(window=n, min_periods=1).mean()[::n].to_numpy()

    else:
        raise TypeError("Input must be a pandas Series or numpy ndarray")

    return downsampled_ts

def downsample_extrema(ts: np.ndarray, window_size: int, threshold: float = 1e-3) -> np.ndarray:
    """
    Downsample a time series by selecting extrema (min, max, mean) in each window.

    Parameters:
    -----------
    ts : np.ndarray
        The input time series as a numpy array.
    window_size : int
        The size of the window used for downsampling.

    Returns:
    --------
    np.ndarray:
        Downsampled time series as a numpy array.
    """
    if not isinstance(ts, np.ndarray):
        raise TypeError("Input time series must be a numpy array.")

    downsampled_ts = []

    for i in range(0, len(ts), window_size):
        window = ts[i:i + window_size]
        if len(window) > 0:
            # Append min, max, and mean values from the window
            if np.std(window) < threshold:
                # If constant, add the mean value of the window
                downsampled_ts.extend([window.mean(), window.mean()])
            else:
                # Otherwise, find min and max positions
                min_idx = np.argmin(window)
                max_idx = np.argmax(window)

                # Append extrema in the order they appear
                if min_idx < max_idx:
                    downsampled_ts.extend([window[min_idx], window[max_idx]])
                else:
                    downsampled_ts.extend([window[max_idx], window[min_idx]])

    return np.array(downsampled_ts)

def downsample_extrema2(ts: np.ndarray, window_size: int, threshold: float = 1e-3) -> np.ndarray:
    """
    Downsample a time series by selecting extrema (min, max, mean) in each window.

    Parameters:
    -----------
    ts : np.ndarray
        The input time series as a numpy array.
    window_size : int
        The size of the window used for downsampling.

    Returns:
    --------
    np.ndarray:
        Downsampled time series as a numpy array.
    """
    if not isinstance(ts, np.ndarray):
        raise TypeError("Input time series must be a numpy array.")

    downsampled_ts = []

    for i in range(0, len(ts), window_size):
        window = ts[i:i + window_size]
        if len(window) > 0:
            # Append min, max, and mean values from the window
            if np.std(window) < threshold:
                # If constant, add the mean value of the window
                downsampled_ts.extend([window.mean(), window.mean(), window.mean()])
            else:
                # Otherwise, find min and max positions
                min_idx = np.argmin(window)
                max_idx = np.argmax(window)

                # Append extrema in the order they appear
                if min_idx < max_idx:
                    downsampled_ts.extend([window[min_idx], window.mean(), window[max_idx]])
                else:
                    downsampled_ts.extend([window[max_idx], window.mean(), window[min_idx]])

    return np.array(downsampled_ts)

def runDownsampling(ts, ts_name, downAlgName, downAlgParam, window_size, pltshow, cps):
    org_len = len(ts)
    tracemalloc.start()
    start_time = datetime.now()

    if downAlgName == "nth":
        ts = downsample_every_nth(ts, downAlgParam)
    elif downAlgName == "Extrema":
        ts = downsample_extrema(ts, downAlgParam)
    elif downAlgName == "Extrema2":
        ts = downsample_extrema2(ts, downAlgParam)
    elif downAlgName == "nth2":
        ts = downsample_every_nth2(ts, downAlgParam)
    else:
        print("Kein Downsampling wird ausgeführt.")
        return ts, window_size, 0
    
    elapsed_time = (datetime.now() - start_time).total_seconds()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    memory_usage = {
        "current_memory": current,
        "peak_memory": peak
    }

    print("Länge der downsampled Time Series: ", len(ts))
    downsampled_length = len(ts)
    window_size = suss(ts)

    new_len = len(ts)

    # if True: 
    #     fig, ax = visualize_time_series(ts, ts_name + ": after Downsampling", np.round(cps * (new_len / org_len)).astype(int))
    #     plt.show()

    return ts, window_size, downsampled_length, elapsed_time, memory_usage