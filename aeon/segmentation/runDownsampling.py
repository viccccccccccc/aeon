import tracemalloc
import math
import pandas as pd
import numpy as np
from datetime import datetime

from claspy.window_size import suss
from timessb.tssb.utils import visualize_time_series

################################

nth_value = 4       # Hyperparameter, für nth downsampling. Entscheidet, dass der nte wert immer aussortiert wird

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

    window_size = math.floor(window_size * 0.45)
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

def runDownsampling(ts, ts_name, downAlg, window_size, pltshow):
    tracemalloc.start()
    start_time = datetime.now()

    if downAlg == "nth":
        ts = downsample_every_nth(ts, nth_value)
    elif downAlg == "Extrema":
        ts = downsample_extrema(ts, window_size)
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

    if pltshow: 
        visualize_time_series(ts, ts_name + ": Datensatz nach Downsampling")

    return ts, window_size, downsampled_length, elapsed_time, memory_usage