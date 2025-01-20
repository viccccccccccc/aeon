import tracemalloc
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

from timessb.tssb.evaluation import covering
from timessb.tssb.utils import visualize_time_series
from claspy.segmentation import BinaryClaSPSegmentation
from aeon.segmentation import ClaSPSegmenter

def upscale_change_points(downsampled_cps, original_length, downsampled_length):
    """
    Upscale the change points from the downsampled time series to match the original time series.
    
    Parameters:
    downsampled_cps (list): Change points detected in the downsampled time series.
    original_length (int): Length of the original time series.
    downsampled_length (int): Length of the downsampled time series.
    
    Returns:
    list: Upscaled change points for the original time series.
    """
    upscaled_cps = []
    for cp in downsampled_cps:
        relative_cp = cp / downsampled_length  # Calculate relative position
        upscaled_cp = round(relative_cp * original_length)  # Scale to original length
        upscaled_cps.append(upscaled_cp)  # Add to the result list
    return np.array(upscaled_cps)

def runClaSP(ts, ts_name, cps, window_size, ts_down, downAlg, downsampled_length, pltshow):
    tracemalloc.start()
    start_time = datetime.now()

    clasp = BinaryClaSPSegmentation(window_size=window_size)
    if downAlg != "" and downAlg != "None":
        found_cps = clasp.fit_predict(ts_down)
        found_cps = upscale_change_points(found_cps, ts.shape[0], downsampled_length)  # found_cps muss angepasst werden, falls ein downsampling angewandt wurde
    else:
        found_cps = clasp.fit_predict(ts)

    clasp_fit_time = (datetime.now() - start_time).total_seconds()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    memory_usage = {
        "current_memory": current,
        "peak_memory": peak
    }

    score = covering({0: cps}, found_cps, ts.shape[0])
    print(f"Time Series: {ts_name}, True Change Points: {cps}, Found Change Points: {found_cps.tolist()}, Score: {score}")

    if pltshow: 
        visualize_time_series(ts, ts_name + ": Finale Bestimmung der CP", found_cps)
        plt.show()

    return score, clasp_fit_time, memory_usage, found_cps