# import tracemalloc
# import numpy as np
# import matplotlib.pyplot as plt

# from datetime import datetime

# from timessb.tssb.evaluation import covering
# from timessb.tssb.utils import visualize_time_series
# from claspy.segmentation import BinaryClaSPSegmentation
# from aeon.segmentation import ClaSPSegmenter

# def upscale_change_points(downsampled_cps, original_length, downsampled_length):
#     """
#     Upscale the change points from the downsampled time series to match the original time series.
    
#     Parameters:
#     downsampled_cps (list): Change points detected in the downsampled time series.
#     original_length (int): Length of the original time series.
#     downsampled_length (int): Length of the downsampled time series.
    
#     Returns:
#     list: Upscaled change points for the original time series.
#     """
#     upscaled_cps = []
#     for cp in downsampled_cps:
#         relative_cp = cp / downsampled_length  # Calculate relative position
#         upscaled_cp = round(relative_cp * original_length)  # Scale to original length
#         upscaled_cps.append(upscaled_cp)  # Add to the result list
#     return np.array(upscaled_cps)

# def runClaSP(ts, ts_name, cps, window_size, ts_down, downAlg, downsampled_length, pltshow):
#     org_len = len(ts)
#     new_len = len(ts_down)
#     tracemalloc.start()
#     start_time = datetime.now()

#     clasp = BinaryClaSPSegmentation(window_size=window_size)
#     if downAlg != "" and downAlg != "None":
#         found_cps_o = clasp.fit_predict(ts_down)
#         found_cps = upscale_change_points(found_cps_o, ts.shape[0], downsampled_length)  # found_cps muss angepasst werden, falls ein downsampling angewandt wurde
#     else:
#         found_cps = clasp.fit_predict(ts_down)

#     clasp_fit_time = (datetime.now() - start_time).total_seconds()
#     current, peak = tracemalloc.get_traced_memory()
#     tracemalloc.stop()
    
#     memory_usage = {
#         "current_memory": current,
#         "peak_memory": peak
#     }

#     score = covering({0: cps}, found_cps, ts.shape[0])
#     print(f"Time Series: {ts_name}, True Change Points: {cps}, Found Change Points: {found_cps.tolist()}, Score: {score}")

#     if True: 
#         visualize_time_series(ts_down, ts_name + ": Final CPs", np.round(cps * (new_len / org_len)).astype(int), found_cps_o)
#         plt.show()

#     return score, clasp_fit_time, memory_usage, found_cps

import tracemalloc
import numpy as np
import pandas as pd
import os
import psutil
from datetime import datetime

from timessb.tssb.utils import visualize_time_series
from timessb.tssb.evaluation import covering
from claspy.segmentation import BinaryClaSPSegmentation


def upscale_change_points(downsampled_cps, original_length, downsampled_length):
    """
    Upscale the change points from the downsampled time series to match the original time series.
    """
    upscaled_cps = [round(cp / downsampled_length * original_length) for cp in downsampled_cps]
    return np.array(upscaled_cps)

def measure_memory_usage(func, *args, **kwargs):
    """Führt eine Funktion aus und misst Peak Memory, Average Memory und Memory Footprint."""
    tracemalloc.start()
    process = psutil.Process()
    start_mem = process.memory_info().rss / (1024 ** 2)  # Initialer Speicherverbrauch in MB
    
    result = func(*args, **kwargs)  # Algorithmus ausführen
    
    current, peak = tracemalloc.get_traced_memory()
    peak_mem = peak / (1024 ** 2)  # Peak Memory in MB
    avg_mem = (current + peak) / (2 * 1024 ** 2)  # Durchschnittliche Nutzung in MB
    memory_footprint = peak_mem - start_mem  # Speicherbedarf pro Datensatzgröße
    
    tracemalloc.stop()
    
    return result, peak_mem, avg_mem, memory_footprint

def runClaSP(ts, ts_name, cps, window_size, ts_down, downAlg, downsampled_length, pltshow):
    org_len = len(ts)
    start_time = datetime.now()
    
    clasp = BinaryClaSPSegmentation(window_size=window_size)

    def clasp_fit():
        """Hier wird der Algorithmus ausgeführt"""
        if downAlg != "" and downAlg != "None":
            found_cps_o = clasp.fit_predict(ts_down)
            return upscale_change_points(found_cps_o, ts.shape[0], downsampled_length)
        else:
            return clasp.fit_predict(ts_down)
    
    # Speicherverbrauch messen
    found_cps, peak_memory, avg_memory, memory_footprint = measure_memory_usage(clasp_fit)
    clasp_fit_time = (datetime.now() - start_time).total_seconds()
    memory_usage = {"peak_memory": peak_memory, "average_memory": avg_memory, "memory_footprint": memory_footprint}
    
    # Speicherwerte in CSV speichern
    output_dir = r"C:\Users\Victor\Desktop\Uni\Bachelor\output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "memory_usage.csv")
    
    # Ergebnisse als Dictionary
    results = {
        "Time Series Name": ts_name,
        "Dataset Size": org_len,
        "Peak Memory (MB)": peak_memory,
        "Average Memory (MB)": avg_memory,
        "Memory Footprint (MB)": memory_footprint
    }
    
    # DataFrame erstellen und speichern
    df = pd.DataFrame([results])
    if not os.path.exists(output_file):
        df.to_csv(output_file, index=False)
    else:
        df.to_csv(output_file, mode="a", header=False, index=False)

    score = covering({0: cps}, found_cps, ts.shape[0])
    print(f"Time Series: {ts_name}, True Change Points: {cps}, Found Change Points: {found_cps.tolist()}, Score: {score}")
    
    # Optionale Visualisierung
    # if pltshow:
    #     visualize_time_series(ts_down, ts_name + ": Final CPs", np.round(ts.shape[0] / downsampled_length).astype(int))

    memory_usage = {
        "current_memory": 0,
        "peak_memory": 0
    }
    
    return score, clasp_fit_time, memory_usage, found_cps
