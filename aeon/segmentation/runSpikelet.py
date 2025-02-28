import pandas as pd
import tracemalloc
import matplotlib.pyplot as plt
from datetime import datetime

from timessb.tssb.utils import visualize_time_series
from aeon.transformations.spikelet.spikelet import motif_discovery_and_clasp

def runSpikelet(analysis_mode, ts, ts_name, cps, pltshow, mat=None, cot=None):
    tracemalloc.start()
    start_time = datetime.now()

    ts = motif_discovery_and_clasp(pd.Series(ts), mat, cot, ts_name, analysis_mode)
    
    transformation_time = (datetime.now() - start_time).total_seconds()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    memory_usage = {
        "current_memory": current,
        "peak_memory": peak
    }

    # if True: 
    #     fig, ax = visualize_time_series(ts, ts_name + ": after Spikelet-Transformation", cps)
    #     plt.show()

    return ts, transformation_time, memory_usage