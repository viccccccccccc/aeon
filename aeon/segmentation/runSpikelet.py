import pandas as pd
import tracemalloc
from datetime import datetime

from timessb.tssb.utils import visualize_time_series
from aeon.transformations.spikelet.spikelet import motif_discovery_and_clasp

def runSpikelet(ts, ts_name, cps, pltshow, mat=None, cot=None):
    tracemalloc.start()
    start_time = datetime.now()

    ts = motif_discovery_and_clasp(pd.Series(ts), mat, cot)
    
    transformation_time = (datetime.now() - start_time).total_seconds()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    memory_usage = {
        "current_memory": current,
        "peak_memory": peak
    }

    if pltshow: 
        visualize_time_series(ts, ts_name + ": Datensatz nach Spikelet-Transformation")

    return ts, transformation_time, memory_usage