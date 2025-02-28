import time
import pdb

import numpy as np
from aeon.transformations.spikelet.sshist import sshist
from aeon.transformations.spikelet.histx import histx
from aeon.transformations.spikelet.histcounts import histcounts
from aeon.transformations.spikelet.histcounts2 import histcounts2

def Spikelet_Stat_decide_binsize_all(D):
    t_start = time.time()
    if np.max(D) != np.min(D):
        BinN_shimasaki, edges_mid_shimasaki, Count_shimasaki = sshist(D)
        
    else:
        Count_shimasaki, edges_mid_shimasaki = histcounts2(D)
        BinN_shimasaki = len(edges_mid_shimasaki)
    time_shimasaki = time.time() - t_start

    t_start = time.time()
    Count_fd, edges_mid_fd = histx(D, 'fd')
    time_fd = time.time() - t_start

    t_start = time.time()
    Count_scott, edges_mid_scott = histx(D, 'scott')
    time_scott = time.time() - t_start

    t_start = time.time()
    Count_sturges, edges_mid_sturges = histx(D, 'sturges')
    time_sturges = time.time() - t_start

    t_start = time.time()
    Count_middle, edges_mid_middle = histx(D, 'middle')
    time_middle = time.time() - t_start

    t_start = time.time()
    Count_matlab, edges_matlab = histcounts(D)
    time_matlab = time.time() - t_start

    BinN_list_names = ["shimasaki", "fd", "scott", "sturges", "matlab"]
    BinN_list = [
        len(edges_mid_shimasaki),
        len(edges_mid_fd),
        len(edges_mid_scott),
        len(edges_mid_sturges),
        len(Count_matlab)
    ]
    Time_list = [
        time_shimasaki,
        time_fd,
        time_scott,
        time_sturges,
        time_middle,
        time_matlab
    ]

    return BinN_list, BinN_list_names, Time_list

