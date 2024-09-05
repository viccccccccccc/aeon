import time

import numpy as np

from aeon.transformations.spikelet.Spikelet_Stat_decide_binsize_all import (Spikelet_Stat_decide_binsize_all)
from aeon.transformations.spikelet.sshist import sshist


def Spikelet_Stat_decide_binsize(D, Method="auto"):
    t_start = time.time()

    BinListInfo = {
        "bin": np.nan,
        "time": np.nan,
        "bin_names": np.nan,
        "time_all": np.nan,
    }

    if Method == "shimasaki":
        if np.max(D) != np.min(D):
            BinN_shimasaki, edges, Count = sshist(D)  # Placeholder for sshist implementation
            BinN = len(edges)
        else:
            Count, edges = np.histogram(D)
            BinN = len(edges)

    elif Method in ["fd", "scott", "sturges", "middle"]:
        Count, edges = np.histogram(D, bins=Method)
        BinN = len(edges)

    elif Method == "matlab":
        Count, edges = np.histogram(D)
        BinN = len(edges)

    elif Method == "auto":
        BinN_list, BinN_list_names, Timelist = Spikelet_Stat_decide_binsize_all(D)
        print(f"BinN_list: {(BinN_list)}, BinN_list_names: {(BinN_list_names)}, Timelist: {len(Timelist)}")
        print("")
        BinListInfo["bin"] = BinN_list
        BinListInfo["bin_names"] = BinN_list_names
        BinListInfo["time"] = Timelist

        BinN_all_sorted = np.sort(BinN_list)
        BinN = int(np.ceil(np.mean(BinN_all_sorted[1:-1])))

        N, edges = np.histogram(D, bins=BinN)

    elif Method == "min":
        BinN_list, BinN_list_names, Timelist = Spikelet_Stat_decide_binsize_all(D)
        BinListInfo["bin"] = BinN_list
        BinListInfo["bin_names"] = BinN_list_names
        BinListInfo["time"] = Timelist

        BinN = np.min(BinN_list)

    elif Method == "max":
        BinN_list, BinN_list_names, Timelist = Spikelet_Stat_decide_binsize_all(D)
        BinListInfo["bin"] = BinN_list
        BinListInfo["bin_names"] = BinN_list_names
        BinListInfo["time"] = Timelist

        BinN = np.max(BinN_list)

    BinListInfo["time_all"] = time.time() - t_start

    return BinN, edges, BinListInfo
