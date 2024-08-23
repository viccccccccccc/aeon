import time

import numpy as np

def Spikelet_Stat_decide_binsize_all(D):
    start_time = time.time()
    print("länge von D ist: ", len(D))
    # If all values in D are not the same
    if np.max(D) != np.min(D):
        BinN_shimasaki, edges_mid_shimasaki, Count_shimasaki = sshist(D)
    else:
        Count_shimasaki, edges_mid_shimasaki = np.histogram(D)

    time_shimasaki = time.time() - start_time

    start_time = time.time()
    Count_fd, edges_mid_fd = np.histogram(D, bins="fd")
    time_fd = time.time() - start_time

    start_time = time.time()
    Count_scott, edges_mid_scott = np.histogram(D, bins="scott")
    time_scott = time.time() - start_time

    start_time = time.time()
    Count_sturges, edges_mid_sturges = np.histogram(D, bins="sturges")
    time_sturges = time.time() - start_time

    start_time = time.time()
    Count_middle, edges_mid_middle = np.histogram(
        D, bins="auto"
    )  # Equivalent to 'middle'
    time_middle = time.time() - start_time

    start_time = time.time()
    Count_matlab, edges_matlab = np.histogram(D)
    time_matlab = time.time() - start_time

    BinN_list_names = ["shimasaki", "fd", "scott", "sturges", "matlab"]
    BinN_list = [
        len(edges_mid_shimasaki) - 1,
        len(edges_mid_fd) - 1,
        len(edges_mid_scott) - 1,
        len(edges_mid_sturges) - 1,
        len(edges_matlab) - 1,
    ]
    Time_list = [
        time_shimasaki,
        time_fd,
        time_scott,
        time_sturges,
        time_middle,
        time_matlab,
    ]

    return BinN_list, BinN_list_names, Time_list


"""def sshist(x, N=None):
    # Here, a basic histogram calculation is used as a placeholder
    
    count, edges = np.histogram(x, bins=N)
    optN = len(edges) - 1
    return optN, edges, count"""

def sshist(x, N=None):
    """
    Function `sshist` returns the optimal number of bins in a histogram
    used for density estimation.
    """
    # Flatten x and compute basic statistics
    x = np.ravel(x)
    x_min = np.min(x)
    x_max = np.max(x)
    if N is None:
        buf = np.abs(np.diff(np.sort(x)))
        dx = np.min(buf[buf != 0])
        N_MIN = 2  # Minimum number of bins (integer)
        N_MAX = min(int(np.floor((x_max - x_min) / (2 * dx))), 50)
        N = np.arange(N_MIN, N_MAX + 1)

    SN = 30  # Number of partitioning positions for shift average
    D = (x_max - x_min) / N  # Bin size vector

    # Computation of the Cost Function
    Cs = np.zeros((len(N), SN))
    for i in range(len(N)):
        shift = np.linspace(0, D[i], SN)
        for p in range(SN):
            # Bin edges considering the shift
            edges = np.linspace(
                x_min + shift[p] - D[i] / 2, x_max + shift[p] - D[i] / 2, N[i] + 1
            )

            # Count number of events in bins

            ki = np.histogram(x, bins=edges)[0]

            # Calculate mean and variance of event count
            k = np.mean(ki)
            v = np.sum((ki - k) ** 2) / N[i]

            # Calculate the cost function
            Cs[i, p] = (2 * k - v) / D[i] ** 2

    # Average cost over the shifts
    C = np.mean(Cs, axis=1)

    # Optimal bin size selection
    Cmin_idx = np.argmin(C)
    optN = N[Cmin_idx]

    return optN, C, N
