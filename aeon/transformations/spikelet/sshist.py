import numpy as np

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
        N_MIN = 2              # Minimum number of bins (integer)
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
            edges = np.linspace(x_min + shift[p] - D[i] / 2,
                                x_max + shift[p] - D[i] / 2, N[i] + 1)

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
