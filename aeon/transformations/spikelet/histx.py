import matplotlib.pyplot as plt

from aeon.transformations.spikelet.calcbins import calcnbins

def histx(y, nbins=None, minimum=None, maximum=None):
    if nbins is None:
        nbins = 'middle'
    if isinstance(nbins, int):
        # If nbins is an integer, we use it directly
        calculated_nbins = nbins
    else:
        # If nbins is a string (method), we calculate the number of bins
        calculated_nbins = calcnbins(y, nbins, minimum, maximum)
        print(f"calculated_nbins: {calculated_nbins}")

    if isinstance(nbins, str) and nbins == 'all':
        # Plot histograms for each method if 'all' is specified
        fig, axs = plt.subplots(3, 1, figsize=(8, 12))
        methods = ['fd', 'scott', 'sturges']
        for ax, method in zip(axs, methods):
            method_nbins = calcnbins(y, method)
            ax.hist(y, bins=method_nbins)
            ax.set_title(f"{method.capitalize()} method")
        plt.tight_layout()
        plt.show()
        return None, None
    else:
        # Standard case: plot or return histogram data
        n, bins, patches = plt.hist(y, bins=calculated_nbins)
        print(f"n: {len(n)}, bins: {len(bins)}")
        #plt.show()
        return n, bins

