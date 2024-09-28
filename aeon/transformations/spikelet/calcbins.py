import numpy as np

def calcnbins(x, method='middle', minimum=1, maximum=np.inf):
    """Calculate the ideal number of bins to use in a histogram, using a choice of methods."""

    # Ensure x is a numpy array
    x = np.asarray(x)

    # Handle complex numbers
    if np.iscomplexobj(x):
        x = np.real(x)
        print("Warning: Imaginary parts of x will be ignored.")

    # Flatten x if it is not a vector
    if x.ndim > 1:
        x = x.flatten()
        print("Warning: x will be coerced to a vector.")

    # Remove NaN values
    x = x[~np.isnan(x)]
    if x.size == 0:
        raise ValueError("No valid data points in x after removing NaN values.")

    # Validate method
    validmethods = ['fd', 'scott', 'sturges', 'all', 'middle']
    method = method.lower()
    if method not in validmethods:
        raise ValueError(f"Unknown or ambiguous method: {method}")

    # Perform the calculation
    if method == 'fd':
        nbins = calcfd(x)
    elif method == 'scott':
        nbins = calcscott(x)
    elif method == 'sturges':
        nbins = calcsturges(x)
    elif method == 'all':
        nbins = {
            'fd': calcfd(x),
            'scott': calcscott(x),
            'sturges': calcsturges(x),
        }
    elif method == 'middle':
        nbins = np.median([calcfd(x), calcscott(x), calcsturges(x)])

    # Constrain nbins within the specified range
    if isinstance(nbins, dict):
        for key in nbins:
            nbins[key] = confine2range(nbins[key], minimum, maximum)
    else:
        nbins = confine2range(nbins, minimum, maximum)

    return nbins

def calcfd(x):
    """Freedman-Diaconis method for calculating the number of bins."""
    h = np.subtract(*np.percentile(x, [75, 25]))  # Interquartile range

    file_path = r'C:\Users\Victor\Desktop\Uni\Bachelor\stuff\x_array.npy'

    # Save the array to the specified path
    np.save(file_path, x)
    
    if h == 0:
        h = 2 * np.median(np.abs(x - np.median(x)))  # Twice the median absolute deviation
    if h > 0:
        nbins = np.ceil((x.max() - x.min()) / (2 * h * len(x) ** (-1 / 3)))
    else:
        nbins = 1
    return int(nbins)

def calcscott(x):
    """Scott's method for calculating the number of bins."""
    h = 3.5 * np.std(x) * len(x) ** (-1 / 3)
    if h > 0:
        nbins = np.ceil((x.max() - x.min()) / h)
    else:
        nbins = 1
    return int(nbins)

def calcsturges(x):
    """Sturges' method for calculating the number of bins."""
    nbins = np.ceil(np.log2(len(x)) + 1)
    return int(nbins)

def confine2range(x, lower=None, upper=None):
    # Ensure lower and upper are not None
    if lower is None:
        lower = -np.inf  # Set to negative infinity if not provided
    if upper is None:
        upper = np.inf   # Set to infinity if not provided
    
    # Handle cases where lower or upper could be infinity
    if np.isinf(lower):
        lower = -np.inf
    if np.isinf(upper):
        upper = np.inf

    # Ensure lower and upper are within proper numeric range
    y = max(min(int(x), int(upper) if upper != np.inf else x), int(lower) if lower != -np.inf else x)
    return y

