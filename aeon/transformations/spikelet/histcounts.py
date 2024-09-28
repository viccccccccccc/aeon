import numpy as np

def histcounts(x, bins=None):
    """
    A simplified version of histcounts that only accepts a single array input.
    
    Parameters:
    x (array-like): Input data array.

    Returns:
    n (ndarray): The counts of each bin.
    edges (ndarray): The bin edges.
    """
    # Ensure the input is numeric and real
    if not (np.issubdtype(x.dtype, np.number) or np.issubdtype(x.dtype, np.bool_)):
        raise ValueError('Input must be numeric or logical.')
    
    if not np.isrealobj(x):
        raise ValueError('Input must be real.')

    # Flatten the input array
    xc = x.ravel()

    # Handle NaNs and Infs by filtering them out
    xc = xc[np.isfinite(xc)]

    # Determine the bin edges using the autorule function
    minx = np.min(xc)
    maxx = np.max(xc)

    if bins is None:
        # Use autorule if no bin count is provided
        if minx == maxx:
            edges = np.array([minx, maxx])
        else:
            edges = autorule(xc, minx, maxx)
    elif isinstance(bins, int):
        # Calculate edges based on specified number of bins
        xrange = maxx - minx
        edges = binpicker(minx, maxx, nbins=bins, rawBinWidth=xrange / bins)
    else:
        raise ValueError('The second argument must be an integer specifying the number of bins.')

    # Compute the histogram counts using the determined edges
    n, edges = np.histogram(xc, bins=edges)

    return n, edges

def autorule(x, minx, maxx):
    """
    Determines bin edges using a rule similar to MATLAB's autorule function.
    
    Parameters:
    x (array-like): Input data array.
    minx (float): Minimum value of the array.
    maxx (float): Maximum value of the array.

    Returns:
    edges (ndarray): Calculated bin edges.
    """
    xrange = maxx - minx
    is_integer_or_logical = np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.bool_)

    # Check if the data is integer-like and within specific range conditions
    if is_integer_or_logical and xrange <= 50 and maxx <= np.finfo(np.float64).max/2 and minx >= -np.finfo(np.float64).max/2:
        edges = integerrule(x, minx, maxx)
    else:
        edges = scottsrule(x, minx, maxx)

    return edges

def integerrule(x, minx, maxx, max_bins=65536):
    """
    A simplified version of MATLAB's integerrule for integer or logical arrays.
    
    Parameters:
    x (array-like): Input data array.
    minx (float): Minimum value of the array.
    maxx (float): Maximum value of the array.
    max_bins (int): Maximum number of bins.

    Returns:
    edges (ndarray): Calculated bin edges.
    """
    # Create bin edges for integer data with constraints on maximum number of bins
    step = 1
    edges = np.arange(minx, maxx + step, step)
    if len(edges) > max_bins:
        edges = np.linspace(minx, maxx, max_bins)
    return edges

import numpy as np

def scottsrule(x, minx, maxx, hardlimits=False):
    """
    Implements Scott's normal reference rule for calculating bin edges.
    
    Parameters:
    x (array-like): Input data array.
    minx (float): Minimum value of the array.
    maxx (float): Maximum value of the array.
    hardlimits (bool): If true, ensures edges cover the range exactly.

    Returns:
    edges (ndarray): Calculated bin edges using Scott's rule.
    """
    # Convert to float if necessary
    x = np.asarray(x, dtype=float)
    
    # Calculate bin width using Scott's rule
    binwidth = 3.5 * np.std(x) / (len(x) ** (1/3))
    
    if not hardlimits:
        # Bin edges without strict range limits
        edges = binpicker(minx, maxx, None, binwidth)
    else:
        # Bin edges with strict limits, ensuring the edges include minx and maxx
        edges = binpickerbl(min(x), max(x), minx, maxx, binwidth)
    
    return edges

def binpicker(xmin, xmax, nbins=None, rawBinWidth=None):
    # Choose histogram bins based on "nice" bin widths.
    
    if xmin is not None and xmax is not None:
        # Check input types
        if not np.issubdtype(type(xmin), np.floating):
            raise ValueError('Input values must be floats.')

        xscale = max(abs(xmin), abs(xmax))
        xrange = xmax - xmin

        # Ensure bin width is not effectively zero
        rawBinWidth = max(rawBinWidth, np.finfo(float).eps * xscale)

        # If data are not constant, place bins at "nice" locations
        if xrange > max(np.sqrt(np.finfo(float).eps) * xscale, np.finfo(float).tiny):
            # Choose the bin width as a "nice" value
            powOfTen = 10 ** np.floor(np.log10(rawBinWidth))  # next lower power of 10
            relSize = rawBinWidth / powOfTen  # guaranteed in [1, 10)

            # Automatic rule specified
            if nbins is None:
                if relSize < 1.5:
                    binWidth = 1 * powOfTen
                elif relSize < 2.5:
                    binWidth = 2 * powOfTen
                elif relSize < 4:
                    binWidth = 3 * powOfTen
                elif relSize < 7.5:
                    binWidth = 5 * powOfTen
                else:
                    binWidth = 10 * powOfTen

                # Put the bin edges at multiples of the bin width, covering x
                leftEdge = max(min(binWidth * np.floor(xmin / binWidth), xmin), -np.finfo(float).max)
                nbinsActual = max(1, int(np.ceil((xmax - leftEdge) / binWidth)))
                rightEdge = min(max(leftEdge + nbinsActual * binWidth, xmax), np.finfo(float).max)

            else:
                # Number of bins specified
                binWidth = powOfTen * np.floor(relSize)
                leftEdge = max(min(binWidth * np.floor(xmin / binWidth), xmin), -np.finfo(float).max)
                
                if nbins > 1:
                    ll = (xmax - leftEdge) / nbins  # lower bound of bin width, xmax on right edge of last bin
                    ul = (xmax - leftEdge) / (nbins - 1)  # upper bound of bin width, xmax on left edge of last bin
                    p10 = 10 ** np.floor(np.log10(ul - ll))
                    binWidth = p10 * np.ceil(ll / p10)  # binWidth-ll < p10 <= ul-ll
                    # Thus, binWidth < ul

                nbinsActual = nbins
                rightEdge = min(max(leftEdge + nbinsActual * binWidth, xmax), np.finfo(float).max)

        else:  # the data are nearly constant
            if nbins is None:
                nbins = 1

            # Make the bins cover a unit width, or as small an integer width as possible
            binRange = max(1, int(np.ceil(nbins * np.finfo(float).eps * xscale)))
            leftEdge = np.floor(2 * (xmin - binRange / 4)) / 2
            rightEdge = np.ceil(2 * (xmax + binRange / 4)) / 2

            binWidth = (rightEdge - leftEdge) / nbins
            nbinsActual = nbins

        if not np.isfinite(binWidth):
            # if binWidth overflows, create evenly spaced bins
            edges = np.linspace(leftEdge, rightEdge, nbinsActual + 1)
        else:
            edges = np.concatenate(([leftEdge], leftEdge + np.arange(1, nbinsActual) * binWidth, [rightEdge]))
    else:
        # Handle empty input
        if nbins is not None:
            edges = np.arange(nbins + 1, dtype=float)
        else:
            edges = np.array([0, 1], dtype=float)

    return edges


def binpickerbl(min_val, max_val, minx, maxx, binwidth):
    """
    Mimics MATLAB's binpickerbl behavior for setting bin edges with hard limits.
    
    Parameters:
    min_val (float): Minimum data value.
    max_val (float): Maximum data value.
    minx (float): Minimum limit for the edges.
    maxx (float): Maximum limit for the edges.
    binwidth (float): Width of each bin.
    
    Returns:
    edges (ndarray): Calculated bin edges that cover the specified limits exactly.
    """
    # Compute starting and ending edges based on hard limits
    left_edge = binwidth * np.floor(minx / binwidth)
    right_edge = binwidth * np.ceil(maxx / binwidth)
    
    # Generate bin edges
    edges = np.arange(left_edge, right_edge + binwidth, binwidth)
    
    # Ensure the edges include the minx and maxx exactly
    if edges[0] > minx:
        edges = np.insert(edges, 0, minx)
    if edges[-1] < maxx:
        edges = np.append(edges, maxx)
    
    return edges
