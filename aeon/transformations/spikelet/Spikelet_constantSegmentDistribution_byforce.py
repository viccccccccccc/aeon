import numpy as np

def Spikelet_constant_segment_distribution_byforce(D, MagThr, UpperLimit=np.inf):
    """
    Computes constant segment distribution for a given data array.
    
    Parameters:
        D (numpy.ndarray): Data array.
        MagThr (float): Magnitude threshold.
        UpperLimit (int, optional): Upper limit for search range. Default is infinity.
        
    Returns:
        Clen (numpy.ndarray): Array containing constant lengths, left, and right lengths.
        Column_names (list): Names of the columns in Clen.
    """
    # Function name for logging
    func_name = "spikelet_constant_segment_distribution_byforce"
    
    # Output initialization
    Clen = np.full((len(D), 3), np.nan)
    Column_names = ["constant_length", "left_constant_length", "right_constant_length"]
    
    # Default value
    for i in range(len(D)):
        count_right = search_right(i, D, MagThr, UpperLimit)
        count_left = search_left(i, D, MagThr, UpperLimit)
        count_len = count_left + count_right - 1  # Total length
        Clen[i, :] = [count_len, count_left, count_right]
    
    return Clen, Column_names


def search_right(pos_i, D, MagThr, UpperLimit):
    """
    Search for constant segment to the right of a position.
    
    Parameters:
        pos_i (int): Starting position.
        D (numpy.ndarray): Data array.
        MagThr (float): Magnitude threshold.
        UpperLimit (int): Upper limit for search range.
        
    Returns:
        count (int): Length of constant segment to the right.
    """
    MagThr_half = MagThr / 2
    count = 0
    i = pos_i
    while i < len(D):
        if abs(D[i] - D[pos_i]) <= MagThr_half:
            count += 1
            if count >= UpperLimit:
                break
            i += 1
        else:
            break
    return count


def search_left(pos_i, D, MagThr, UpperLimit):
    """
    Search for constant segment to the left of a position.
    
    Parameters:
        pos_i (int): Starting position.
        D (numpy.ndarray): Data array.
        MagThr (float): Magnitude threshold.
        UpperLimit (int): Upper limit for search range.
        
    Returns:
        count (int): Length of constant segment to the left.
    """
    MagThr_half = MagThr / 2
    count = 0
    i = pos_i
    while i >= 0:
        if abs(D[i] - D[pos_i]) <= MagThr_half:
            count += 1
            if count >= UpperLimit:
                break
            i -= 1
        else:
            break
    return count


# Example usage
if __name__ == "__main__":
    # Example data
    D = np.array([1, 1, 1.2, 1.1, 1, 2, 2, 2, 3, 3, 3])
    MagThr = 0.5
    UpperLimit = 5

    # Run function
    Clen, Column_names = spikelet_constant_segment_distribution_byforce(D, MagThr, UpperLimit)
    print("Column Names:", Column_names)
    print("Clen Array:")
    print(Clen)
