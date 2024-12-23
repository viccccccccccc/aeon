import numpy as np

from aeon.transformations.spikelet.Spikelet_constant_segment_distribution_byforce import Spikelet_constant_segment_distribution_byforce

def Spikelet_Stat_extractConstantSegmentInitial(MagInfo, MagThr, METHOD_BYFORCE):
    """
    Extract initial constant segments from the data based on a threshold.

    Parameters:
        MagInfo (dict): Dictionary containing 'data_org' and 'magnitude'.
        MagThr (float): Magnitude threshold for constant segment extraction.
        METHOD_BYFORCE (str): Method for constant segment extraction ('simple' or 'magstop').

    Returns:
        tuple: (Segment_initial, Clen, Clen_all, Segment_names, Clen_names)
            Segment_initial : np.ndarray - Initial segments [from, to, length, mean_value].
            Clen : np.ndarray - Constant segment lengths.
            Clen_all : np.ndarray - Detailed constant segment info [length, left, right].
            Segment_names : list - Names of the segment columns.
            Clen_names : list - Names of Clen_all columns.
    """
    FuncName = "spikelet_stat_extract_constant_segment_initial"
    DEBUG = False

    # Argument extraction
    data_org = MagInfo["data_org"].copy()
    magnitude = MagInfo["magnitude"].copy()

    # Set magnitude values below the threshold to zero
    index = np.abs(magnitude) < MagThr
    magnitude[index] = 0

    # Select method for constant segment distribution
    if METHOD_BYFORCE == "simple":
        Clen_all, Clen_names = Spikelet_constant_segment_distribution_byforce(data_org, MagThr)
    elif METHOD_BYFORCE == "magstop":
       1==1
        #Clen_all, Clen_names = Spikelet_constant_segment_distribution_byforce_magstop(data_org, MagThr, magnitude)
    else:
        print(f"[{FuncName}] unknown METHOD_BYFORCE ({METHOD_BYFORCE})")
        return None, None, None, None, None

    Clen = Clen_all[:, 0]  # Constant segment lengths
    Cindex = np.ones(len(Clen), dtype=bool)
    Clen_diff = np.append(0, np.diff(Clen))

    # Initialize output
    Segment_names = ["from", "to", "length", "mean_value"]
    Segment_initial = np.full((len(data_org), len(Segment_names)), np.nan)
    IndexS = np.ones(len(data_org), dtype=bool)
    count = 0
    sumindex_pre = np.sum(IndexS)

    # Constant segment extraction loop
    while np.sum(IndexS) != 0:
        IndexS_times = np.where(IndexS)[0]
        maxlen = np.max(Clen[IndexS_times])
        maxtime_ref = np.argmax(Clen[IndexS_times])
        maxtime = IndexS_times[maxtime_ref]

        left = Clen_all[maxtime, 1]
        right = Clen_all[maxtime, 2]
        length = Clen_all[maxtime, 0]
        from_idx = maxtime - left
        to_idx = maxtime + right

        from_idx = int(maxtime - left)
        to_idx = int(maxtime + right)

        if IndexS[maxtime]:
            count += 1
            Segment_initial[count - 1, :] = [from_idx, to_idx, length, np.mean(data_org[from_idx:to_idx + 1])]
            IndexS[from_idx:to_idx + 1] = False
        else:
            IndexS[maxtime] = False

        if np.sum(IndexS) == sumindex_pre:
            #print(f"[{FuncName}] error in IndexS")
            break

        sumindex_pre = np.sum(IndexS)

        if DEBUG:
            print(f"({count}, {maxtime}, {sumindex_pre}) = (count, maxtime, sumindex) [{FuncName}]")

    # Remove empty rows
    Segment_initial = Segment_initial[~np.isnan(Segment_initial[:, 0])]

    # Sort segments by "from" index
    sort_indices = np.argsort(Segment_initial[:, 0])
    Segment_initial = Segment_initial[sort_indices]

    return Segment_initial, Clen, Clen_all, Segment_names, Clen_names

