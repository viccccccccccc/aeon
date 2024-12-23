import numpy as np
import time
import pdb

#from aeon.transformations.spikelet.spikelet_approx import Spikelet_aproximation_ver_03
from aeon.transformations.spikelet.Spikelet_MagInfo_post_processing import Spikelet_MagInfo_post_processing
from aeon.transformations.spikelet.Spikelet_Stat_extractConstantSegmentInitial import Spikelet_Stat_extractConstantSegmentInitial
from aeon.transformations.spikelet.Spikelet_Stat_decideCLhrFromSegmentInitial import Spikelet_Stat_decideCLhrFromSegmentInitial
from aeon.transformations.spikelet.Spikelet_Stat_decideConstantSegment import Spikelet_Stat_decideConstantSegment

def Spikelet_Op_extractConstantSegment_ver02(MagInfo):
    func_name = "Spikelet_Op_extractConstantSegment_ver02"
    op_name = "extractConstantSegment"

    # Debug parameters
    DEBUG_PLOT_TREND = False
    DEBUG_SPIKE_REDUCTION = False
    VERBOSE = False

    # Arguments
    MagThr = MagInfo["output"]["reduceSpikeByMagnitude"]["magnitude_threshold"]
    Data_org = MagInfo["data_org"]
    Magnitude = MagInfo["magnitude"]
    Left = MagInfo["left"]
    Right = MagInfo["right"]
    Peak_times = np.where(Magnitude != 0)[0]

    # Parameters
    Param = MagInfo["param"]["operation"][op_name]
    Method = Param["lengthThreshold"]["method"]

    # Initial constant segment
    METHOD_BYFORCE = "simple"
    Segment_initial, Clen, Clen_all, Segment_names, Clen_names = (
        Spikelet_Stat_extractConstantSegmentInitial(MagInfo, MagThr, METHOD_BYFORCE)
    )

    # Decide CLThr
    if Method == "manual":
        CLThr = Param["lengthThreshold"][Method]["CLThr"]
        Param_auto = {"supp_max": np.inf, "supp_min": 0}
    elif Method == "auto":
        Param_auto = Param["auto"]
        if Segment_initial.shape[0] > 1:
            CLThr, KneeInfo = Spikelet_Stat_decideCLhrFromSegmentInitial(
                Segment_initial, Segment_names, Param_auto
            )
        else:
            CLThr, KneeInfo = np.nan, np.nan
    else:
        print(f"[{func_name}] Unknown method ({Method})")
        return MagInfo

    # Decide constant segment
    start_cot = time.time()
    if not np.isnan(CLThr):
        Op = {
            "operation_sequence": ["restrictSupportByWindowLength", "restrictSupportByMagnitudeRatio"],
            "restrictSupportByMagnitudeRatio": {"magnitude_ratio": 0.5},
            "restrictSupportByWindowLength": {"window_band": [Param_auto["supp_max"], Param_auto["supp_max"]]},
        }
        Param_cnst = {"operation": Op}
        from aeon.transformations.spikelet.spikelet_approx import Spikelet_aproximation_ver_03
        MagInfo_for_cnst = Spikelet_aproximation_ver_03(Data_org, Param_cnst)

        if Segment_initial.shape[0] > 2:
            Segment, Segment_names, Segment_hist = Spikelet_Stat_decideConstantSegment(
                Data_org, Segment_initial, Segment_names, Param_auto, MagThr, CLThr, MagInfo_for_cnst
            )
        else:
            Segment, Segment_hist = Segment_initial, np.nan
    else:
        Segment, Segment_hist = Segment_initial, np.nan
    
    time_cot = time.time() - start_cot
    MagInfo["time"]["estimate_threshold"] = {"CoT": time_cot}

    # Update output
    MagInfo["CLThr"] = CLThr
    MagInfo["constantSegmentInitial"] = Segment_initial
    MagInfo["constantSegment"] = Segment
    MagInfo["output"][op_name] = {
        "length_threshold": CLThr,
        "segment": Segment,
        "segment_initial": Segment_initial,
        "segment_hist": Segment_hist,
    }
    if "KneeInfo" in locals():
        MagInfo["output"][op_name]["CLThr_kneeInfo"] = KneeInfo

    # Create ConstantSegment table
    constant_segment_names = [
        "constant_time_from", "constant_time_to", "band_value_mean"
    ]
    ConstantSegment = np.zeros((Segment.shape[0], len(constant_segment_names))) * np.nan
    for i in range(Segment.shape[0]):
        ConstantSegment[i, 0] = Segment[i, Segment_names.index("from")]
        ConstantSegment[i, 1] = Segment[i, Segment_names.index("to")]
        ConstantSegment[i, 2] = Segment[i, Segment_names.index("mean_value")]

    # Spike reduction
    for peak_i in Peak_times:
        if ConstantSegment[:, 0].astype(int).any() <= peak_i <= ConstantSegment[:, 1].astype(int).any():
            MagInfo = reduce_spike(MagInfo, peak_i)
            if VERBOSE:
                print(f"Spike reduction at {peak_i}")

    # Decide centers of constant segment
    for i in range(ConstantSegment.shape[0]):
        from_i = int(ConstantSegment[i, 0])
        to_i = int(ConstantSegment[i, 1])
        val_i = ConstantSegment[i, 2]

        center_i = (from_i + to_i) // 2
        MagInfo["type"][center_i] = 0
        MagInfo["left"][center_i] = from_i
        MagInfo["right"][center_i] = to_i
        MagInfo["value"][center_i] = val_i

    MagInfo["output"][op_name]["type"] = MagInfo["type"]

    return MagInfo

def reduce_spike(MagInfo, peak_i):
    MagInfo["type"][peak_i] = np.nan
    MagInfo["value"][peak_i] = np.nan
    MagInfo["magnitude"][peak_i] = 0
    MagInfo["left"][peak_i] = peak_i
    MagInfo["right"][peak_i] = peak_i
    return MagInfo

def refine_right_terminal(MagInfo, peak_time, right_time):
    """
    Refine the right terminal of a spike.
    
    Parameters:
        MagInfo (dict): Contains spike data and metadata.
        peak_time (int): Index of the peak.
        right_time (int): Updated right terminal time.
    
    Returns:
        dict: Updated MagInfo with refined right terminal.
    """
    MagInfo['right'][peak_time] = right_time
    MagInfo['magnitude'][peak_time] = recalculate_magnitude(MagInfo, peak_time)
    return MagInfo


def refine_left_terminal(MagInfo, peak_time, left_time):
    """
    Refine the left terminal of a spike.
    
    Parameters:
        MagInfo (dict): Contains spike data and metadata.
        peak_time (int): Index of the peak.
        left_time (int): Updated left terminal time.
    
    Returns:
        dict: Updated MagInfo with refined left terminal.
    """
    MagInfo['left'][peak_time] = left_time
    MagInfo['magnitude'][peak_time] = recalculate_magnitude(MagInfo, peak_time)
    return MagInfo


def reduce_spike(MagInfo, peak_i):
    """
    Reduce a spike by setting its attributes to neutral values.
    
    Parameters:
        MagInfo (dict): Contains spike data and metadata.
        peak_i (int): Index of the spike to reduce.
    
    Returns:
        dict: Updated MagInfo with the spike reduced.
    """
    MagInfo['type'][peak_i] = np.nan
    MagInfo['value'][peak_i] = np.nan
    MagInfo['magnitude'][peak_i] = 0
    MagInfo['left'][peak_i] = peak_i
    MagInfo['right'][peak_i] = peak_i
    return MagInfo


def recalculate_magnitude(MagInfo, peak_time):
    """
    Recalculate the magnitude of a spike.
    
    Parameters:
        MagInfo (dict): Contains spike data and metadata.
        peak_time (int): Index of the spike peak.
    
    Returns:
        float: The recalculated magnitude of the spike.
    """
    Data_org = MagInfo['data_org']
    left_time = MagInfo['left'][peak_time]
    right_time = MagInfo['right'][peak_time]
    
    left_mag = abs(Data_org[peak_time] - Data_org[left_time])
    right_mag = abs(Data_org[peak_time] - Data_org[right_time])
    
    return min(left_mag, right_mag) * np.sign(Data_org[peak_time] - Data_org[left_time])

import numpy as np

def find_largest_magnitude_terminal_left(data_org, peak, sign_peak, constant_to):
    """
    Find the left terminal with the largest magnitude difference.
    
    Parameters:
        data_org (array): The original data array.
        peak (int): Index of the peak.
        sign_peak (int): Sign of the peak (+1 or -1).
        constant_to (int): Start of the constant region.
    
    Returns:
        int: The left terminal index.
    """
    func_name = "find_largest_magnitude_terminal_left"
    data_range = data_org[constant_to:peak + 1]  # Extract range
    differences = sign_peak * (data_org[peak] - data_range)
    
    if differences.size == 0 or np.all(differences == 0):
        print(f"[{func_name}] Unexpected left terminal {peak}")
        return peak
    else:
        left_ref = np.argmax(differences)
        return constant_to + left_ref


def find_largest_magnitude_terminal_right(data_org, peak, sign_peak, constant_from):
    """
    Find the right terminal with the largest magnitude difference.
    
    Parameters:
        data_org (array): The original data array.
        peak (int): Index of the peak.
        sign_peak (int): Sign of the peak (+1 or -1).
        constant_from (int): End of the constant region.
    
    Returns:
        int: The right terminal index.
    """
    func_name = "find_largest_magnitude_terminal_right"
    data_range = data_org[peak:constant_from + 1]  # Extract range
    differences = sign_peak * (data_org[peak] - data_range)
    
    if differences.size == 0 or np.all(differences == 0):
        print(f"[{func_name}] Unexpected right terminal {peak}")
        return peak
    else:
        right_ref = np.argmax(differences)
        return peak + right_ref


def find_right_terminal(data_org, peak, right_limit, segment_from, segment_to):
    """
    Find the right terminal where the trend crosses the mean value.
    
    Parameters:
        data_org (array): The original data array.
        peak (int): Index of the peak.
        right_limit (int): The upper limit for the search.
        segment_from (int): Start of the constant region.
        segment_to (int): End of the constant region.
    
    Returns:
        int: The right terminal index.
    """
    func_name = "find_right_terminal"
    peak_val = data_org[peak]
    constant_region = data_org[segment_from:segment_to + 1]
    mean_val = np.mean(constant_region)
    
    right_ref = np.where((constant_region - mean_val) * (peak_val - mean_val) < 0)[0]
    
    if right_ref.size == 0:
        print(f"[{func_name}] Unexpected right limit from {peak}")
        return right_limit
    else:
        right = segment_from + right_ref[0]
        return min(right, right_limit)


def find_left_terminal(data_org, peak, left_limit, segment_from, segment_to):
    """
    Find the left terminal where the trend crosses the mean value.
    
    Parameters:
        data_org (array): The original data array.
        peak (int): Index of the peak.
        left_limit (int): The lower limit for the search.
        segment_from (int): Start of the constant region.
        segment_to (int): End of the constant region.
    
    Returns:
        int: The left terminal index.
    """
    func_name = "find_left_terminal"
    peak_val = data_org[peak]
    constant_region = data_org[segment_from:segment_to + 1]
    mean_val = np.mean(constant_region)
    
    left_ref = np.where((constant_region - mean_val) * (peak_val - mean_val) < 0)[0]
    
    if left_ref.size == 0:
        print(f"[{func_name}] Unexpected left limit from {peak}")
        return left_limit
    else:
        left = segment_from + left_ref[-1]  # Use the last crossing
        return max(left, left_limit)

import numpy as np

def recalculate_magnitude(MagInfo, peak_time):
    """
    Recalculate the magnitude of a peak based on its left and right values.

    Parameters:
        MagInfo (dict): Dictionary containing data_org, left, and right indices.
        peak_time (int): Index of the peak time.

    Returns:
        float: Recalculated magnitude value.
    """
    data_org = MagInfo["data_org"]
    left_time = MagInfo["left"][peak_time]
    right_time = MagInfo["right"][peak_time]

    # Calculate the context sign
    context = np.sign(data_org[peak_time] - data_org[left_time])
    
    # Calculate left and right magnitude differences
    left_mag = data_org[peak_time] - data_org[left_time]
    right_mag = data_org[peak_time] - data_org[right_time]

    # Return the minimum absolute magnitude scaled by the context
    return context * min(abs(left_mag), abs(right_mag))


def mean_cross(X):
    """
    Count the number of zero-crossings in a signal X around its mean.

    Parameters:
        X (array-like): Input signal.

    Returns:
        int: The number of zero-crossings.
    """
    X = X - np.mean(X)  # Subtract mean to center the signal
    C = 0  # Initialize counter

    for i in range(len(X) - 1):
        # Count zero-crossings between consecutive points
        if X[i] * X[i + 1] < 0:
            C += 1
        # Special case when X[i] is exactly 0
        if X[i] == 0 and i != 0:
            if X[i - 1] * X[i + 1] < 0:
                C += 1

    return C


