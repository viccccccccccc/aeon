from typing import Tuple, Any, List
import numpy as np

# Import the knee-finding algorithms (you need to implement or mock these)
from aeon.transformations.spikelet.Spikelet_Stat_knee_find_double_2nd_under_heuristics import Spikelet_Stat_knee_find_double_2nd_under_heuristics
from aeon.transformations.spikelet.Spikelet_Stat_knee_find_2nd_zerocross import Spikelet_Stat_knee_find_2nd_zerocross
from aeon.transformations.spikelet.Spikelet_Stat_knee_find_repeat import Spikelet_Stat_knee_find_repeat

def Spikelet_Stat_decideCLhrFromSegmentInitial(
    Segment_initial: np.ndarray, Segment_names: List[str], Param_EF: dict
) -> Tuple[float, Any]:
    """
    Decide the constant length threshold (CLThr) from the initial constant segments.

    Parameters:
        Segment_initial (np.ndarray): Array of initial segments with their properties.
        Segment_names (List[str]): List of names corresponding to columns in Segment_initial.
        Param_EF (dict): Parameter dictionary containing 'knee_find' configuration.

    Returns:
        CLThr_initial (float): The decided constant length threshold.
        KneeInfo_initial (Any): Information from the knee-finding algorithm.
    """
    # Default approximation function
    if (
        "knee_find" in Param_EF
        and "approx_function" in Param_EF["knee_find"]
    ):
        Func = Param_EF["knee_find"]["approx_function"]
    else:
        Func = ["poly_1_poly_0", "poly_1_poly_0"]

    # Extract constant length distribution (CLD)
    CLD = Segment_initial[:, Segment_names.index("length")]

    # Decide the knee-finding method
    if (
        "knee_find" in Param_EF
        and "type" in Param_EF["knee_find"]
        and Param_EF["knee_find"]["type"] == "2nd_under_heuristics"
    ):
        CLThr_initial, KneeInfo_initial = Spikelet_Stat_knee_find_double_2nd_under_heuristics(
            CLD, Func
        )
    elif (
        "knee_find" in Param_EF
        and "type" in Param_EF["knee_find"]
        and Param_EF["knee_find"]["type"] == "2nd_zerocross"
    ):
        CLThr_initial, KneeInfo_initial = Spikelet_Stat_knee_find_2nd_zerocross(
            CLD, Func
        )
    else:
        CLThr_initial, KneeInfo_initial = Spikelet_Stat_knee_find_repeat(CLD, Func)

    return CLThr_initial, KneeInfo_initial
