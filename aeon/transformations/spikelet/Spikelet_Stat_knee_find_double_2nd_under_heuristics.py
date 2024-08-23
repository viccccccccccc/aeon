import numpy as np

from aeon.transformations.spikelet.Spikelet_Stat_knee_find import (
    Spikelet_Stat_knee_find,
    plot_kee_find,
)


def Spikelet_Stat_knee_find_double_2nd_under_heuristics(MagDist, FuncList, Weight=None):
    """
    Find the knee point using a double 2nd under heuristic method.

    Parameters
    ----------
    MagDist : numpy.ndarray
        The magnitude distribution data.
    FuncList : list of str
        A list of functions to be used for knee point detection.
    Weight : numpy.ndarray or None, optional
        Weights for the knee point detection, by default None.

    Returns
    -------
    KeeOpt : float
        The optimized knee point.
    Info : dict
        Information related to the knee point detection process.
    """
    DEBUG = False

    if Weight is None:
        Weight = np.nan

    # First knee detection
    KeeOpt_1, Info_1 = Spikelet_Stat_knee_find(MagDist, FuncList[0], Weight)

    # Magnitude distribution for the second phase
    MagDist_2 = MagDist[MagDist >= KeeOpt_1]
    KeeOpt_2, Info_2 = Spikelet_Stat_knee_find(MagDist_2, FuncList[1], Weight)

    # Correction phase
    fwd_model_type_2 = Info_2["bwd_model_type"]
    fwd_model_2 = Info_2["fwd_model"]
    Y_fwd = Info_2["Y_fwd"]
    MD_fwd = Info_2["MD_fwd"]

    # Magnitude distribution for the third phase
    MagDist_3 = MagDist[(KeeOpt_1 <= MagDist) & (MagDist <= KeeOpt_2)]
    KeeOpt_3, Info_3 = Spikelet_Stat_knee_find(MagDist_3, FuncList[2], Weight)

    # Final optimized knee point
    KeeOpt = KeeOpt_3
    aprx_level = 1.5

    # Information collection
    Info = {
        "knee_opt": KeeOpt,
        "aprx_level": aprx_level,
        "detail": [Info_1, Info_2, Info_3],
    }

    if DEBUG:
        plot_kee_find(Info)

    return KeeOpt, Info
