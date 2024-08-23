from aeon.transformations.spikelet.Spikelet_Stat_knee_find import (
    Spikelet_Stat_knee_find,
    Spikelet_Stat_knee_find_double,
)


def Spikelet_Stat_knee_find_repeat(MagDist, Func):
    """
    Find the knee point using the specified approximation function.

    Parameters
    ----------
    MagDist : numpy.ndarray
        The magnitude distribution data.
    Func : str or list
        The approximation function(s) to be used for knee point detection.

    Returns
    -------
    MagThr : float
        The determined magnitude threshold (knee point).
    Info_S : dict
        Information related to the knee point detection process.
    """
    if isinstance(Func, (list, str)):
        if isinstance(Func, str) or len(Func) == 1:
            MagThr, Info_S = Spikelet_Stat_knee_find(MagDist, Func)  # No weight
        elif len(Func) == 2:
            MagThr, Info_S = Spikelet_Stat_knee_find_double(MagDist, Func)  # No weight
        else:
            print(
                f"[{Spikelet_Stat_knee_find_repeat.__name__}] unknown approx_function length ({len(Func)})"
            )
            return None, None
    else:
        print(
            f"[{Spikelet_Stat_knee_find_repeat.__name__}] unknown approx_function datatype"
        )
        return None, None

    return MagThr, Info_S
