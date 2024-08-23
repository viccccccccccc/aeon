import numpy as np
import os


from aeon.transformations.spikelet.Spikelet_Stat_knee_find_2nd_zerocross import (
    Spikelet_Stat_knee_find_2nd_zerocross,
)
from aeon.transformations.spikelet.Spikelet_Stat_knee_find_double_2nd_under_heuristics import (
    Spikelet_Stat_knee_find_double_2nd_under_heuristics,
)
from aeon.transformations.spikelet.Spikelet_Stat_knee_find_of_mag_using_magband_support import (
    Spikelet_Stat_knee_find_of_mag_using_magband_support,
)
from aeon.transformations.spikelet.Spikelet_Stat_knee_find_of_mag_using_support_knee import (
    Spikelet_Stat_knee_find_of_mag_using_support_knee,
)
from aeon.transformations.spikelet.Spikelet_Stat_knee_find_repeat import (
    Spikelet_Stat_knee_find_repeat,
)


def Spikelet_Stat_magnitude_threshold_SuppMax_SuppMRatio(MagInfo, ParamMagThr):
    """
    Calculate the magnitude threshold using the support maximum to support magnitude ratio.

    Parameters
    ----------
    MagInfo : dict
        Dictionary containing magnitude, left, right, and other related fields.
    ParamMagThr : dict
        Dictionary containing parameters for threshold calculation.

    Returns
    -------
    MagThr : float
        The calculated magnitude threshold.
    Mag_selected : numpy.ndarray
        The selected magnitude values.
    MagInfo : dict
        Updated MagInfo with any necessary changes.
    KneeInfo : Any
        Information related to the knee point found during threshold calculation.
    """
    # Output initialization
    KneeInfo = np.nan

    # Knee find parameters
    Type = ParamMagThr["knee_find"]["type"]
    Sign = ParamMagThr["knee_find"]["sign"]
    Func = ParamMagThr["knee_find"]["approx_function"]

    # Magnitude distribution and support distribution
    Mag = MagInfo["magnitude"]
    Left = MagInfo["left"]
    Right = MagInfo["right"]
    Supp = Right - Left + 1

    if Sign == "positive":
        Mag_index = Mag > 0
    elif Sign == "negative":
        Mag_index = Mag < 0
    elif Sign == "nonzero":
        Mag_index = Mag != 0
    else:  # Default is nonzero
        Mag_index = Mag != 0

    Mag_selected = np.abs(Mag[Mag_index])
    Supp_selected = Supp[Mag_index]

    save_directory = r'C:\Users\Victor\Desktop\Uni\Bachelor\stuff'

    # Make sure the directory exists; if not, create it
    os.makedirs(save_directory, exist_ok=True)

    # Save the file
    np.save(os.path.join(save_directory, 'Mag_index.npy'), Mag_index)

    print(f"Mag len: {len(Mag)}, Mag_index len: {len(Mag_index)}")
    print(f"Mag_selected len: {len(Mag_selected)}, Supp_selected len: {len(Supp_selected)}")

    print(f"in magnitude threshold: Mag[167] = {Mag[167]}")
    print(f"Mag[168] = {Mag[168]}")
    print(f"Mag[169] = {Mag[169]}")
    print(f"Mag[192] = {Mag[192]}")
    print(f"Mag[8509] = {Mag[8509]}")
    print(f"Mag[45996] = {Mag[45996]}")
    print(f"Mag[114436] = {Mag[114436]}")

    # Find knee
    if Type == "simple":
        MagThr, Info_S = Spikelet_Stat_knee_find_repeat(Mag_selected, Func)
    elif Type == "2nd_under_heuristics":
        MagThr, KneeInfo = Spikelet_Stat_knee_find_double_2nd_under_heuristics(
            Mag_selected, Func
        )
    elif Type == "2nd_zerocross":
        MagThr, KneeInfo = Spikelet_Stat_knee_find_2nd_zerocross(Mag_selected, Func)
    elif Type == "magband_support":
        BandFunc = ParamMagThr["knee_find"].get("band_aggfunction", "sum")
        MagThr, KneeInfo = Spikelet_Stat_knee_find_of_mag_using_magband_support(
            Mag_selected, Supp_selected, Func, BandFunc
        )
    elif Type == "support_knee":
        Support_knee_func = ParamMagThr["knee_find"].get(
            "support_knee_func", "poly_1_0"
        )

        if not ParamMagThr["knee_find"].get("weight") or (
            ParamMagThr["knee_find"].get("weight") == "none"
        ):
            Weight = np.nan
        elif ParamMagThr["knee_find"].get("weight") == "support_length":
            Weight = Supp_selected

        MagThr, KneeInfo = Spikelet_Stat_knee_find_of_mag_using_support_knee(
            Mag_selected, Supp_selected, Support_knee_func, Func, Weight
        )
    else:
        print(f"[{Func}] type ({Type}) is not implemented")
        return None, None, MagInfo, KneeInfo

    return MagThr, Mag_selected, MagInfo, KneeInfo
