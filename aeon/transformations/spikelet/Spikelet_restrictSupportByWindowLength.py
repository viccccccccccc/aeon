import numpy as np

from aeon.transformations.spikelet.Spikelet_MagInfo_post_processing import Spikelet_MagInfo_post_processing

def Spikelet_restrictSupportByWindowLength(MagInfo):
    Debug_On = False

    FuncName = "Spikelet_restrictSupportByWindowLength"
    OpName = "restrictSupportByWindowLength"

    # Parameter
    ParamSRB = MagInfo["param"]["operation"][OpName]
    M_formula = ParamSRB.get(
        "query", "Mag != 0"
    )  # Default to a condition that is always true
    LeftSupportLength = ParamSRB["window_band"][0]
    RightSupportLength = ParamSRB["window_band"][1]

    # Input and output
    Data_org = MagInfo["data_org"].copy()
    Data = MagInfo["data"].copy()
    Mag = MagInfo["magnitude"].copy()
    Left = MagInfo["left"].copy()
    Right = MagInfo["right"].copy()

    # Support restriction by window
    Left_new = Left.copy()
    Right_new = Right.copy()

    # Find target times
    Time = np.arange(0, len(Mag))
    Supp_left = Time - Left + 1
    Supp_right = Right - Time + 1

    Index_M = eval(M_formula)
    Index_MW = np.logical_or(Index_M & (Supp_left > LeftSupportLength), (Supp_right > RightSupportLength))
    
    M_pos = np.where(Index_MW == True)[0]

    # Restrict support
    for t in M_pos:
        if Supp_left[t] > LeftSupportLength:
            from_idx = max(t - LeftSupportLength + 1, 1)
            left_val, left_rel = np.min(Data_org[from_idx : t + 1] * np.sign(Mag[t])), np.argmin(Data_org[from_idx : t + 1] * np.sign(Mag[t]))
            Left_new[t] = from_idx + left_rel

        if Supp_right[t] > RightSupportLength:
            to = min(t + RightSupportLength, len(Data_org))
            right_val, right_rel = np.min(Data_org[t:to] * np.sign(Mag[t])), np.argmin(Data_org[t:to] * np.sign(Mag[t]))
            Right_new[t] = t + right_rel

    # Revise magnitude
    Mag_left_new = Data_org - Data_org[Left_new].values
    Mag_right_new = Data_org - Data_org[Right_new].values

    Mag_new = np.sign(Mag_left_new) * np.minimum(np.abs(Mag_left_new), np.abs(Mag_right_new))

    # Set result
    MagInfo["left"] = Left_new
    MagInfo["right"] = Right_new
    MagInfo["magnitude"] = Mag_new
    MagInfo["data"] = Spikelet_MagInfo_timeSeriesForm(MagInfo)

    # Post processing
    MagInfo = Spikelet_MagInfo_post_processing(MagInfo, OpName)
    MagInfo["output"][OpName].update(
        {
            "query": M_formula,
            "window_band": ParamSRB["window_band"],
            "param_str": f"window_band = {ParamSRB['window_band']}",
        }
    )

    # Debug
    if Debug_On:
        import matplotlib.pyplot as plt

        Row = 6
        plt.figure()
        plt.subplot(Row, 1, 1)
        plt.plot(Data)
        plt.xlim([1, len(Data_org)])
        plt.title("input data")

        plt.subplot(Row, 1, 3)
        plt.plot(Mag)
        plt.xlim([1, len(Data_org)])
        plt.title("input magnitude")

        plt.subplot(Row, 1, 5)
        D_left = Left - Time
        plt.plot(D_left)
        D_right = Right - Time
        plt.plot(D_right)
        plt.xlim([1, len(Data_org)])
        plt.title("input support")

        plt.subplot(Row, 1, 4)
        plt.plot(MagInfo["magnitude"])
        plt.xlim([1, len(Data_org)])
        plt.title(f"{OpName} magnitude")

        plt.subplot(Row, 1, 2)
        plt.plot(MagInfo["data"])
        plt.xlim([1, len(Data_org)])
        plt.title(f"{OpName} time series form")

        plt.subplot(Row, 1, 6)
        D_left_new = Left_new - Time
        plt.plot(D_left_new)
        D_right_new = Right_new - Time
        plt.plot(D_right_new)
        plt.xlim([1, len(Data_org)])
        plt.title(f"{OpName} support")

        plt.suptitle(f"Debug: {FuncName}")
        plt.show()
        print(f"[{FuncName}] debug")

    return MagInfo

def Spikelet_MagInfo_timeSeriesForm(MagInfo):
    """
    Generates the time series representation of the Spikelet decomposition.

    Parameters
    ----------
    MagInfo (dict): A dictionary containing the Spikelet decomposition information.

    Returns
    -------
    numpy.ndarray: The time series representation of the Spikelet decomposition.
    """
    Data = MagInfo["data"]
    Mag = MagInfo["magnitude"]
    Left = MagInfo["left"]
    Right = MagInfo["right"]

    # Find non-zero magnitude points
    P = np.where(Mag != 0)[0]

    if P.size == 0:
        TSS = np.zeros_like(Data)
    else:
        L = Left[P]
        R = Right[P]
        Time = np.unique(np.concatenate(([0], P, L, R, [len(Data) - 1])))
        TSS = np.interp(np.arange(len(Data)), Time, Data[Time])

    return TSS