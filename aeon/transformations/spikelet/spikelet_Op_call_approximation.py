import time
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aeon.transformations.spikelet.spikelet_Stat_magnitude_threshold_SuppMax_SuppMRatio import (
    Spikelet_Stat_magnitude_threshold_SuppMax_SuppMRatio,
)


def print_keys(d, prefix=""):
    for key, value in d.items():
        print(f"{prefix}{key}")
        if isinstance(value, dict):
            print_keys(value, prefix + "    ")


def Spikelet_Op_call_approximation(MagInfo):
    InitialOpName = "generateInitialSpikelet"
    Ops = MagInfo["param"]["operation"]["operation_sequence"]

    # print_keys(MagInfo)

    for Op in Ops:
        print("OP: ", Op)
        start_time = time.time()

        # 1. Spike reduction
        if Op == "reduceSpikeByMagnitude":  # revised version for filterByMagnitude
            MagInfo = Spikelet_Op_reduceSpikeByMagnitude(MagInfo)
            Query = MagInfo["output"][Op]["query"]
            print(f"{Op}: Query = {Query}")

        elif (
            Op == "reduceSpikeByMagnitudeRatio"
        ):  # revised version for reduceLocalSpike
            MagInfo = Spikelet_reduceSpikeByMagnitudeRatio(MagInfo)

        elif Op == "reduceSpikeByQuery":
            MagInfo = Spikelet_reduceSpikeByQuery(MagInfo)

        # 2. Support restriction
        elif Op == "restrictSupportByWindowLength":
            MagInfo = Spikelet_restrictSupportByWindowLength(MagInfo)

        elif Op == "restrictSupportByMagnitudeRatio":

            indices = [167, 168, 169, 192, 8509, 45996, 114436]
            for index in indices:
                if index < len(MagInfo['magnitude']):
                    print(f"Wert an Index {index}: {MagInfo['magnitude'][index]}")
                else:
                    print(f"Index {index} liegt außerhalb des gültigen Bereichs.")
                    
            MagInfo = Spikelet_restrictSupportByMagnitudeRatio(MagInfo)

        elif Op == "restrictSupportByMagnitudeRatioInitial":
            MagInfo = Spikelet_restrictSupportByMagnitudeRatioInitial(MagInfo)

        # 3. Constant segment extraction
        elif Op == "extractConstantSegment":
            # MagInfo = Spikelet_Op_extractConstantSegment(MagInfo)  # correct version
            MagInfo = Spikelet_Op_extractConstantSegment_ver02(MagInfo)  # test version
            CLThr = MagInfo["output"][Op]["length_threshold"]
            print(f"{Op}: Constant Length Threshold = {CLThr}")

        else:
            print(f"unknown operation ({Op})")

        # Time and size
        MagInfo["time"][Op] = time.time() - start_time
        MagInfo["size"][Op] = np.sum(~np.isnan(MagInfo["output"][Op]["type"]))
        print("MagInfo['size'][Op]: ", MagInfo["size"][Op])
        print(f"{Op}: time = {MagInfo['time'][Op]}, size = {MagInfo['size'][Op]}")

        print("--------------------------------durchgang durch------------------------------")

    # Total time
    Total_time = MagInfo["time"][InitialOpName]
    for Op in Ops:
        Total_time += MagInfo["time"][Op]

    MagInfo["time"]["operation_all"] = Total_time
    MagInfo["size"]["operation_all"] = MagInfo["size"][Ops[-1]]


    
    return MagInfo


# Helper functions
def Spikelet_Op_reduceSpikeByMagnitude(MagInfo):
    FuncName = "Spikelet_reduceSpikeByMagnitude"
    OpName = "reduceSpikeByMagnitude"

    # Parameters
    Param = MagInfo["param"]["operation"][OpName]
    Method = Param["method"]
    start_mat = time.time()

    if Method == "manual":
        Query = Param[Method]["query"]
        MagThr = Param[Method]["magnitude_threshold"]
    elif Method == "auto":
        Param_Auto = Param[Method]
        METHOD_auto = Param_Auto["method"]
        if "op" in Param_Auto:
            ParamTmp = {"operation": Param_Auto["op"]}
        else:
            ParamTmp = Param_Auto

        MagInfo_tmp = Spikelet_MagInfo_get_initial(MagInfo, ParamTmp)  # inherit only decomposition not parameters
        if METHOD_auto == "initial":
            MagThr, MagDist, MagInfo_auto = (Spikelet_Stat_magnitude_threshold_1stversion(MagInfo_tmp, Param_Auto))
        elif METHOD_auto == "stdMagThr":
            nstd = Param_Auto["std"]
            Mag_tmp = MagInfo_tmp["magnitude"]
            Mag_tmp_nz = Mag_tmp[Mag_tmp != 0]
            MagThr = nstd * np.std(np.abs(Mag_tmp_nz))
        elif METHOD_auto == "SuppMax-SuppMRatio":
            MagInfo_tmp = Spikelet_Op_call_approximation(MagInfo_tmp)

            MagThr, MagDist, MagInfo_auto, KneeInfo = (
                Spikelet_Stat_magnitude_threshold_SuppMax_SuppMRatio(MagInfo_tmp, Param_Auto)
            )

        Query = f"abs(Mag) > {MagThr}"
    else:
        print(f"[{FuncName}] Unknown Method ({Method})")

    print("rausgekommen")
    time_mat = time.time() - start_mat
    MagInfo["time"]["estimate_threshold"] = {"MaT": time_mat}

    # Find center
    SpikeCenter, ReducedTime = Spikelet_eval_query(Query, MagInfo)

    # Revise spikelet decomposition
    if ReducedTime.size > 0:
        MagInfo["magnitude"][ReducedTime] = np.zeros(len(ReducedTime))
        MagInfo["value"][ReducedTime] = np.nan
        MagInfo["type"][ReducedTime] = np.nan
        MagInfo["left"][ReducedTime] = ReducedTime
        MagInfo["right"][ReducedTime] = ReducedTime

    # Output
    MagInfo = Spikelet_MagInfo_post_processing(MagInfo, OpName)
    MagInfo["MagThr"] = MagThr
    MagInfo["output"][OpName] = {
        "magnitude_threshold": MagThr,
        "query": Query,
        "param_str": Query,
    }

    if "KneeInfo" in locals():
        MagInfo["output"][OpName]["magnitude_threshold_kneeInfo"] = KneeInfo

    return MagInfo


def Spikelet_reduceSpikeByMagnitudeRatio(MagInfo):
    # Mock function for spike reduction by magnitude ratio
    return MagInfo


def Spikelet_reduceSpikeByQuery(MagInfo):
    # Mock function for spike reduction by query
    return MagInfo


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
    Time = np.arange(1, len(Mag) + 1)
    Supp_left = Time - Left + 1
    Supp_right = Right - Time + 1

    Index_M = eval(M_formula)
    Index_MW = np.logical_or(
        Index_M & (Supp_left > LeftSupportLength), (Supp_right > RightSupportLength)
    )
    
    M_pos = np.where(Index_MW == True)[0]

    # Restrict support
    for t in M_pos:
        if Supp_left[t] > LeftSupportLength:
            from_idx = max(t - LeftSupportLength + 1, 1)
            left_val, left_rel = np.min(Data_org[from_idx - 1 : t] * np.sign(Mag[t])), np.argmin(Data_org[from_idx - 1 : t] * np.sign(Mag[t]))
            Left_new[t] = from_idx + left_rel - 1

        if Supp_right[t] > RightSupportLength:
            to = min(t + RightSupportLength - 1, len(Data_org) - 1)  # -1 da Python 0-basiert ist
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


def Spikelet_restrictSupportByMagnitudeRatio(MagInfo):
    Debug_On = False
    FuncName = "Spikelet_restrictSupportByMagnitudeRatio"
    OpName = "restrictSupportByMagnitudeRatio"

    # Parameters
    ParamSRM = MagInfo["param"]["operation"][OpName]
    MRatio = ParamSRM["magnitude_ratio"]

    # Input and output
    Data_org = MagInfo["data_org"]
    Mag = MagInfo["magnitude"]
    Left = MagInfo["left"]
    Right = MagInfo["right"]

    # Support restriction by magnitude (8 fields)

    Mag_new, Left_new, Right_new = support_restriction_by_magnitude_ratio(Data_org, Mag, Left, Right, ParamSRM)

    indices = [167, 168, 169, 192, 8509, 45996, 114436]
    for index in indices:
            if index < len(Mag_new):
                print(f"nach support_restriction_by_magnitude_ratio: Wert an Index {index}: {Mag_new[index]}")
            else:
                print(f"Index {index} liegt außerhalb des gültigen Bereichs.")

    # Revised fields (magnitude, left, right, leg_magnitude)
    MagInfo["magnitude"] = Mag_new
    MagInfo["left"] = Left_new
    MagInfo["right"] = Right_new

    # Post processing
    MagInfo = Spikelet_MagInfo_post_processing(MagInfo, OpName)
    MagInfo["output"][OpName]["magnitude_ratio_threshold"] = MRatio
    MagInfo["output"][OpName]["param_str"] = f"magnitude_ratio_threshold ={MRatio}"

    

    # Debug
    if Debug_On:
        Mag_org = Mag
        Supp_org = Right - Left
        Supp_new = Right_new - Left_new

        Index = np.where(Supp_org != Supp_new)[0]

        Row = 5
        plt.figure()

        plt.subplot(Row, 1, 1)
        plt.plot(Data_org)
        plt.xlim([0, len(Data_org)])
        plt.title("data original")
        for i in Index:
            plt.axvline(x=i, color="r")

        plt.subplot(Row, 1, 2)
        plt.plot(Mag_org)
        plt.xlim([0, len(Mag_org)])
        plt.title("magnitude before")
        MagRange = [np.min(Mag_org), np.max(Mag_org)]
        plt.ylim(MagRange)

        plt.subplot(Row, 1, 3)
        plt.plot(Mag_new)
        plt.xlim([0, len(Mag_new)])
        plt.title("magnitude after")
        plt.ylim(MagRange)

        plt.subplot(Row, 1, 4)
        Supp_org_signed = np.sign(Mag_org) * Supp_org
        plt.plot(Supp_org_signed)
        plt.xlim([0, len(Supp_org)])
        plt.title("support length before")
        SuppRange = [np.min(Supp_org_signed), np.max(Supp_org_signed)]
        plt.ylim(SuppRange)

        plt.subplot(Row, 1, 5)
        Supp_new_signed = np.sign(Mag_new) * Supp_new
        plt.plot(Supp_new_signed)
        plt.xlim([0, len(Supp_new)])
        plt.title("support length after")
        plt.ylim(SuppRange)

        plt.show()

    return MagInfo


def support_restriction_by_magnitude_ratio(Data_org, Mag_org, Left_org, Right_org, ParamSRM):
    print("**************************************")
    MRatio = ParamSRM["magnitude_ratio"]

    Supp = np.column_stack((Left_org, Right_org))

    # Output arrays initialized to the original length
    Left = Left_org.copy()
    Right = Right_org.copy()
    Mag = np.zeros_like(Mag_org)

    # Convert Pandas Series to numpy array if necessary
    if isinstance(Mag_org, pd.Series):
        Mag_org = Mag_org.values
    if isinstance(Data_org, pd.Series):
        Data_org = Data_org.values

    # Find positions where Mag_org is not zero
    M_pos = np.where(Mag_org != 0)[0]

    indices_to_check = [167, 168, 169, 192, 8509, 45996, 114436]

    for t in M_pos:
        if t in indices_to_check:
            print(f"\nIndex {t}:")
            print(f"Original Mag[t]: {Mag_org[t]}")
        # Find left terminal
        Mag_org_rel = Mag_org[Supp[t, 0] : t]

        left_mag_rel = np.where((Mag_org_rel * np.sign(Mag_org[t]) > 0)& (np.abs(Mag_org_rel) > np.abs(Mag_org[t]) * MRatio))[0]

        if len(left_mag_rel) > 0:
            left_mag_rel = left_mag_rel[-1]
            left_mag = Supp[t, 0] + left_mag_rel

            left_boundary = Right_org[left_mag]
            left_val, left_rel = np.min(Data_org[left_boundary:t] * np.sign(Mag_org[t])), np.argmin(Data_org[left_boundary:t] * np.sign(Mag_org[t]))
            Left[t] = left_boundary + left_rel

            if t in indices_to_check:
                print(f"Left terminal found:")
                print(f"  left_boundary: {left_boundary}")
                print(f"  left_val: {left_val}, left_rel: {left_rel}")
                print(f"  New Left[t]: {Left[t]}")

        # Find right terminal
        Mag_org_rel = Mag_org[t + 1 : Supp[t, 1]]

        right_boundary_rel = np.where((Mag_org_rel * np.sign(Mag_org[t]) > 0)& (np.abs(Mag_org_rel) > np.abs(Mag_org[t]) * MRatio))[0]

        if len(right_boundary_rel) > 0:
            right_boundary_rel = right_boundary_rel[0]
            right_mag = t + 1 + right_boundary_rel
            right_boundary = Left_org[right_mag]
            right_val, right_rel = np.min(Data_org[t:right_boundary] * np.sign(Mag_org[t])), np.argmin(Data_org[t:right_boundary] * np.sign(Mag_org[t]))
            Right[t] = t + 1 + right_rel

            if t in indices_to_check:
                print(f"Right terminal found:")
                print(f"  right_boundary: {right_boundary}")
                print(f"  right_val: {right_val}, right_rel: {right_rel}")
                print(f"  New Right[t]: {Right[t]}")

    # Ensure Mag_left and Mag_right have the same shape
    Mag_left = Data_org - Data_org[Left]
    Mag_right = Data_org - Data_org[Right]

    # Revise magnitude with the same length as the original Mag_org
    Mag = np.sign(Mag_left) * np.minimum(np.abs(Mag_left), np.abs(Mag_right))

    for t in indices_to_check:
        if t < len(Mag):
            print(f"\nFinal Values for Index {t}:")
            print(f"  Mag_left: {Mag_left[t]}")
            print(f"  Mag_right: {Mag_right[t]}")
            print(f"  Final Mag[t]: {Mag[t]}")
        else:
            print(f"Index {t} is out of bounds for the calculated arrays.")

    return Mag, Left, Right


def plot_range(X, Ratio):
    max_x = np.max(X)
    min_x = np.min(X)

    Max = max_x + Ratio * (max_x - min_x)
    Min = min_x - Ratio * (max_x - min_x)

    return [Min, Max]


def Spikelet_restrictSupportByMagnitudeRatioInitial(MagInfo):
    # Mock function for initial magnitude ratio support restriction
    return MagInfo


def Spikelet_Op_extractConstantSegment(MagInfo):
    # Mock function for extracting constant segments
    return MagInfo


def Spikelet_MagInfo_get_initial(MagInfo, Param):
    InitialOpName = "generateInitialSpikelet"

    # Initialize the MagInfo_initial dictionary with all the required keys
    MagInfo_initial = {
        "type": MagInfo["type"].copy(),
        "value": MagInfo["value"].copy(),
        "magnitude": MagInfo["magnitude"].copy(),
        "leg_magnitude": MagInfo["leg_magnitude"].copy(),
        "left": MagInfo["left"].copy(),
        "right": MagInfo["right"].copy(),
        "data": MagInfo["data"].copy(),
        "center": MagInfo["center"].copy(),
        "output": {InitialOpName: MagInfo["output"][InitialOpName].copy()},
        "data_org": MagInfo["data_org"].copy(),
        "size": {"original": MagInfo["size"]["original"]},
        "time": {InitialOpName: MagInfo["time"][InitialOpName]},
        "param": Param,
    }

    return MagInfo_initial


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


def Spikelet_MagInfo_post_processing(MagInfo, OpName):
    """
    Post-processing function for Spikelet decomposition.

    Parameters
    ----------
    MagInfo (dict): A dictionary containing the Spikelet decomposition information.
    OpName (str): The operation name to be used for output.

    Returns
    -------
    dict: Updated MagInfo dictionary after post-processing.
    """
    if "output" in MagInfo:
        # Latest dependent fields
        MagInfo["center"] = np.where(~np.isnan(MagInfo["type"]))[0]
        MagInfo["data"] = Spikelet_get_TimeSeriesForm_from_SpikeletDecomposition(
            MagInfo
        )

        # Output fields (independent)
        MagInfo["output"][OpName] = {
            "type": MagInfo["type"],
            "value": MagInfo["value"],
            "magnitude": MagInfo["magnitude"],
            "leg_magnitude": MagInfo["leg_magnitude"],
            "left": MagInfo["left"],
            "right": MagInfo["right"],
            # Dependent
            "data": MagInfo["data"],
            "center": MagInfo["center"],
        }

    else:
        MagInfo["operation"][OpName] = {
            "data": MagInfo["data"],
            "magnitude": MagInfo["magnitude"],
            "left": MagInfo["left"],
            "right": MagInfo["right"],
        }
    return MagInfo


def Spikelet_get_TimeSeriesForm_from_SpikeletDecomposition(MagInfo):
    """
    Generate the time series form from Spikelet decomposition.

    Parameters
    ----------
    MagInfo (dict): A dictionary containing the Spikelet decomposition information.

    Returns
    -------
    np.ndarray: The reconstructed time series data.
    """
    Data_org = MagInfo["data_org"]
    Center = np.where(~np.isnan(MagInfo["type"]))[0]
    Left = MagInfo["left"][Center]
    Right = MagInfo["right"][Center]

    Apx_time = np.unique(
        np.concatenate(([0], Left, Center, Right, [len(Data_org) - 1]))
    )

    if len(Apx_time) >= 2:
        Data = np.interp(np.arange(len(Data_org)), Apx_time, Data_org[Apx_time])
    else:
        Data = Data_org.copy()

    Center_const = np.where(MagInfo["type"] == 0)[0]

    for i in Center_const:
        from_idx = MagInfo["left"][i]
        to_idx = MagInfo["right"][i]
        value = MagInfo["value"][i]
        Data[from_idx : to_idx + 1] = value

    return Data
