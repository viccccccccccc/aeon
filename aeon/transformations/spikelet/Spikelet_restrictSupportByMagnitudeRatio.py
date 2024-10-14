import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aeon.transformations.spikelet.Spikelet_MagInfo_post_processing import Spikelet_MagInfo_post_processing

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

    for t in M_pos:
        # Find left terminal
        Mag_org_rel = Mag_org[Supp[t, 0] : t]

        left_mag_rel = np.where((Mag_org_rel * np.sign(Mag_org[t]) > 0) & (np.abs(Mag_org_rel) > np.abs(Mag_org[t]) * MRatio))[0]

        if len(left_mag_rel) > 0:
            left_mag_rel = left_mag_rel[-1]
            left_mag = Supp[t, 0] + left_mag_rel

            left_boundary = Right_org[left_mag]
            left_val, left_rel = np.min(Data_org[left_boundary:t] * np.sign(Mag_org[t])), np.argmin(Data_org[left_boundary:t] * np.sign(Mag_org[t]))
            Left[t] = left_boundary + left_rel

        # Find right terminal
        Mag_org_rel = Mag_org[t + 1 : Supp[t, 1]]

        right_boundary_rel = np.where((Mag_org_rel * np.sign(Mag_org[t]) > 0)& (np.abs(Mag_org_rel) > np.abs(Mag_org[t]) * MRatio))[0]

        if len(right_boundary_rel) > 0:
            right_boundary_rel = right_boundary_rel[0]
            right_mag = t + 1 + right_boundary_rel
            right_boundary = Left_org[right_mag]
            right_val, right_rel = np.min(Data_org[t:right_boundary + 1] * np.sign(Mag_org[t])), np.argmin(Data_org[t:right_boundary + 1] * np.sign(Mag_org[t]))
            Right[t] = t + right_rel

    # Ensure Mag_left and Mag_right have the same shape
    Mag_left = Data_org - Data_org[Left]
    Mag_right = Data_org - Data_org[Right]

    # Revise magnitude with the same length as the original Mag_org
    Mag = np.sign(Mag_left) * np.minimum(np.abs(Mag_left), np.abs(Mag_right))

    return Mag, Left, Right