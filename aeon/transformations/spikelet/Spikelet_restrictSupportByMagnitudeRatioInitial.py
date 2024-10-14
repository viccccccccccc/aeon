import pdb
import pickle

import numpy as np
import matplotlib.pyplot as plt

from aeon.transformations.spikelet.Spikelet_MagInfo_post_processing import Spikelet_MagInfo_post_processing

def Spikelet_restrictSupportByMagnitudeRatioInitial(MagInfo):
    Debug_On = False
    FuncName = "Spikelet_restrictSupportByMagnitudeRatioInitial"
    OpName = "restrictSupportByMagnitudeRatioInitial"

    # Parameter extrahieren
    ParamSRM = MagInfo["param"]["operation"][OpName]
    MRatio = ParamSRM["magnitude_ratio"]

    # Input und Output
    Data_org = MagInfo["data_org"]
    Mag = MagInfo["magnitude"]
    Left = MagInfo["left"]
    Right = MagInfo["right"]

    # Initialwerte laden
    MagInfo_initial = MagInfo["output"]["generateInitialSpikelet"]
    Mag_initial = MagInfo_initial["magnitude"]
    Left_initial = MagInfo_initial["left"]
    Right_initial = MagInfo_initial["right"]

    # Magnitude-Beschränkung anwenden
    
    Mag_new, Left_new, Right_new = support_restriction_by_magnitude_ratio_initial(Data_org, Mag, Left, Right, Mag_initial, ParamSRM)

    # Überarbeitete Felder setzen
    MagInfo["magnitude"] = Mag_new
    MagInfo["left"] = Left_new
    MagInfo["right"] = Right_new
    
    # Nachbearbeitung
    MagInfo = Spikelet_MagInfo_post_processing(MagInfo, OpName)
    MagInfo["output"][OpName]["magnitude_ratio_threshold"] = MRatio
    MagInfo["output"][OpName]["param_str"] = f"magnitude_ratio_threshold = {MRatio}"

    # Debugging-Plot, falls aktiviert
    if Debug_On:
        Mag_org = Mag
        Supp_org = Right - Left
        Supp_new = Right_new - Left_new
        Index = np.where(Supp_org != Supp_new)[0]

        Row = 5
        plt.figure()
        plt.subplot(Row, 1, 1)
        plt.plot(Data_org)
        plt.xlim([1, len(Data_org)])
        plt.title("data original")
        for i in Index:
            plt.axvline(x=i, color="r")

        plt.subplot(Row, 1, 2)
        plt.plot(Mag_org)
        plt.xlim([1, len(Mag_org)])
        plt.title("magnitude before")
        MagRange = [np.min(Mag_org), np.max(Mag_org)]
        plt.ylim(MagRange)

        plt.subplot(Row, 1, 3)
        plt.plot(Mag_new)
        plt.xlim([1, len(Mag_new)])
        plt.title("magnitude after")
        plt.ylim(MagRange)

        plt.subplot(Row, 1, 4)
        Supp_org_signed = np.sign(Mag_org) * Supp_org
        plt.plot(Supp_org_signed)
        plt.xlim([1, len(Supp_org)])
        plt.title("support length before")
        SuppRange = [np.min(Supp_org_signed), np.max(Supp_org_signed)]
        plt.ylim(SuppRange)

        plt.subplot(Row, 1, 5)
        Supp_new_signed = np.sign(Mag_new) * Supp_new
        plt.plot(Supp_new_signed)
        plt.xlim([1, len(Supp_new_signed)])
        plt.title("support length after")
        plt.ylim(SuppRange)

        plt.show()

    return MagInfo


def support_restriction_by_magnitude_ratio_initial(Data_org, Mag_org, Left_org, Right_org, Mag_initial, ParamSRM):
    MRatio = ParamSRM["magnitude_ratio"]
    Supp = np.column_stack((Left_org, Right_org))

    # Ausgabeinitialisierung
    Left = Left_org.copy()
    Right = Right_org.copy()

    # Hauptoperation
    M_pos = np.where(Mag_org != 0)[0]

    for t in M_pos:
        # Finde linkes Terminal
        Mag_org_rel = Mag_org[Supp[t, 0]: t]
        Mag_initial_rel = Mag_initial[Supp[t, 0]: t]
        left_mag_rel = np.where((Mag_initial_rel * np.sign(Mag_org[t]) > 0) & (np.abs(Mag_initial_rel) > np.abs(Mag_org[t]) * MRatio))[0]

        if len(left_mag_rel) > 0:
            left_mag = Supp[t, 0] + left_mag_rel[-1]
            left_boundary = Right_org[left_mag]
            left_val, left_rel = np.min(Data_org[left_boundary: t] * np.sign(Mag_org[t])), np.argmin(Data_org[left_boundary: t] * np.sign(Mag_org[t]))
            Left[t] = left_boundary + left_rel

        # Finde rechtes Terminal
        start_idx = t + 1
        end_idx = int(Supp[t, 1]) + 1  # +1, um das Endelement einzuschließen
        Mag_org_rel = Mag_org[start_idx: end_idx]
        Mag_initial_rel = Mag_initial[start_idx: end_idx]
        condition = (Mag_initial_rel * np.sign(Mag_org[t]) > 0) & (np.abs(Mag_initial_rel) > np.abs(Mag_org[t]) * MRatio)
        right_mag_rel = np.where(condition)[0]

        if len(right_mag_rel) > 0:
            right_mag = start_idx + right_mag_rel[0]
            right_boundary = int(Left_org[right_mag])

            if right_boundary >= t:
                data_range = Data_org[t: right_boundary + 1] * np.sign(Mag_org[t])

                if len(data_range) > 0:
                    right_rel = np.argmin(data_range)
                    Right[t] = t + right_rel
                else:
                    Right[t] = t
            else:
                Right[t] = t
        else:
            # Falls keine passenden Werte gefunden wurden, bleibt Right[t] unverändert
            pass

    # Magnitude überarbeiten
    Mag_left = np.array(Data_org) - np.array(Data_org[Left])
    Mag_right = np.array(Data_org) - np.array(Data_org[Right])
    Mag = np.sign(Mag_left) * np.minimum(np.abs(Mag_left), np.abs(Mag_right))

    return Mag, Left, Right
