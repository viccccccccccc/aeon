import time
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aeon.transformations.spikelet.spikelet_Stat_magnitude_threshold_SuppMax_SuppMRatio import Spikelet_Stat_magnitude_threshold_SuppMax_SuppMRatio
from aeon.transformations.spikelet.Spikelet_restrictSupportByMagnitudeRatioInitial import Spikelet_restrictSupportByMagnitudeRatioInitial
from aeon.transformations.spikelet.Spikelet_MagInfo_post_processing import Spikelet_MagInfo_post_processing
from aeon.transformations.spikelet.Spikelet_restrictSupportByWindowLength import Spikelet_restrictSupportByWindowLength
from aeon.transformations.spikelet.Spikelet_restrictSupportByMagnitudeRatio import Spikelet_restrictSupportByMagnitudeRatio

def print_structure(d, indent=0):
    """Recursively prints the structure of a dictionary."""
    for key, value in d.items():
        print(" " * indent + str(key) + ":", end=" ")
        if isinstance(value, dict):
            print()
            print_structure(value, indent + 2)
        elif isinstance(value, list):
            print("[", end="")
            print(", ".join(str(type(i)) for i in value), end="]\n")
        else:
            print(type(value))


def Spikelet_Op_call_approximation(MagInfo):
    InitialOpName = "generateInitialSpikelet"
    Ops = MagInfo["param"]["operation"]["operation_sequence"]

    for Op in Ops:
        start_time = time.time()

        # 1. Spike reduction
        if Op == "reduceSpikeByMagnitude":  # revised version for filterByMagnitude
            MagInfo = Spikelet_Op_reduceSpikeByMagnitude(MagInfo)
            Query = MagInfo["output"][Op]["query"]

        elif Op == "reduceSpikeByMagnitudeRatio":  # revised version for reduceLocalSpike
            MagInfo = Spikelet_reduceSpikeByMagnitudeRatio(MagInfo)

        elif Op == "reduceSpikeByQuery":
            MagInfo = Spikelet_reduceSpikeByQuery(MagInfo)

        # 2. Support restriction
        elif Op == "restrictSupportByWindowLength":
            MagInfo = Spikelet_restrictSupportByWindowLength(MagInfo)

        elif Op == "restrictSupportByMagnitudeRatio":                    
            MagInfo = Spikelet_restrictSupportByMagnitudeRatio(MagInfo)
    
        elif Op == "restrictSupportByMagnitudeRatioInitial":
            MagInfo = Spikelet_restrictSupportByMagnitudeRatioInitial(MagInfo)
            file_path = r'C:\Users\Victor\Desktop\Uni\Bachelor\stuff\MagInfoo.npy'

            # Save the array to the specified path
            np.save(file_path, MagInfo["magnitude"])

            print(f"Array saved to {file_path}")
            

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

            MagThr, MagDist, MagInfo_auto, KneeInfo = (Spikelet_Stat_magnitude_threshold_SuppMax_SuppMRatio(MagInfo_tmp, Param_Auto))

        Query = f"abs(Mag) > {MagThr}"
    else:
        print(f"[{FuncName}] Unknown Method ({Method})")

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
    MagInfo["output"][OpName].update({
        "magnitude_threshold": MagThr,
        "query": Query,
        "param_str": Query,
    })

    if "KneeInfo" in locals():
        MagInfo["output"][OpName]["magnitude_threshold_kneeInfo"] = KneeInfo

    return MagInfo

def Spikelet_eval_query(Query, MagInfo):
    Mag = MagInfo['magnitude']
    Mag_index = np.where(Mag != 0)[0]
    
    Val = MagInfo['value']
    Supp = MagInfo['right'] - MagInfo['left'] + 1
    
    Q_index = eval(Query)
    Q_time = np.where(Q_index == True)[0]
    
    Reduced_time = np.setdiff1d(MagInfo['center'], Q_time)

    return Q_time, Reduced_time

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

def Spikelet_reduceSpikeByMagnitudeRatio(MagInfo):
    # Mock function for spike reduction by magnitude ratio
    return MagInfo

def Spikelet_reduceSpikeByQuery(MagInfo):
    # Mock function for spike reduction by query
    return MagInfo

def Spikelet_Op_extractConstantSegment(MagInfo):
    # Mock function for extracting constant segments
    return MagInfo
