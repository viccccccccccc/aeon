import time
import pdb
import pickle

import numpy as np

from aeon.transformations.spikelet.spikelet_generateInitialSpikelet import (
    Spikelet_generateInitialSpikelet,
)
from aeon.transformations.spikelet.spikelet_Op_call_approximation import (
    Spikelet_Op_call_approximation,
    Spikelet_Op_extractConstantSegment,
    Spikelet_Op_reduceSpikeByMagnitude,
    Spikelet_reduceSpikeByMagnitudeRatio,
    Spikelet_reduceSpikeByQuery,
    Spikelet_restrictSupportByMagnitudeRatio,
    Spikelet_restrictSupportByMagnitudeRatioInitial,
    Spikelet_restrictSupportByWindowLength,
)


def Spikelet_aproximation_ver_03(X, Param):
    print("-------------------------spikelet_approx BEGIN------------------------")

    # Debug flags
    FigOn_seg = False
    FigOn_leg = False
    Debug_On = False
    FuncName = "Spikelet_aproximation_ver_03"

    # Initial spikelet generation
    start_SAt = time.time()

    MagInfo = Spikelet_generateInitialSpikelet(X, float("inf"))

    # Set data name
    MagInfo["dataname"] = Param.get("dataname", "unknown")

    # Initial operation size and time
    InitialOpName = "generateInitialSpikelet"
    MagInfo["size"] = {"original": len(X)}
    MagInfo["time"] = {InitialOpName: time.time() - start_SAt}

    if isinstance(MagInfo["output"], list):
        MagInfo["output"] = {InitialOpName: {"type": np.ones(len(X))}}

    MagInfo["size"][InitialOpName] = np.sum(
        ~np.isnan(MagInfo["output"][InitialOpName]["type"])
    )
    MagInfo["param"] = Param

    OLD_ON = False

    # Call operation sequence
    if not OLD_ON:
        MagInfo = Spikelet_Op_call_approximation(MagInfo)

    if OLD_ON:
        Ops = Param["operation"]["operation_sequence"]
        for Op in Ops:
            start_opt = time.time()
            # Spike reduction
            if Op == "reduceSpikeByMagnitude":
                MagInfo = Spikelet_Op_reduceSpikeByMagnitude(MagInfo)
                Query = MagInfo["output"][Op]["query"]
                print(f"{Op}: Query = {Query}")

            elif Op == "reduceSpikeByMagnitudeRatio":
                MagInfo = Spikelet_reduceSpikeByMagnitudeRatio(MagInfo)

            elif Op == "reduceSpikeByQuery":
                MagInfo = Spikelet_reduceSpikeByQuery(MagInfo)

            # Support restriction
            elif Op == "restrictSupportByWindowLength":
                MagInfo = Spikelet_restrictSupportByWindowLength(MagInfo)

            elif Op == "restrictSupportByMagnitudeRatio":
                MagInfo = Spikelet_restrictSupportByMagnitudeRatio(MagInfo)

            elif Op == "restrictSupportByMagnitudeRatioInitial":
                MagInfo = Spikelet_restrictSupportByMagnitudeRatioInitial(MagInfo)

            # Constant segment extraction
            elif Op == "extractConstantSegment":
                MagInfo = Spikelet_Op_extractConstantSegment(MagInfo)
                CLThr = MagInfo["output"][Op]["length_threshold"]
                print(f"{Op}: Constant Length Threshold = {CLThr}")

            else:
                print(f"unknown operation ({Op})")

            # Operation time and size
            MagInfo["time"][Op] = time.time() - start_opt
            MagInfo["size"][Op] = np.sum(~np.isnan(MagInfo["output"][Op]["type"]))
            print(f"{Op}: time = {MagInfo['time'][Op]}, size = {MagInfo['size'][Op]}")

        # Total time
        Total_time = MagInfo["time"][InitialOpName]
        for Op in Ops:
            Total_time += MagInfo["time"][Op]
        MagInfo["time"]["operation_all"] = Total_time
        MagInfo["size"]["operation_all"] = MagInfo["size"][Ops[-1]]

    end_SAt = time.time()
    print("-------------------------spikelet_approx END------------------------")
    return MagInfo
