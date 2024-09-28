import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal, getcontext


from aeon.transformations.spikelet.Spikelet_Stat_decide_binsize import Spikelet_Stat_decide_binsize


def Spikelet_Stat_data2distribution(D, Arg=None, StepDivLength=10):
    FuncName = "Spikelet_Stat_data2distribution"

    # Debug flag
    DEBUG = False

    # Ensure Arg is a dictionary
    if Arg is None or not isinstance(Arg, dict):
        Arg = {"weight": np.nan}

    Weight = Arg.get("weight", np.nan)
    BinMethod = Arg.get("BinMethod", "auto")

    # Decide bin size
    BinSize, edges, BinListInfo = Spikelet_Stat_decide_binsize(D, BinMethod)

    if Arg.get("edge_type") == "int":
        Edge_interval = np.ceil(BinSize)
    else:
        Max = np.max(D)
        Min = np.min(D)
        Edge_interval = (Max - Min) / BinSize

    StepWidth = Arg.get("StepWidth", Edge_interval / StepDivLength)

    # Mean average
    if np.max(D) != np.min(D):
        Y, X = spikelet_stat_meanAverage(D, Edge_interval, StepWidth, Weight)
    else:
        Y = len(D)
        X = np.array([D[0]])

    return Y, X, BinSize, StepDivLength, BinListInfo

# WICHTIG: in dieser Funktion werden Vergleiche mit extrem hoher Präzision in der ersten if abfrage in Matlab durchgeführt.
# Bis jetzt habe ich keinen Weg gefunden diese Präzision in Python nachzustellen, dementsprechend ist das Y Array was zurückgegeben
# wird ganz leicht anders (wir reden von ganz vielen Nachkommastellen). Ich habe erstmal gerundet und weil diese Lösung am besten scheint.
def spikelet_stat_meanAverage(CLD, EdgeInterval, StepWidth, Weight=np.nan):
    print("potentielle Baustelle in Spikelet_Stat_data2distribution.py, Funktion: spikelet_stat_meanAverage(). Mehr Informationen in Kommentare.")
    DEBUG = False
    
    Max = np.max(CLD)
    Min = np.min(CLD)

    CLD = np.array(CLD, dtype=np.float64)

    EdgeInterval = np.float64(EdgeInterval)
    StepWidth = np.float64(StepWidth)

    X = np.arange(Min, Max + StepWidth / 2, StepWidth, dtype=np.float64)
    X = np.array(X, dtype=np.float64)
    Y = np.empty(len(X), dtype=np.float64)

    tolerance = 1e-12

    for i in range(len(X)):

        lower_bound = X[i] - EdgeInterval / 2
        upper_bound = X[i] + EdgeInterval / 2

        if i == len(X) - 1:
            #Index_i = ((X[i] - EdgeInterval / 2) <= CLD) & (CLD < (X[i] + EdgeInterval / 2))
            Index_i = (np.isclose(lower_bound, CLD, atol=tolerance) | (lower_bound < CLD)) & (CLD < upper_bound)
        else:
            #Index_i = ((X[i] - EdgeInterval / 2) <= CLD) & (CLD <= (X[i] + EdgeInterval / 2))
            Index_i = (np.isclose(lower_bound, CLD, atol=tolerance) | (lower_bound < CLD)) & (np.isclose(CLD, upper_bound, atol=tolerance) | (CLD <= upper_bound))

        Ratio = 1
        if (X[i] - Min) < EdgeInterval / 2:
            Ratio = EdgeInterval / ((X[i] - Min) + EdgeInterval / 2)
        elif (Max - X[i]) < EdgeInterval / 2:
            Ratio = EdgeInterval / ((Max - X[i]) + EdgeInterval / 2)

        if np.isnan(Weight).all():
            Y[i] = np.sum(Index_i) * Ratio
        else:
            Y[i] = np.sum(Weight[Index_i]) * Ratio

    if DEBUG:
        plt.figure()
        plt.plot(X, Y)
        plt.xlim([X[0], X[-1]])
        plt.show()

    return Y, X
