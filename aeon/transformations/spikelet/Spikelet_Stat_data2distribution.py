import numpy as np
import matplotlib.pyplot as plt

from aeon.transformations.spikelet.Spikelet_Stat_decide_binsize import Spikelet_Stat_decide_binsize

def Spikelet_Stat_data2distribution(D, Arg=None, StepDivLength=10):
    FuncName = 'Spikelet_Stat_data2distribution'
    
    # Debug flag
    DEBUG = False

    # Ensure Arg is a dictionary
    if Arg is None or not isinstance(Arg, dict):
        Arg = {'weight': np.nan}

    Weight = Arg.get('weight', np.nan)
    BinMethod = Arg.get('BinMethod', 'auto')
    
    # Decide bin size
    BinSize, edges, BinListInfo = Spikelet_Stat_decide_binsize(D, BinMethod)
    
    if Arg.get('edge_type') == 'int':
        Edge_interval = np.ceil(BinSize)
    else:
        Max = np.max(D)
        Min = np.min(D)
        Edge_interval = (Max - Min) / BinSize

    StepWidth = Arg.get('StepWidth', Edge_interval / StepDivLength)

    # Mean average
    if np.max(D) != np.min(D):
        Y, X = spikelet_stat_meanAverage(D, Edge_interval, StepWidth, Weight)
    else:
        Y = len(D)
        X = np.array([D[0]])
    
    return Y, X, BinSize, StepDivLength, BinListInfo

def spikelet_stat_meanAverage(CLD, EdgeInterval, StepWidth, Weight=np.nan):
    DEBUG = False
    
    Max = np.max(CLD)
    Min = np.min(CLD)
    X = np.arange(Min, Max + StepWidth, StepWidth)
    Y = np.empty(len(X))
    
    for i in range(len(X)):
        if i == len(X) - 1:
            Index_i = (X[i] - EdgeInterval / 2 <= CLD) & (CLD < X[i] + EdgeInterval / 2)
        else:
            Index_i = (X[i] - EdgeInterval / 2 <= CLD) & (CLD <= X[i] + EdgeInterval / 2)
        
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
