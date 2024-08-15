import numpy as np
import matplotlib.pyplot as plt

from aeon.transformations.spikelet.Spikelet_Stat_knee_find import (
    Spikelet_Stat_knee_find,
    Spikelet_Stat_knee_find_double
)

def Spikelet_Stat_knee_find_of_mag_using_support_knee(M, S, Support_knee_func, Func, Weight=None):
    FuncName = 'Spikelet_Stat_knee_find_of_mag_using_magband_support'
    PLOT = False
    
    if Weight is None:
        Weight = np.nan
    
    if isinstance(Support_knee_func, (list, str)):
        if len(Support_knee_func) == 1:
            KneeOpt_S, Info_S = Spikelet_Stat_knee_find(S, Support_knee_func)  # no weight
        elif len(Support_knee_func) == 2:
            KneeOpt_S, Info_S = Spikelet_Stat_knee_find_double(S, Support_knee_func)  # no weight
        else:
            print(f'[{FuncName}] unknown support_knee_func length ({len(Support_knee_func)})')
            return None, None
    
    if isinstance(Support_knee_func, (list, str)):
        Index_scut = S >= KneeOpt_S
        M_scut = M[Index_scut]
        if not np.isnan(Weight).all():
            Weight = Weight[Index_scut]
    else:
        M_scut = M
    
    if len(Func) == 1:
        MagThr, KneeInfo = Spikelet_Stat_knee_find(M_scut, Func, Weight)
    elif len(Func) == 2:
        MagThr, KneeInfo = Spikelet_Stat_knee_find_double(M_scut, Func, Weight)
    else:
        print(f'[{FuncName}] unknown Func length ({len(Func)})')
        return None, None
    
    if PLOT:
        plot_scatter_MS(M, S, MagThr)
    
    return MagThr, KneeInfo

def extract_fb_function(Func):
    C = Func.split('_')
    Func_fwd = C[0]
    Dim_fwd = int(C[1])
    Func_bwd = C[2]
    Dim_bwd = int(C[3])
    return Func_fwd, Dim_fwd, Func_bwd, Dim_bwd

def plot_scatter_MS(M, S, MagThr):
    KneeOpt = MagThr
    
    Index_bwd = M >= KneeOpt
    M_bwd = M[Index_bwd]
    S_bwd = S[Index_bwd]
    
    MS_matrix_org = Spikelet_Stat_scatter2matrix(M, S)
    MS_matrix = np.flipud(MS_matrix_org)
    MS_matrix_bwd_org = Spikelet_Stat_scatter2matrix(M_bwd, S_bwd)
    MS_matrix_bwd = np.flipud(MS_matrix_bwd_org)
    
    fig_1 = plt.figure()
    Row, Column = 2, 4
    
    plt.subplot(Row, Column, 1)
    plt.hist(M)
    plt.title('magnitude')
    
    plt.subplot(Row, Column, 2)
    plt.hist(M_bwd)
    plt.title(f'magnitude bwd ({KneeOpt})')
    
    plt.subplot(Row, Column, 3)
    plt.hist(S)
    plt.title('support')
    
    plt.subplot(Row, Column, 4)
    plt.hist(S_bwd)
    plt.title(f'support ( magcut: {KneeOpt})')
    
    plt.subplot(Row, Column, 5)
    plt.scatter(S, M)
    plt.xlabel('support')
    plt.ylabel('magnitude')
    plt.title('mag-supp')
    
    plt.subplot(Row, Column, 6)
    plt.scatter(S_bwd, M_bwd)
    plt.xlabel('support')
    plt.ylabel('magnitude')
    plt.title('mag-supp-bwd')
    
    fig_2 = plt.figure()
    plt.imshow(MS_matrix, aspect='auto')
    plt.xlabel('support')
    plt.ylabel('magnitude')
    plt.title('mag-supp')
    
    fig_3 = plt.figure()
    plt.imshow(MS_matrix_bwd, aspect='auto')
    plt.xlabel('support')
    plt.ylabel('magnitude')
    plt.title('mag-supp-bwd')
    
    return fig_1, fig_2, fig_3

def Spikelet_Stat_scatter2matrix(M, S):
    # This function needs to be implemented based on the actual requirement.
    # For now, let's assume it converts scatter data to a matrix form.
    # Replace with the actual logic.
    return np.vstack((M, S)).T

