import numpy as np
import matplotlib.pyplot as plt

from aeon.transformations.spikelet.Spikelet_Stat_decide_binsize import Spikelet_Stat_decide_binsize
from aeon.transformations.spikelet.Spikelet_Stat_knee_find_poly_for_XY import Spikelet_Stat_knee_find_poly_for_XY

def Spikelet_Stat_knee_find_of_mag_using_magband_support(M, S, Func, BandFunc):
    FuncName = 'Spikelet_Stat_knee_find_of_mag_using_magband_support'
    PLOT = True
    
    BinNumSelection = 'auto'
    if isinstance(Func, str):
        Func_fwd, Dim_fwd, Func_bwd, Dim_bwd = extract_fb_function(Func)
    else:
        print(f'[{FuncName}] Cell type of Func is not implemented yet')
        return None, None

    # Decide bin size
    BinN, M_Edges = Spikelet_Stat_decide_binsize(M, BinNumSelection)
    Band = edge_band(M_Edges, M, S, BandFunc)
    
    # Knee find
    M_band = Band[:, 0]
    S_band = Band[:, 1]
    
    if Func_fwd == 'poly' and Func_bwd == 'poly':
        MagThr, KneeInfo = Spikelet_Stat_knee_find_poly_for_XY(M_band, S_band, Dim_fwd, Dim_bwd)
    else:
        print(f'[{FuncName}] Funcs except for poly_poly are not implemented yet')
        return None, None
    
    if PLOT and Func_fwd == 'poly' and Func_bwd == 'poly':
        fig_1, fig_2, fig_3 = plot_scatter_MS(M, S, M_band, S_band, MagThr, KneeInfo, BandFunc)
    
    return MagThr, KneeInfo

def extract_fb_function(Func):
    C = Func.split('_')
    Func_fwd = C[0]
    Dim_fwd = int(C[1])
    Func_bwd = C[2]
    Dim_bwd = int(C[3])
    return Func_fwd, Dim_fwd, Func_bwd, Dim_bwd

def edge_band(Edges, M, S, Func):
    Band = np.full((len(Edges) - 1, 2), np.nan)
    for i in range(len(Edges) - 1):
        from_edge = Edges[i]
        to_edge = Edges[i + 1]
        if i == len(Edges) - 1:
            Index = (from_edge <= M) & (to_edge <= M)
        else:
            Index = (from_edge <= M) & (to_edge < M)
        S_i = S[Index]
        if Func == 'std':
            Band[i, :] = [(from_edge + to_edge) / 2, np.std(S_i)]
        elif Func == 'sum':
            Band[i, :] = [(from_edge + to_edge) / 2, np.sum(S_i)]
    return Band

def plot_scatter_MS(M, S, M_band, S_band, MagThr, KneeInfo, BandFunc):
    KeeOpt = KneeInfo['knee_opt']
    Index_bwd = M >= KeeOpt
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
    plt.title(f'magnitude bwd ({KeeOpt})')
    plt.subplot(Row, Column, 3)
    plt.hist(S)
    plt.title('support')
    plt.subplot(Row, Column, 4)
    plt.hist(S_bwd)
    plt.title(f'support (magcut: {KeeOpt})')

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

    plt.subplot(Row, Column, 7)
    plt.plot(M_band, S_band)
    plt.xlabel('M-band')
    plt.ylabel(f'S-{BandFunc}')
    plt.title(f'mag-supp knee={MagThr}')

    fig_2 = plt.figure()
    plt.imshow(MS_matrix, aspect='auto', origin='lower')
    plt.xlabel('support')
    plt.ylabel('magnitude')
    plt.title('mag-supp')
    
    fig_3 = plt.figure()
    plt.imshow(MS_matrix_bwd, aspect='auto', origin='lower')
    plt.xlabel('support')
    plt.ylabel('magnitude')
    plt.title('mag-supp-bwd')

    return fig_1, fig_2, fig_3

def Spikelet_Stat_scatter2matrix(M, S):
    # This function needs to be implemented based on the actual requirement.
    # For now, let's assume it converts scatter data to a matrix form.
    # Replace with the actual logic.
    print("hier sollte ich nicht sein (Spikelet_Stat_scatter2matrix)")
    return np.vstack((M, S)).T