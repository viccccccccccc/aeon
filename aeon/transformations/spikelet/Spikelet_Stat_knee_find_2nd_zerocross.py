import numpy as np
import time
import matplotlib.pyplot as plt

from aeon.transformations.spikelet.Spikelet_Stat_knee_find import Spikelet_Stat_knee_find

def Spikelet_Stat_knee_find_2nd_zerocross(MagDist, FuncList, Weight=None):
    print("MagDist in Spikelet_Stat_knee_find_2nd_zerocross: ", len(MagDist))
    
    DEBUG = False
    PAPER = False

    FuncName = 'Spikelet_Stat_knee_find_2nd_zerocross'
    t_0 = time.time()
    t_0_name = FuncName

    if Weight is None:
        Weight = np.nan

    t_1 = time.time()
    t_1_name = 'first_knee_opt'
    KneeOpt_1, Info_1, BinListInfo_1 = Spikelet_Stat_knee_find(MagDist, FuncList[0], Weight)
    t_1end = time.time() - t_1

    t_2 = time.time()
    t_2_name = 'second_knee_opt'
    MagDist_2 = MagDist[MagDist >= KneeOpt_1]
    KneeOpt_2, Info_2, BinListInfo_2 = Spikelet_Stat_knee_find(MagDist_2, FuncList[1], Weight)
    t_2end = time.time() - t_2

    t_3 = time.time()
    t_3_name = 'zerocross'
    fwd_model_type_2 = Info_2['bwd_model_type']
    fwd_model_2 = Info_2['fwd_model']
    X_fwd = Info_2['X_fwd']
    Y_fwd = Info_2['Y_fwd']
    Opt_fwd = Info_2['opt'][:, 1]
    KneeOpt_zerocross_ref, InOut = zerocross_inout_region(Y_fwd, Opt_fwd)
    KneeOpt_zerocross = X_fwd[KneeOpt_zerocross_ref]
    t_3end = time.time() - t_3

    if len(InOut) > 0:
        KneeOpt = KneeOpt_zerocross
        aprx_level = 'zerocross'
    elif fwd_model_type_2 == 'poly' and len(fwd_model_2) == 2:
        if len(InOut) > 0:
            KneeOpt = KneeOpt_zerocross
            aprx_level = 'freq-over-bwd'
        elif fwd_model_2[0] >= 0:
            KneeOpt = KneeOpt_1
            aprx_level = 'first'
        else:
            KneeOpt = KneeOpt_2
            aprx_level = 'second'
    else:
        KneeOpt = KneeOpt_2
        aprx_level = 'second'

    t_0end = time.time() - t_0

    Info = {
        'knee_opt': KneeOpt,
        'aprx_level': aprx_level,
        'detail': [Info_1, Info_2],
        'bin': [BinListInfo_1, BinListInfo_2],
        'time': {
            t_0_name: t_0end,
            t_1_name: t_1end,
            t_2_name: t_2end,
            t_3_name: t_3end,
            'first_bin': BinListInfo_1['time_all'],
            'second_bin': BinListInfo_2['time_all'],
        }
    }

    if DEBUG or PAPER:
        fig_id1, fig_id2, fig_id3 = plot_zerocross(Info, PAPER)
        print(InOut)

    return KneeOpt, Info

def zerocross_inout_region(V, Pred):
    Vn = V - Pred
    In_pos = None
    Status = 0 if Vn[0] == 0 else (-1 if Vn[0] < 0 else 1)

    InOut = np.full((len(V), 2), np.nan)
    count = 0
    UpDown = np.sign(Vn)

    for i in range(len(UpDown)):
        if UpDown[i] * Status < 0:
            if Status == -1:
                if In_pos is not None:
                    count += 1
                    InOut[count - 1, :] = [In_pos, i - 1]
                    In_pos = None
            elif Status == 1:
                In_pos = i
            Status *= -1
        elif Status == 0 and UpDown[i] != 0:
            Status = UpDown[i]

    InOut = InOut[~np.isnan(InOut[:, 0])]

    KneeOpt_rel = len(V)
    if len(InOut) > 0:
        Min = 0
        for i in range(len(InOut)):
            min_i = np.min(Vn[int(InOut[i, 0]):int(InOut[i, 1])])
            pos_i = np.argmin(Vn[int(InOut[i, 0]):int(InOut[i, 1])])
            if min_i < Min:
                KneeOpt_rel = int(InOut[i, 0]) + pos_i
                Min = min_i

    return KneeOpt_rel, InOut

def plot_zerocross(Info_zc, PAPER):
    Info_1 = Info_zc['detail'][0]
    Info_2 = Info_zc['detail'][1]

    KneeOpt = Info_zc['knee_opt']
    Aprx_level = Info_zc['aprx_level']

    fig_id1 = plot_histPdfKnee(Info_1, PAPER)
    fig_id2 = plot_histPdfKnee(Info_2, PAPER)
    fig_id3 = plot_2ndzerocross(Info_zc)

    return fig_id1, fig_id2, fig_id3

def plot_2ndzerocross(Info_zc):
    KneeOpt = Info_zc['knee_opt']
    Info_2 = Info_zc['detail'][1]

    KneeOpt_2 = Info_2['knee_opt']
    X = Info_2['raw_smoothed'][:, 0]
    Y = Info_2['raw_smoothed'][:, 1]
    X_opt = Info_2['opt'][:, 0]
    Y_opt = Info_2['opt'][:, 1]
    X_fwd = X[X <= KneeOpt_2]
    Y_fwd = Y[X <= KneeOpt_2]
    X_fwd_opt = X_opt[X <= KneeOpt_2]
    Y_fwd_opt = Y_opt[X <= KneeOpt_2]

    plt.figure()
    plt.plot(X_fwd, Y_fwd, linewidth=1.5)
    plt.plot(X_fwd_opt, Y_fwd_opt, 'r--')
    plt.axvline(x=KneeOpt, color='r')
    plt.title(f'Zerocross opt: Knee={KneeOpt} 2nd knee={KneeOpt_2}')
    plt.show()

def plot_histPdfKnee(Info, PAPER):
    RawDist = Info['raw']
    BinSize = Info['bin_size']
    StepDivLength = Info['step_div_length']
    X = Info['raw_smoothed'][:, 0]
    Y = Info['raw_smoothed'][:, 1]
    X_opt = Info['opt'][:, 0]
    Y_opt = Info['opt'][:, 1]

    fig_id1 = plt.figure()
    plt.hist(RawDist, bins=BinSize)
    plt.ylim(np.min(RawDist), np.max(RawDist))
    plt.title(f'BinNum= {BinSize}')

    fig_id2 = plt.figure()
    plt.plot(X, Y, linewidth=1.5)
    plt.xlim([np.min(X), np.max(X)])
    plt.ylim([np.min(Y), np.max(Y)])
    plt.title(f'StepDivLength= {StepDivLength}')

    fig_id3 = plt.figure()
    plt.plot(X, Y, linewidth=1.5)
    plt.plot(X_opt, Y_opt, 'r--', linewidth=1)
    plt.axvline(x=Info['knee_opt'], color='r')
    plt.title(f'Knee point= {Info["knee_opt"]}')

    return fig_id1, fig_id2, fig_id3

def widen_lim(X, Ratio=0.1):
    Max = np.max(X)
    Min = np.min(X)
    W = (Max - Min) * Ratio
    return Min - W, Max + W

def plot_kee_find(Info):
    Info_1 = Info['detail'][0]
    Info_2 = Info['detail'][1]

    Knee_opt = Info['knee_opt']
    Aprx_level = Info['aprx_level']

    MagDist = Info_1['raw']
    KneeOpt_1 = Info_1['knee_opt']
    KneeOpt_2 = Info_2['knee_opt']
    ModelType_1 = Info_1['model_type']
    ModelType_2 = Info_2['model_type']
    X1 = Info_1['raw_smoothed'][:, 0]
    Y1 = Info_1['raw_smoothed'][:, 1]
    X1_opt = Info_1['opt'][:, 0]
    Y1_opt = Info_1['opt'][:, 1]

    X1_fwd = X1[X1 <= KneeOpt_1]
    Y1_fwd = Y1[X1 <= KneeOpt_1]
    X1_fwd_opt = X1_opt[X1 <= KneeOpt_1]
    Y1_fwd_opt = Y1_opt[X1 <= KneeOpt_1]

    X2 = Info_2['raw_smoothed'][:, 0]
    Y2 = Info_2['raw_smoothed'][:, 1]
    X2_opt = Info_2['opt'][:, 0]
    Y2_opt = Info_2['opt'][:, 1]

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.hist(MagDist)
    plt.subplot(3, 1, 2)
    plt.plot(X1, Y1)
    plt.plot(X1_opt, Y1_opt, 'r')
    plt.legend(['raw data', '1st'])
    plt.subplot(3, 1, 3)
    plt.plot(X1_fwd, Y1_fwd, linewidth=1.5)
    plt.plot(X1_fwd_opt, Y1_fwd_opt, 'r')
    plt.plot(X2, Y2)
    plt.plot(X2_opt, Y2_opt, 'r')
    plt.axhline(y=np.max(Info_2['Y_bwd']), color='r', linestyle='--')
    plt.axvline(x=Knee_opt)
    plt.legend(['raw data', '1st', '2nd'])
    Title = f'kee_opt= {Knee_opt} aprx_level= {Aprx_level} [{ModelType_1} k1= {KneeOpt_1}] [{ModelType_2} k2= {KneeOpt_2}]'
    plt.suptitle(Title.replace('_', '\_'))
    plt.show()