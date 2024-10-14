import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import scipy.io
import os

from aeon.transformations.spikelet.Spikelet_Stat_data2distribution import (
    Spikelet_Stat_data2distribution,
)


def Spikelet_Stat_knee_find(MagDist, Func, Weight=None):
    FuncName = "Spikelet_Stat_knee_find"
    DEBUG = False

    # Parameters
    ErrFunc = "l1"  # 'l2' is another option
    StepDivLength = 10

    # Argument processing
    Func_fwd, Dim_fwd, Func_bwd, Dim_bwd = extract_fb_function(Func)

    if "Weight" in locals():  # Check if 'Weight' exists in the local scope
        D2D_arg_weight = Weight
    else:
        D2D_arg_weight = np.nan

    # Spikelet_Stat_data2distribution needs to be defined
    Y, X, BinSize, StepDivLength, BinListInfo = Spikelet_Stat_data2distribution(MagDist, D2D_arg_weight, StepDivLength)

    # base_path = r'C:\Users\Victor\Desktop\Uni\Bachelor\stuff'

    # if len(MagDist) != 62744:
    #     file_path = os.path.join(base_path, f'vorlage_Y.mat')
    # else:
    #     file_path = os.path.join(base_path, f'vorlage_Y2.mat')
    
    # mat_data = scipy.io.loadmat(file_path)
    # Y = mat_data["Y"].squeeze()
    # print(f"----------------------Array 'Y' successfully loaded in Spikelet_Stat_knee_find----------------------")

    # islocalmin
    if Func_fwd == "islocalmin":
        TF = np.r_[False, (Y[1:-1] < Y[:-2]) & (Y[1:-1] < Y[2:]), False]
        P = Y - np.minimum(np.r_[np.inf, Y[:-1]], np.r_[Y[1:], np.inf])
        max_pos = np.argmax(P)
        KneeOpt = X[max_pos]
        Info = {
            "model_type": Func_fwd,
            "knee_opt": KneeOpt,
            "x": X,
            "y": Y,
            "localminimum": TF,
            "prominence": P,
        }

        if DEBUG:
            plt.figure()
            Row = 3
            plt.subplot(Row, 1, 1)
            plt.hist(MagDist)
            plt.title("original distribution")
            plt.subplot(Row, 1, 2)
            plt.plot(X, Y)
            plt.title("data2distribution")
            plt.subplot(Row, 1, 3)
            plt.plot(X, P)
            plt.title("prominence of local minimum")
            plt.suptitle(f"knee opt = {KneeOpt}")
            plt.show()
        return KneeOpt, Info, BinListInfo

    # Approximate function
    X_sorted = np.sort(X)

    Knee_list = X_sorted[Dim_fwd:-(1 + Dim_bwd)]

    Err = np.inf * np.ones_like(Knee_list)

    for i in range(len(Knee_list)):
        breakpt = Knee_list[i]
        Fwd_index = X <= breakpt
        MD_index = MagDist <= breakpt
        X_fwd = X[Fwd_index]
        Y_fwd = Y[Fwd_index]
        X_bwd = X[~Fwd_index]
        Y_bwd = Y[~Fwd_index]
        MD_fwd = MagDist[MD_index]
        MD_bwd = MagDist[~MD_index]

        YE_fwd, P_fwd = apploximate_XY_or_MD(X_fwd, Y_fwd, MD_fwd, Func_fwd, Dim_fwd)
        YE_bwd, P_bwd = apploximate_XY_or_MD(X_bwd, Y_bwd, MD_bwd, Func_bwd, Dim_bwd)

        YE = np.concatenate([YE_fwd, YE_bwd])
        if ErrFunc == "l1":
            Err[i] = np.sum(np.abs(YE - Y))
        elif ErrFunc == "l2":
            Err[i] = np.sum((YE - Y) ** 2)

    opt_error = np.min(Err)
    opt_pos = np.argmin(Err)
    KneeOpt = Knee_list[opt_pos]

    # Correction for third poly
    if Func_fwd == "poly" and Dim_fwd == 3:
        from_idx = X_fwd[0]
        to = Knee_list[opt_pos]
        Fwd_index = X <= to
        X_fwd, Y_fwd = X[Fwd_index], Y[Fwd_index]
        YE_fwd, P_fwd = apploximate_XY_or_MD(X_fwd, Y_fwd, MD_fwd, Func_fwd, Dim_fwd)

        if DEBUG:
            plt.figure()
            plt.subplot(2, 1, 2)
            plt.plot(X_fwd, Y_fwd)
            plt.plot(X_fwd, YE_fwd)
            MD_index = MagDist <= to
            plt.subplot(2, 1, 1)
            plt.hist(MagDist[MD_index])
            plt.show()

        a, b, c = P_fwd[0], P_fwd[1], P_fwd[2]
        ev = (-(2 * b) + np.sqrt((2 * b) ** 2 - 4 * (3 * a) * c)) / (2 * (3 * a))
        if a < 0 and from_idx <= ev <= to:
            KneeOpt = ev
        else:
            KneeOpt = Knee_list[opt_pos]

    # Output
    Fwd_index = X <= KneeOpt
    MD_index = MagDist <= KneeOpt
    X_fwd, Y_fwd = X[Fwd_index], Y[Fwd_index]
    X_bwd, Y_bwd = X[~Fwd_index], Y[~Fwd_index]
    MD_fwd, MD_bwd = MagDist[MD_index], MagDist[~MD_index]
    YE_fwd, P_fwd = apploximate_XY_or_MD(X_fwd, Y_fwd, MD_fwd, Func_fwd, Dim_fwd)
    YE_bwd, P_bwd = apploximate_XY_or_MD(X_bwd, Y_bwd, MD_bwd, Func_bwd, Dim_bwd)
    YE_opt = np.concatenate([YE_fwd, YE_bwd])

    Info = {
        "X_fwd": X_fwd,
        "Y_fwd": Y_fwd,
        "X_bwd": X_bwd,
        "Y_bwd": Y_bwd,
        "MD_fwd": MD_fwd,
        "MD_bwd": MD_bwd,
        "model_type": Func,
        "knee_opt": KneeOpt,
        "out_error": opt_error,
        "fwd_model_type": Func_fwd,
        "bwd_model_type": Func_bwd,
        "fwd_model": P_fwd,
        "bwd_model": P_bwd,
        "raw": MagDist,
        "bin_size": BinSize,
        "step_div_length": StepDivLength,
        "raw_smoothed": np.column_stack((X, Y)),
        "opt": np.column_stack((X, YE_opt)),
    }

    if DEBUG:
        plot_kee_find(Info)

    return KneeOpt, Info, BinListInfo


def plot_kee_find(Info):
    MagDist = Info["raw"]
    KneeOpt = Info["knee_opt"]
    X = Info["raw_smoothed"][:, 0]
    Y = Info["raw_smoothed"][:, 1]
    X_opt = Info["opt"][:, 0]
    Y_opt = Info["opt"][:, 1]
    X_fwd_index = X <= KneeOpt
    X_opt_fwd_index = X_opt <= KneeOpt

    ModelType = Info["model_type"]
    Type_fwd = Info["fwd_model_type"]
    Type_bwd = Info["bwd_model_type"]
    P_fwd = Info["fwd_model"]
    P_bwd = Info["bwd_model"]

    plt.figure()
    Row = 4
    plt.subplot(Row, 1, 1)
    plt.hist(MagDist)
    plt.subplot(Row, 1, 2)
    plt.plot(X, Y)
    plt.plot(X_opt, Y_opt)
    plt.legend(["raw data", "approximation"])
    plt.subplot(Row, 1, 3)
    plt.plot(X[X_fwd_index], Y[X_fwd_index])
    plt.plot(X_opt[X_opt_fwd_index], Y_opt[X_opt_fwd_index])
    plt.legend(["raw data", "approximation"])
    plt.subplot(Row, 1, 4)
    plt.plot(X[~X_fwd_index], Y[~X_fwd_index])
    plt.plot(X_opt[~X_opt_fwd_index], Y_opt[~X_opt_fwd_index])
    plt.legend(["raw data", "approximation"])
    Title = f"{ModelType} knee= {KneeOpt}"
    plt.suptitle(Title.replace("_", "\\_"))
    plt.show()


def extract_fb_function(Func):
    if isinstance(Func, list):
        Func = Func[0]

    C = Func.split("_")
    if len(C) == 4:
        Func_fwd = C[0]
        Dim_fwd = int(C[1])
        Func_bwd = C[2]
        Dim_bwd = int(C[3])
    elif len(C) == 1:
        Func_fwd = C[0]
        Dim_fwd = Dim_bwd = Func_bwd = None
    else:
        print(f"[extract_fb_function] unknown Func({Func})")
        Func_fwd = Dim_fwd = Func_bwd = Dim_bwd = None

    return Func_fwd, Dim_fwd, Func_bwd, Dim_bwd


def apploximate_XY_or_MD(X, Y, MD, Func, Dim):
    if Func == "poly":
        P = np.polyfit(X, Y, Dim)
        YE = np.polyval(P, X)
    elif Func == "normal":
        if len(MD) > 3:
            P = norm.fit(MD)
            distribution = norm(loc=P[0], scale=P[1])
            YE = distribution.pdf(X)
        else:
            P = np.polyfit(X, Y, 0)
            YE = np.polyval(P, X)
    return YE, P


def Spikelet_Stat_knee_find_double(MagDist, FuncList, Weight=None):
    DEBUG = False

    if Weight is None:
        Weight = np.nan

    KeeOpt_1, Info_1 = Spikelet_Stat_knee_find(MagDist, FuncList[0], Weight)
    MagDist_2 = MagDist[MagDist >= KeeOpt_1]
    KeeOpt_2, Info_2 = Spikelet_Stat_knee_find(MagDist_2, FuncList[1], Weight)

    fwd_model_type_2 = Info_2["bwd_model_type"]
    fwd_model_2 = Info_2["fwd_model"]
    KeeOptover_pos = (
        np.where(Info_2["Y_fwd"] >= np.max(Info_2["Y_bwd"]))[0][-1]
        if len(np.where(Info_2["Y_fwd"] >= np.max(Info_2["Y_bwd"]))[0]) > 0
        else len(Info_2["Y_fwd"])
    )

    KeeOpt = KeeOpt_2
    aprx_level = "second"
    if fwd_model_type_2 == "poly" and len(fwd_model_2) == 2:
        if KeeOptover_pos != len(Info_2["Y_fwd"]):
            KeeOpt = Info_2["X_fwd"][KeeOptover_pos]
            aprx_level = "freq-over-bwd"
        elif fwd_model_2[0] >= 0:
            KeeOpt = KeeOpt_1
            aprx_level = "first"

    # Information
    Info = {"knee_opt": KeeOpt, "aprx_level": aprx_level, "detail": [Info_1, Info_2]}

    if DEBUG:
        plot_kee_find(Info)

    return KeeOpt, Info
