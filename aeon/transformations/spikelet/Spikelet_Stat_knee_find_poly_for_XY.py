import numpy as np


def Spikelet_Stat_knee_find_poly_for_XY(X, Y, Dim_fwd, Dim_bwd):
    ErrFunc = "l1"

    X_sorted = np.sort(X)
    Knee_list = X_sorted[Dim_fwd:-Dim_bwd]
    Err = np.inf * np.ones(len(Knee_list))

    for i in range(len(Knee_list)):
        breakpt = Knee_list[i]
        Fwd_index = X <= breakpt
        X_fwd = X[Fwd_index]
        Y_fwd = Y[Fwd_index]
        X_bwd = X[~Fwd_index]
        Y_bwd = Y[~Fwd_index]

        YE_fwd, P_fwd = apploximate_poly_XY(X_fwd, Y_fwd, Dim_fwd)
        YE_bwd, P_bwd = apploximate_poly_XY(X_bwd, Y_bwd, Dim_bwd)

        YE = np.concatenate([YE_fwd, YE_bwd])
        if ErrFunc == "l1":
            Err[i] = np.sum(np.abs(YE - Y))
        elif ErrFunc == "l2":
            Err[i] = np.sum((YE - Y) ** 2)

    # Find optimal value
    opt_pos = np.argmin(Err)
    KneeOpt = Knee_list[opt_pos]

    # Optimal approximation
    Fwd_index = X <= KneeOpt
    X_fwd = X[Fwd_index]
    Y_fwd = Y[Fwd_index]
    X_bwd = X[~Fwd_index]
    Y_bwd = Y[~Fwd_index]

    YE_fwd, P_fwd = apploximate_poly_XY(X_fwd, Y_fwd, Dim_fwd)
    YE_bwd, P_bwd = apploximate_poly_XY(X_bwd, Y_bwd, Dim_bwd)
    YE_opt = np.concatenate([YE_fwd, YE_bwd])

    # Output
    Info = {
        "knee_opt": KneeOpt,
        "out_error": Err[opt_pos],
        "fwd_model_dim": Dim_fwd,
        "bwd_model_dim": Dim_bwd,
        "fwd_model": P_fwd,
        "bwd_model": P_bwd,
        "raw_smoothed": np.column_stack((X, Y)),
        "opt": np.column_stack((X, YE_opt)),
    }

    return KneeOpt, Info


def apploximate_poly_XY(X, Y, Dim):
    P = np.polyfit(X, Y, Dim)
    YE = np.polyval(P, X)
    return YE, P
