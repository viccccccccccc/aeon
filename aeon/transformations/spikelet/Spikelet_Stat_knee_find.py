import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.io import savemat
import scipy.io
import os
import json
import pdb

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

    if len(Knee_list) == 0:
        Knee_list = np.array([])

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

    # opt_error = np.min(Err)
    # opt_pos = np.argmin(Err)
    # KneeOpt = Knee_list[opt_pos]

    # Überprüfen, ob Err leer ist
    if Err.size == 0:
        opt_error = np.array([])  # Standardwert für Fehler, wenn Err leer ist
        opt_pos = np.array([])  # Standardposition (erste Position, falls erforderlich)
        KneeOpt = np.array([])  # Fallback: erster Wert in X_sorted
    else:
        # Normale Berechnung, wenn Err nicht leer ist
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
    # Fwd_index = X <= KneeOpt
    # MD_index = MagDist <= KneeOpt
    if KneeOpt.size == 0:
        # Leere Bool-Maske wie 1x0 logical in MATLAB
        Fwd_index = np.array([False]) #np.zeros(0)
        MD_index = np.array([False]) #np.zeros(0)
        X_fwd, Y_fwd = np.array([]), np.array([])
        X_bwd, Y_bwd = np.array([]), np.array([])
        MD_fwd, MD_bwd = np.array([]), np.array([])
        YE_fwd, P_fwd = apploximate_XY_or_MD(X_fwd, Y_fwd, MD_fwd, Func_fwd, Dim_fwd)
        YE_bwd, P_bwd = apploximate_XY_or_MD(X_bwd, Y_bwd, MD_bwd, Func_bwd, Dim_bwd)
        YE_opt = np.concatenate([YE_fwd, YE_bwd])
    else:
        Fwd_index = (X <= KneeOpt)
        MD_index = MagDist <= KneeOpt
        X_fwd, Y_fwd = X[Fwd_index], Y[Fwd_index]
        X_bwd, Y_bwd = X[~Fwd_index], Y[~Fwd_index]
        MD_fwd, MD_bwd = MagDist[MD_index], MagDist[~MD_index]
        YE_fwd, P_fwd = apploximate_XY_or_MD(X_fwd, Y_fwd, MD_fwd, Func_fwd, Dim_fwd)
        YE_bwd, P_bwd = apploximate_XY_or_MD(X_bwd, Y_bwd, MD_bwd, Func_bwd, Dim_bwd)
        YE_opt = np.concatenate([YE_fwd, YE_bwd])


    if YE_opt is None or len(YE_opt) == 0:
    # Falls YE_opt leer ist, nur den ersten Wert von X nehmen und bspw. 0 oder np.nan als Y
        opt_array = np.column_stack((X[:1], [0.0]))  
    else:
        # Normale Fall: X und YE_opt haben gleiche Länge
        opt_array = np.column_stack((X, YE_opt))

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
        "opt": opt_array,
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
        if X.size < 2 or Y.size < 2:
            # Standardpolynom mit Null-Werten, falls nicht genug Punkte vorhanden sind
            P = np.zeros(Dim + 1)  # Polynom-Koeffizienten alle auf 0 setzen
            YE = np.zeros_like(X)  # Approximierte Werte ebenfalls 0
        else:
            P = np.polyfit(X, Y, Dim)  # Polynom berechnen
            YE = np.polyval(P, X)      # Werte des Polynoms berechnen
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

import numpy as np
import scipy.linalg as la
import warnings

def polyfit(x, y, n):
    """
    Python-Pendant zu MATLAB polyfit(x, y, n).

    Gibt zurück:
      p  : array (Länge n+1), Polynomkoeffizienten in absteigender Potenzreihenfolge
      S  : dict mit Schlüsseln 'R', 'df', 'normr', 'rsquared'
      mu : array(2), [mean_x, std_x], sofern Zentrierung/Skalierung erfolgt
    """

    # ------------------------------------------------------------
    # 1) MATLAB: Größencheck, x und y müssen gleich viele Elemente haben
    # ------------------------------------------------------------
    if len(x) != len(y):
        raise ValueError("MATLAB:polyfit:XYSizeMismatch")

    # In numpy-Arrays wandeln; float-Konvertierung wie in MATLAB (double).
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Ermitteln wir die "Ausgangs-Klasse" (in MATLAB: superiorfloat), hier
    # lassen wir es i.d.R. bei float64. Für Single-Präzision müsste man extra
    # Logik einbauen.
    # outputClass = float   # (Placebo im Vergleich zu MATLAB.)

    # ------------------------------------------------------------
    # 2) Wenn in MATLAB nargout > 2 => mu = [mean(x); std(x)] und
    #    interne Zentrierung und Skalierung von x.
    #
    #    Wir machen es hier IMMER, damit p,S,mu kompatibel sind.
    #    (Du kannst gerne eine Option 'return_mu' einführen, falls gewünscht.)
    # ------------------------------------------------------------
    mx = np.mean(x)
    # MATLABs std() default ist die Stichprobenvarianz (N-1) -> ddof=1
    sx = np.std(x, ddof=1)
    mu = np.array([mx, sx])

    if sx != 0.0:
        x_scaled = (x - mx)/sx
    else:
        # Falls alle x identisch sind: std=0 => verhüte Division durch 0
        x_scaled = x - mx  # alles Null
        # MATLAB warnt bei "X might need centering and scaling", wir können
        # analog warnen, wenn du 1:1-Verhalten willst.
        # Für Einfachheit hier: tun wir so.
        warnings.warn("MATLAB:polyfit:RepeatedPointsOrRescale", UserWarning)

    # ------------------------------------------------------------
    # 3) Vandermonde-Matrix V konstruieren (wie in MATLAB: x^n ... x^1, x^0)
    #
    # MATLAB tut:
    #   V(:, n+1) = ones(...)
    #   for j = n:-1:1
    #       V(:, j) = x .* V(:, j+1);
    #   end
    #
    # In NumPy gibt es np.vander(x, N+1) => Spalte0 = x^N, ... SpalteN = x^0
    # Das passt perfekt.
    # ------------------------------------------------------------
    V = np.vander(x_scaled, n+1)  # shape = (len(x), n+1)

    # ------------------------------------------------------------
    # 4) Least-Squares-Lösung p = V \ y
    #
    # In MATLAB:
    #   [p, rankV, QRfactor, perm] = matlab.internal.math.leastSquaresFit(V,y1);
    #
    # Dort wird ein QR mit Spalten-Pivoting gemacht. Um das zu matchen:
    #   Q, R, Piv = qr(V, pivoting=True) in SciPy
    #
    # Danach kann man p lösen (ggf. mit R, Q, Piv).
    # ------------------------------------------------------------
    Q, R, pvt = la.qr(V, mode='economic', pivoting=True)

    # rank:
    # Toleranz wie in MATLAB ~ max(size(V))*eps(norm(R,1)) etc.
    # Hier nehmen wir dasselbe, was la.qr() nutzt oder la.lstsq():
    tol = np.finfo(V.dtype).eps * np.abs(R).max() * max(V.shape)
    # Bestimmen wir den (numerischen) Rang:
    diagR = np.abs(np.diag(R))
    rankV = np.sum(diagR > tol)

    # y in passender Form:
    y1 = y.copy()

    # 4a) Löse Q*R*P^T * p = y => R * (P^T p) = Q^T y
    # => p_hat = P * R^{-1} Q^T y
    # Zuerst c = Q^T y
    c = np.dot(Q.T, y1)
    # Dann R^-1 c, aber nur bis rankV
    # Falls rankV < n+1 => wir müssen truncated solve machen
    c[:rankV] = la.solve_triangular(R[:rankV, :rankV], c[:rankV], lower=False)
    c[rankV:] = 0.0
    # Dann p_hat = zeros, aber in der permutierten Reihenfolge füllen
    ptemp = np.zeros(n+1, dtype=float)
    for i, colindex in enumerate(pvt):
        ptemp[colindex] = c[i]

    p = ptemp  # -> p in descending powers

    # ------------------------------------------------------------
    # 5) Warnungen ausgeben
    # ------------------------------------------------------------
    # if size(V,1) < size(V,2)
    if V.shape[0] < V.shape[1]:
        warnings.warn("MATLAB:polyfit:PolyNotUnique", UserWarning)

    # if rankV < size(V,2)
    if rankV < V.shape[1]:
        # in MATLAB: wenn nargout>2 => 'RepeatedPoints', sonst 'RepeatedPointsOrRescale'
        # Da wir hier immer 'mu' zurückgeben, verhalten wir uns wie nargout>2:
        warnings.warn("MATLAB:polyfit:RepeatedPoints", UserWarning)

    # ------------------------------------------------------------
    # 6) S-Struktur aufbauen
    #
    # S.R     = R (im entpivotierten Format)
    # S.df    = max(0, length(y) - (n+1))
    # S.normr = norm(r)
    # S.rsquared = ...
    #
    # r = y - V*p
    # ------------------------------------------------------------
    # Residuen
    r = y - V.dot(p)
    normr = np.linalg.norm(r)

    # R entpivotieren => Rvoll = R[:,pvt^-1]
    # pvt ist z.B. [2,0,1] => invertiere
    invpvt = np.zeros_like(pvt)
    for idx, col in enumerate(pvt):
        invpvt[col] = idx
    # => Rsort = R[:, invpvt], allerdings nur die ersten minmn Zeilen
    #   minmn = min(size(V)) = rank(V) max, aber wir extrahieren:
    minmn = min(*V.shape)
    R_extract = R[:minmn, :]     # oberer Teil
    # In MATLAB: R(:,perminv)
    R_entpivot = R_extract[:, invpvt]

    df = max(0, len(y) - (n+1))

    # R^2
    # S.rsquared = 1 - (norm(r)/norm(y - mean(y)))^2
    ym = np.mean(y)
    norm_ymean = np.linalg.norm(y - ym)
    if norm_ymean == 0.0:
        # Verhindert 0-Division, z.B. wenn alle y identisch
        rsq = 1.0 if normr == 0 else 0.0
    else:
        rsq = 1.0 - (normr / norm_ymean)**2

    S = {
        'R':      R_entpivot,
        'df':     df,
        'normr':  normr,
        'rsquared': rsq
    }

    # ------------------------------------------------------------
    # 7) MATLAB: p wird als ZEILENvektor zurückgegeben => p'
    #
    # In NumPy ist ein 1D-array sowieso (n+1, ), wir können aber
    # anmerken, dass es "row vector" sein soll. Meist reicht 1D in Python.
    # ------------------------------------------------------------
    # => In MATLAB: p hat shape (1, n+1). In Python normalerweise (n+1, ).
    # Wir belassen es bei 1D, was das Übliche in Python ist.

    return p, S, mu
