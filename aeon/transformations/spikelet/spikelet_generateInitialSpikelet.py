import pdb

import numpy as np
from scipy.interpolate import interp1d

DEBUG_FIND_LR = 0  # Debug flag
LEG_MAXIMAL_CONVEX_AMPLITUDE_2_SUPPORT_ON = False  # Global condition flag
LEG_WINDOW_LENGTH_ON = False

def Spikelet_generateInitialSpikelet(X, RightWindowLength=np.inf):
    """
    Spikelet Decomposition for Time Series.

    Parameters
    ----------
    X : numpy array
        Time series data (1D array).
    RightWindowLength : float, optional
        Forward window length. The default is infinity.

    Returns
    -------
    MagInfo : dict
        Magnitude information from Spikelet decomposition.
    """
    # Function Declaration
    OpName = 'generateInitialSpikelet'
    FuncName = 'Spikelet_generateInitialSpikelet'
    
    # Initialize variables
    n = len(X)  # Length of input time series
    A = np.zeros(n)  # Amplitude
    L = np.zeros(n)  # Support length
    SubseqId = np.arange(n)
    C = np.tile(SubseqId, (5, 1)).T  # Support of convex [sign, left, right, balanced_left, balanced_right]
    C[0, 0] = 1
    #C = np.zeros((n, 5))
    S = Stack_create(n)  # Stack
    m = np.nan  # Previous local maximum time
    s = np.full(4, np.nan)  # Local stacks
    Apl = False
    i = 0
    S = Stack_push(S, 0)
    indices_to_check = [277838, 277839, 284112, 284113, 291511, 291512, 309116, 309117, 309809, 309810, 309946, 309947, 309997, 309998]

    # Main loop
    while not np.isnan(i) and (i <= n - 2 or (i == n - 1 and Apl and S['p'] >= 3)):

        # (1) Get next vertex
        if i <= n - 2 and not Apl:
            # (1.1a) Get next local maximum transformation
            next_vertex, C = getNextLocalMaximum(X, C, i)

            while (S['p'] >= 2) and (S['v'][1] < (next_vertex - RightWindowLength + 1)):
                right_end = S.v[1] + RightWindowLength - 1
                S, A, C, L = forced_left_reduction(X, S, A, C, L, right_end)

            i = next_vertex
            if not np.isnan(next_vertex) and next_vertex <= len(X):
                S = Stack_push(S, next_vertex)  # Push the next local maximum

        # (2) Leg reductions
        Apl1 = False
        Apl2 = False

        if S['p'] >= 4:
            
            # (2.1) Middle leg reduction
            p = S['p']
            s1, s2, s3, s4 = S['v'][p - 4:p]

            if abs(X[s4] - X[s3]) >= abs(X[s3] - X[s2]) and abs(X[s2] - X[s1]) > abs(X[s3] - X[s2]):
                S, A, C, L = middle_leg_reduction(X, S, A, C, L, s1, s2, s3, s4)
                Apl1 = True

                # file_path = r'C:\Users\Victor\Desktop\Uni\Bachelor\stuff\C.npy'
                # # Speichere das Array in die Datei
                # np.save(file_path, C)
                # # Ausgabe zur Bestätigung
                # print(f"Array C gespeichert unter: {file_path}")
        

        if S['p'] >= 3:
            # (2.2) Left leg reduction
            s1, s2, s3 = S['v'][:3]
            if abs(X[s3] - X[s2]) >= abs(X[s2] - X[s1]):
                S, A, C, L = left_leg_reduction(X, S, A, C, L, s1, s2, s3)
                Apl2 = True

        Apl = Apl1 + Apl2

    # file_path = r'C:\Users\Victor\Desktop\Uni\Bachelor\stuff\C.npy'
    # # Speichere das Array in die Datei
    # np.save(file_path, C)
    # # Ausgabe zur Bestätigung
    # print(f"Array C gespeichert unter: {file_path}")

    # (3) Right leg reduction (Reversed terminal leg reduction)
    while S['p'] >= 3:
        p = S['p']
        s1, s2, s3 = S['v'][p - 3:p]

        S, A, C, L = right_leg_reduction(X, S, A, C, L, s1, s2, s3)

    # Time series representation of initial Spikelet decomposition
    initial_mag_time = np.where(A != 0)[0]
    initial_mag_trend = np.zeros(len(A))
    initial_mag_trend[initial_mag_time] = A[initial_mag_time]

    if initial_mag_time.size > 0:
        if initial_mag_time[0] != 0:
            initial_mag_time = np.insert(initial_mag_time, 0, 0)
        if initial_mag_time[-1] != len(X) - 1:
            initial_mag_time = np.append(initial_mag_time, len(X) - 1)
        Data_initial = np.interp(np.arange(len(X)), initial_mag_time, X[initial_mag_time])
    else:
        Data_initial = initial_mag_trend

    # Set Spikelet Decomposition
    MagInfo = {
        'fields': ['center', 'type', 'value', 'left', 'right', 'data', 'magnitude', 'leg_magnitude'],
        'data_org': X,
        'leg_magnitude': np.zeros(len(X)),
        'data': Data_initial,
        'magnitude': A,
        'center': np.where(A != 0)[0],
        'type': np.full(len(X), np.nan),
        'value': np.full_like(Data_initial, np.nan),
        'left': C[:, 1],
        'right': C[:, 2]
    }

    if MagInfo['center'].size > 0:
        MagInfo['type'][MagInfo['center']] = 2 * np.sign(A[MagInfo['center']])
        MagInfo['value'][MagInfo['center']] = X[MagInfo['center']]

    # Save initial spikelet
    MagInfo['output'] = []
    MagInfo = Spikelet_MagInfo_post_processing(MagInfo, OpName)
    
    return MagInfo


def Leg_find_left(X, from_idx, to, to_short):
    FuncName = 'Leg_find_left'
    if DEBUG_FIND_LR == 1:
        print(f'[leg_find_left] ({from_idx}, {to}, {to_short})')

    if LEG_MAXIMAL_CONVEX_AMPLITUDE_2_SUPPORT_ON:
        from_idx, to, to_short = int(from_idx), int(to), int(to_short)
        diff_ft = X[to_short] - X[from_idx]

        if diff_ft > 0:  # Increasing toward left
            if from_idx >= to:
                X_slice = X[to:from_idx + 1]  # Include 'from_idx'
                cand_rel = np.where(X_slice >= X[to_short])[0]
            else:
                cand_rel = np.array([])
        elif diff_ft < 0:  # Decreasing toward left
            if from_idx >= to:
                X_slice = X[to:from_idx + 1]
                cand_rel = np.where(X_slice <= X[to_short])[0]
            else:
                cand_rel = np.array([])
        else:
            print(f'[leg_find_left] diff_ft == 0 (from, to, to_short)=({from_idx}, {to}, {to_short})')
            return to

        if cand_rel.size == 0:
            X_slice = X[to:from_idx + 1]
            if diff_ft > 0:
                cand_rel = np.array([np.argmax(X_slice)])
            elif diff_ft < 0:
                cand_rel = np.array([np.argmin(X_slice)])

        from_to_right_rel = cand_rel.max()
        b_left = to + from_to_right_rel
    else:
        b_left = to

    return b_left


def Leg_find_right(X, from_idx, to, to_short):
    global LEG_MAXIMAL_CONVEX_AMPLITUDE_2_SUPPORT_ON, DEBUG_FIND_LR
    FuncName = 'Leg_find_right'
    if DEBUG_FIND_LR == 1:
        print(f'[leg_find_right] ({from_idx}, {to}, {to_short})')

    if LEG_MAXIMAL_CONVEX_AMPLITUDE_2_SUPPORT_ON:
        from_idx, to, to_short = int(from_idx), int(to), int(to_short)
        diff_ft = X[from_idx] - X[to_short]
        if from_idx + 1 <= to:
            X_slice = X[from_idx + 1 : to + 1]  # Include 'to'
            if diff_ft > 0:  # Decreasing toward right
                cand_rel = np.where(X_slice <= X[to_short])[0]
            elif diff_ft < 0:  # Increasing toward right
                cand_rel = np.where(X_slice >= X[to_short])[0]
            else:
                print(f'[leg_find_right] diff_ft == 0 (from, to, to_short)=({from_idx}, {to}, {to_short})')
                return to

            if cand_rel.size == 0:
                if diff_ft > 0:
                    cand_rel = np.array([np.argmin(X_slice)])
                elif diff_ft < 0:
                    cand_rel = np.array([np.argmax(X_slice)])

            from_to_right_rel = cand_rel.min()
            b_right = from_idx + 1 + from_to_right_rel
        else:
            b_right = to
    else:
        b_right = to

    return b_right


def middle_leg_reduction(X, S, A, C, L, s1, s2, s3, s4, WINDOW_LENGTH=0):
    s1, s2, s3, s4 = int(s1), int(s2), int(s3), int(s4)
    
    # (1) Spike s3
    # (1.1) Finde links und rechts im Konvex von s3
    s3_left = s2
    if LEG_WINDOW_LENGTH_ON:
        s2_limit = s3 - WINDOW_LENGTH + 1
        if s2 < s2_limit:
            s3_left = Leg_find_left(X, s3, s2_limit, s4)
    s3_right = Leg_find_right(X, s3, s4, s2)
    
    # (1.2) Berechne das Konvex von s3
    s3_sign = np.sign(X[s3] - X[s2])
    if s3_sign != np.sign(C[s3, 0]):
        print(f'1: The sign of local maximum ({s3}) is mismatched.')
    C[s3, :] = [s3_sign, s2, s4, s3_left, s3_right]
    if LEG_WINDOW_LENGTH_ON:
        A[s3] = s3_sign * min(abs(X[s3] - X[s3_left]), abs(X[s3] - X[s3_right]))
    else:
        A[s3] = X[s3] - X[s2]
    L[s3] = s3_right - s3_left + 1

        # (2) Spike s2
    # (2.1) Finde links und rechts im Konvex von s2
    s2_right = s3
    s1_restricted = s1
    if LEG_WINDOW_LENGTH_ON:
        s1_limit = s2 - WINDOW_LENGTH + 1
        if s1 < s1_limit:
            s1_restricted = Leg_find_left(X, s2, s1_limit, s3)
    s2_left = Leg_find_left(X, s2, s1_restricted, s3)

    # (2.2) Berechne das Konvex von s2
    s2_sign = np.sign(X[s2] - X[s1])
    if s2_sign != -s3_sign:
        print(f'2: The sign of local maximum ({s2}) is mismatched to that of {s3}.')
    C[s2, :] = [s2_sign, s1, s3, s2_left, s2_right]
    if LEG_WINDOW_LENGTH_ON:
        A[s2] = s2_sign * min(abs(X[s2] - X[s2_left]), abs(X[s2] - X[s2_right]))
    else:
        A[s2] = X[s2] - X[s3]
    L[s2] = s2_right - s2_left + 1


    # Pop stack for s2 and s3
    S, val = Stack_pop_2(S)
    S, val = Stack_pop_2(S)
    return S, A, C, L

def left_leg_reduction(X, S, A, C, L, s1, s2, s3, WINDOW_LENGTH=0):
    s1, s2, s3 = int(s1), int(s2), int(s3)
    # s1, s2, s3 sind 0-basierte Indizes

    # (1) Find left and right in the convex
    s2_left = s1
    s2_right = s3
    if LEG_WINDOW_LENGTH_ON:
        s1_limit = s2 - WINDOW_LENGTH + 1
        if s1 < s1_limit:
            s1_restricted = Leg_find_left(X, s2, s1_limit, s3)
            s2_left = Leg_find_left(X, s2, s1_restricted, s3)

    # (2) Calculate convex
    s2_sign = np.sign(X[s2] - X[s1])
    if s2_sign != np.sign(C[s2, 0]):
        print(f'3:The sign of local maximum ({s2}) is mismatched.')
    C[s2, :] = [s2_sign, s1, s3, s2_left, s2_right]
    if LEG_WINDOW_LENGTH_ON:
        A[s2] = s2_sign * min(abs(X[s2] - X[s2_left]), abs(X[s2] - X[s2_right]))
    else:
        A[s2] = X[s2] - X[s1]
    L[s2] = s2_right - s2_left + 1

    # (3) Pop stack
    S, val = Stack_pop_bottom(S)
    return S, A, C, L



def right_leg_reduction(X, S, A, C, L, s1, s2, s3, WINDOW_LENGTH=0):
    s1, s2, s3 = int(s1), int(s2), int(s3)

    # (1) Find left and right in the convex
    s2_left = s1
    s2_right = s3
    if LEG_WINDOW_LENGTH_ON:
        s1_limit = s1 - WINDOW_LENGTH + 1
        if s1 < s1_limit:
            s1_restricted = Leg_find_left(X, s2, s1_limit, s3)
            s2_left = Leg_find_left(X, s2, s1_restricted, s3)

    # (2) Calculate convex
    s2_sign = np.sign(X[s2] - X[s1])  # Anpassung der Indizes

    if s2_sign != np.sign(C[s2, 0]):
        print(f'4:The sign of local maximum ({s2}) is mismatched.')

    if LEG_WINDOW_LENGTH_ON:
        A[s2] = s2_sign * min(abs(X[s2] - X[s2_left]),abs(X[s2] - X[s2_right]))
    else:
        A[s2] = X[s2] - X[s3]


    C[s2, :] = [s2_sign, s1, s3, s2_left, s2_right]
    L[s2] = s2_right - s2_left + 1

    # Pop
    S, val = Stack_pop_1(S)
    return S, A, C, L


# def getNextLocalMaximum(X, C, present):
#     # Find the next local minimum/maximum
#     if present >= len(X) - 1:
#         return np.nan, C

#     i = present + 1
#     present_val = X[present]
#     present_sign = 0 if present == 0 else np.sign(C[present, 0])

#     while X[i] == present_val:
#         C[i, 0] = 2 * present_sign
#         if i == len(X) - 1:
#             return np.nan, C
#         i += 1

#     next_diff = i
#     if next_diff >= len(X) - 1:
#         C[i, 0] = -3 * C[present, 0]
#         return len(X) - 1, C

#     current_slope = X[next_diff] - present_val
#     while i < len(X) - 1 and current_slope * (X[i + 1] - X[i]) >= 0:
#         C[i, 0] = 0
#         if i == len(X) - 1:
#             C[i, 0] = -3 * present_sign
#             return len(X) - 1, C
#         i += 1

#     if i >= len(X) - 1:
#         C[i, 0] = 0
#         return len(X) - 1, C

#     next_vertex = i
#     C[i, 0] = np.sign(X[i] - X[i + 1]) if i + 1 < len(X) else np.sign(X[i])
#     if present == 0:
#         C[0, 0] = -3 * C[i, 0]
#         C[1:i, 0] = 2 * np.sign(C[0, 0])

#     return next_vertex, C

# def getNextLocalMaximum(X, C, present):
#     n = len(X)
#     if present >= n:
#         return np.nan, C

#     # Simulieren der 1-basierten Indexierung
#     present_1b = present + 1

#     i = present_1b + 1

#     if present == 0:
#         present_val = X[i - 1]
#         present_sign = 0
#     else:
#         present_val = X[present_1b - 1]
#         present_sign = np.sign(C[present_1b - 1, 0])

#     while i - 1 < n and X[i - 1] == present_val:
#         C[i - 1, 0] = 2 * present_sign
#         if i == n:
#             return np.nan, C
#         else:
#             i += 1

#     next_diff = i
#     if next_diff >= n:
#         C[i - 1, 0] = -3 * C[present_1b - 1, 0]
#         return n - 1, C  # Zurück zur 0-basierten Indexierung

#     current_slope = X[next_diff - 1] - present_val
#     while i < n and current_slope * (X[i] - X[i - 1]) >= 0:
#         C[i - 1, 0] = 0
#         if i == n:
#             C[i - 1, 0] = -3 * present_sign
#             return n - 1, C
#         else:
#             i += 1
#             if i >= n:
#                 C[i - 1, 0] = -3 * present_sign
#                 return n - 1, C

#     next_vertex = i
#     if i < n:
#         C[i - 1, 0] = np.sign(X[i - 1] - X[i])
#     else:
#         C[i - 1, 0] = np.sign(X[i - 1])

#     if present == 0:
#         C[0, 0] = -3 * C[i - 1, 0]
#         if i - 1 > 1:
#             C[1:i - 1, 0] = 2 * np.sign(C[0, 0])

#     return next_vertex - 1, C  # Zurück zur 0-basierten Indexierung

def getNextLocalMaximum(X, C, present):
    n = len(X)
    if present >= n - 1:
        return np.nan, C
    
    i = present + 1

    if present == -1:
        # Am Anfang der Zeitreihe
        i = 0
        present_val = X[0]
        present_sign = 0  # Unbekannt am Start
    else:
        present_val = X[present]
        present_sign = np.sign(C[present, 0])

    # (1) Finden des nächsten Index, an dem sich der Wert ändert
    while i < n and X[i] == present_val:
        C[i, 0] = 2 * present_sign
        i += 1
        if i == n:
            return np.nan, C

    next_diff = i  # Nächster Index mit unterschiedlichem Wert

    if next_diff >= n - 1:
        # Ende der Zeitreihe erreicht
        C[i, 0] = -3 * present_sign
        return n - 1, C

    # (3) Finden eines lokalen Maximums oder Minimums
    current_slope = X[next_diff] - present_val

    while i < n - 1 and current_slope * (X[i + 1] - X[i]) >= 0:
        C[i, 0] = 0
        i += 1
        if i == n - 1:
            C[i, 0] = -3 * present_sign
            return n - 1, C

    next_vertex = i  # Nächstes lokales Maximum oder Minimum
    if i < n - 1:
        C[i, 0] = np.sign(X[i] - X[i + 1])
    else:
        C[i, 0] = np.sign(X[i])

    if present == -1:
        C[0, 0] = -3 * C[i, 0]
        if i > 1:
            C[1:i, 0] = 2 * np.sign(C[0, 0])

    return next_vertex, C



def forced_left_reduction(X, S, A, C, L, i, LEG_WINDOW_LENGTH_ON, WINDOW_LENGTH):
    FuncName = 'forced_left_reduction'
    # Für Debugging-Zwecke
    # if s2 == 8:
    #     print(f'[{FuncName}] s2 = {s2}')
    
    if not LEG_WINDOW_LENGTH_ON:
        print(f'[{FuncName}] Unexpected LEG_WINDOW_LENGTH_ON = False. i = {i}')
    
    # (1) Lokaler Stack
    if S.p >= 2:
        s1 = S.v[0]  # MATLAB-Index S.v(1) entspricht S.v[0] in Python
        s2 = S.v[1]  # S.v(2) entspricht S.v[1]

        # (2) Finde links und rechts des Convex
        if S.p >= 3:
            s3 = S.v[2]  # S.v(3) entspricht S.v[2]
        else:
            s3 = Leg_find_right(X, s2, i, s1)
        
        s2_right = s3
        s2_left = s1
        if LEG_WINDOW_LENGTH_ON:
            s1_restricted = s2 - WINDOW_LENGTH + 1
            if s1 < s1_restricted:
                s2_left = Leg_find_left(X, s2, s1_restricted, s3)
        
        # (3) Berechne Convex
        c_sign = np.sign(X[s2] - X[s3])
        if c_sign != np.sign(C[s2, 0]):
            print(f'3: The sign of local maximum ({s2}) is mismatched.')
        
        A[s2] = c_sign * min(abs(X[s2] - X[s2_left]), abs(X[s2] - X[s2_right]))
        C[s2, :] = [c_sign, s1, s3, s2_left, s2_right]
        L[s2] = s2_right - s2_left + 1

        # (3) Pop Stack: Entferne den linken Wert s(1) vom lokalen Stack s(1:3)
        S, val = Stack_pop_bottom(S)
    else:
        S = Stack_replace_bottom(S, S.v[0] + 1)
    
    return S, A, C, L


# Stack management functions
def Stack_create(size):
    return {'v': np.full(size, np.nan), 'p': 0, 'size': size}

def Stack_push(S, val):
    if S['p'] >= S['size']:
        S['p'] = np.inf
        print('Push failed because of Stack overflow.')
        return S
    S['v'][S['p']] = val
    S['p'] += 1
    return S

def Stack_pop_1(S):
    if S['p'] <= 0:
        print('Pop failed because of Empty Stack.')
    val = S['v'][S['p'] - 1]
    S['p'] -= 1
    return S, val

def Stack_pop_2(S):
    if S['p'] <= 0:
        print('Pop failed because of Empty Stack.')
    val = S['v'][S['p'] - 2]
    S['v'][S['p'] - 2] = S['v'][S['p'] - 1]
    S['p'] -= 1
    return S, val

def Stack_pop_bottom(S):
    if S['p'] <= 0:
        print('Pop failed because of Empty Stack.')
    val = S['v'][0]
    S['v'][:S['p'] - 1] = S['v'][1:S['p']]
    S['p'] -= 1
    return S, val

def Stack_replace_bottom(S, s_val):
    if S['p'] <= 0:
        print('Pop failed because of Empty Stack.')
    val = S['v'][0]
    S['v'][0] = s_val
    return S, val

def Spikelet_MagInfo_post_processing(MagInfo, OpName):
    # Ensure 'output' is a dictionary
    if not isinstance(MagInfo.get('output'), dict):
        MagInfo['output'] = {}

    # latest dependent fields 
    MagInfo['center'] = np.where(~np.isnan(MagInfo['type']))[0]
    MagInfo['data'] = Spikelet_get_TimeSeriesForm_from_SpikeletDecomposition(MagInfo)
    
    # output fields (independent)
    MagInfo['output'][OpName] = {}
    MagInfo['output'][OpName]['type'] = MagInfo['type']
    MagInfo['output'][OpName]['value'] = MagInfo['value']
    MagInfo['output'][OpName]['magnitude'] = MagInfo['magnitude']
    MagInfo['output'][OpName]['leg_magnitude'] = MagInfo['leg_magnitude']
    MagInfo['output'][OpName]['left'] = MagInfo['left']
    MagInfo['output'][OpName]['right'] = MagInfo['right']
    
    # dependent
    MagInfo['output'][OpName]['data'] = MagInfo['data']
    MagInfo['output'][OpName]['center'] = MagInfo['center']
    
    return MagInfo


def Spikelet_get_TimeSeriesForm_from_SpikeletDecomposition(MagInfo):
    Data_org = MagInfo['data_org']
    Center = np.where(~np.isnan(MagInfo['type']))[0]
    Left = MagInfo['left'][Center] - 1  # Adjust for 0-based indexing
    Right = MagInfo['right'][Center] - 1  # Adjust for 0-based indexing

    # Adjust Apx_time for 0-based indexing and remove any -1 values
    Apx_time = np.unique(np.concatenate(([0], Left, Center, Right, [len(Data_org) - 1])))
    Apx_time = Apx_time[Apx_time >= 0]  # Ensure no negative indices

    if len(Apx_time) >= 2:
        interpolation_func = interp1d(Apx_time, Data_org.iloc[Apx_time], kind='linear', fill_value="extrapolate")
        Data = interpolation_func(np.arange(len(Data_org)))
    else:
        Data = Data_org.copy()

    Center_const = np.where(MagInfo['type'] == 0)[0]
    for i in range(len(Center_const)):
        from_idx = MagInfo['left'][Center_const[i]] - 1  # Adjust for 0-based indexing
        to_idx = MagInfo['right'][Center_const[i]] - 1  # Adjust for 0-based indexing
        value = MagInfo['value'][Center_const[i]]
        Data[from_idx:to_idx + 1] = value * np.ones(to_idx - from_idx + 1)

    return Data



