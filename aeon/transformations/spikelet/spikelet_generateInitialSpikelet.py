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
    S = Stack_create(n)  # Stack
    m = np.nan  # Previous local maximum time
    s = np.full(4, np.nan)  # Local stacks
    Apl = False
    i = 0
    S = Stack_push(S, 0)

    # Main loop
    while not np.isnan(i) and (i <= n - 2 or (i == n - 1 and Apl and S['p'] >= 3)):
        # (1) Get next vertex
        if i <= n - 2 and not Apl:
            # (1.1a) Get next local maximum transformation
            next_vertex, C = getNextLocalMaximum(X, C, i)

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
        

        if S['p'] >= 3:
            # (2.2) Left leg reduction
            s1, s2, s3 = S['v'][:3]
            if abs(X[s3] - X[s2]) >= abs(X[s2] - X[s1]):
                S, A, C, L = left_leg_reduction(X, S, A, C, L, s1, s2, s3)
                Apl2 = True

        Apl = Apl1 or Apl2

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


# Helper functions
def Leg_find_left(X, from_idx, to, to_short):
    """
    Find the first index where the value exceeds a threshold from 'from' to 'to'.

    Parameters:
    X (numpy.ndarray or pandas.Series): The time series data.
    from_idx (int): The starting index.
    to (int): The ending index.
    to_short (int): The index where threshold comparison occurs.

    Returns:
    int: The index where the condition is first met.
    """
    if (from_idx == 1):
        print(f"leg find left from_idx: {from_idx}, to: {to}, to_short: {to_short}")

    if DEBUG_FIND_LR == 1:
        print(f'[leg_find_left] (from: {from_idx}, to: {to}, to_short: {to_short})')

    if LEG_MAXIMAL_CONVEX_AMPLITUDE_2_SUPPORT_ON:
        # Ensure indices are integers
        from_idx, to, to_short = int(from_idx), int(to), int(to_short)

        diff_ft = X[to_short] - X[from_idx]

        # Determine the indices based on the direction of the change
        if diff_ft > 0:  # Increasing toward left
            cand_rel = np.where(X[to:from_idx - 1] >= X[to_short])[0]
        elif diff_ft < 0:  # Decreasing toward left
            cand_rel = np.where(X[to:from_idx - 1] <= X[to_short])[0]
        else:
            print(f'[leg_find_left] diff_ft == 0 (from: {from_idx}, to: {to}, to_short: {to_short})')
            return to

        # If no crossing is found, take the extreme point within the range
        if cand_rel.size == 0:
            if diff_ft > 0:
                if to < from_idx - 1:
                    cand_rel = np.argmax(X[to:from_idx - 1])  # Index of maximum value
                else:
                    return to
            elif diff_ft < 0:
                if to < from_idx - 1:
                    cand_rel = np.argmin(X[to:from_idx - 1])  # Index of minimum value
                else:
                    return to

        # Determine the left boundary
        if isinstance(cand_rel, np.ndarray):
            from_to_right_rel = int(cand_rel[0]) if cand_rel.size > 0 else 0
        else:
            from_to_right_rel = int(cand_rel) if cand_rel.size > 0 else 0
        b_left = to + from_to_right_rel - 1

    else:
        b_left = to

    return b_left


def Leg_find_right(X, from_idx, to, to_short):
    """
    Find the first index where the value exceeds a threshold from 'from' to 'to'.

    Parameters:
    X (numpy.ndarray or pandas.Series): The time series data.
    from_idx (int): The starting index.
    to (int): The ending index.
    to_short (int): The index where threshold comparison occurs.

    Returns:
    int: The index where the condition is first met.
    """
    if (from_idx == 1):
        print(f"leg find right from_idx: {from_idx}, to: {to}, to_short: {to_short}")

    if DEBUG_FIND_LR == 1:
        print(f'[leg_find_right] (from: {from_idx}, to: {to}, to_short: {to_short})')

    if LEG_MAXIMAL_CONVEX_AMPLITUDE_2_SUPPORT_ON:
        # Ensure indices are integers
        from_idx, to, to_short = int(from_idx), int(to), int(to_short)

        diff_ft = X[from_idx] - X[to_short]
        
        # Determine the indices based on the direction of the change
        if diff_ft > 0:  # Decreasing toward right
            cand_rel = np.where(X[from_idx + 1:to] <= X[to_short])[0]
        elif diff_ft < 0:  # Increasing toward right
            cand_rel = np.where(X[from_idx + 1:to] >= X[to_short])[0]
        else:
            print(f'[leg_find_right] diff_ft == 0 (from: {from_idx}, to: {to}, to_short: {to_short})')
            return to

        # If no crossing is found, take the extreme point within the range
        if cand_rel.size == 0:
            if diff_ft > 0:
                if from_idx + 1 < to:
                    cand_rel = np.argmin(X[from_idx + 1:to])  # Index of minimum value
                else:
                    return to
            elif diff_ft < 0:
                if from_idx + 1 < to:
                    cand_rel = np.argmax(X[from_idx + 1:to])  # Index of maximum value
                else:
                    return to
        
        # Determine the right boundary
        if isinstance(cand_rel, np.ndarray):
            from_to_right_rel = int(cand_rel[0]) if cand_rel.size > 0 else 0
        else:
            from_to_right_rel = int(cand_rel) if cand_rel.size > 0 else 0
        b_right = from_idx + 1 + from_to_right_rel

    else:
        b_right = to

    return b_right

def middle_leg_reduction(X, S, A, C, L, s1, s2, s3, s4):
    s1, s2, s3, s4 = int(s1), int(s2), int(s3), int(s4)
    
    # Reduction of middle legs
    s3_left = s2
    s3_right = Leg_find_right(X, s3, s4, s2)

    s3_sign = np.sign(X[s3] - X[s2])
    C[s3, :] = [s3_sign, s2, s4, s3_left, s3_right]
    if LEG_WINDOW_LENGTH_ON:
        A[s3] = s3_sign * min(abs(X[s3] - X[s3_left]), abs(X[s3] - X[s3_right]))
    else:
        A[s3] = X[s3] - X[s2]

    L[s3] = s3_right - s3_left + 1

    # Spike s2
    
    s2_right = s3
    s2_left = Leg_find_left(X, s2, s1, s3)

    s2_sign = np.sign(X[s2] - X[s1])
    C[s2, :] = [s2_sign, s1, s3, s2_left, s2_right]
    A[s2] = s2_sign * np.minimum(np.abs(X[s2] - X[s2_left]), np.abs(X[s2] - X[s2_right]))
    L[s2] = s2_right - s2_left + 1

    # Pop stack for s2 and s3
    S, val = Stack_pop_2(S)
    S, val = Stack_pop_2(S)
    return S, A, C, L

def left_leg_reduction(X, S, A, C, L, s1, s2, s3):
    s1, s2, s3 = int(s1), int(s2), int(s3)
    # Reduction of left legs
    s2_left = s1
    s2_right = s3
    s2_sign = np.sign(X[s2] - X[s1])
    C[s2, :] = [s2_sign, s1, s3, s2_left, s2_right]
    A[s2] = s2_sign * min(abs(X[s2] - X[s2_left]), abs(X[s2] - X[s2_right]))
    L[s2] = s2_right - s2_left + 1

    # Pop stack
    S, val = Stack_pop_bottom(S)
    return S, A, C, L

def right_leg_reduction(X, S, A, C, L, s1, s2, s3):
    s1, s2, s3 = int(s1), int(s2), int(s3)
    # Reduction of right legs
    s2_left = s1
    s2_right = s3
    s2_sign = np.sign(X[s2] - X[s1])
    C[s2, :] = [s2_sign, s1, s3, s2_left, s2_right]
    A[s2] = s2_sign * min(abs(X[s2] - X[s2_left]), abs(X[s2] - X[s2_right]))
    L[s2] = s2_right - s2_left + 1

    # Pop
    S, val = Stack_pop_1(S)
    return S, A, C, L

def getNextLocalMaximum(X, C, present):
    # Find the next local minimum/maximum
    if present >= len(X) - 1:
        return np.nan, C

    i = present + 1
    present_val = X[present]
    present_sign = 0 if present == 0 else np.sign(C[present, 0])

    while X[i] == present_val:
        C[i, 0] = 2 * present_sign
        if i == len(X) - 1:
            return np.nan, C
        i += 1

    next_diff = i
    if next_diff >= len(X) - 1:
        C[i, 0] = -3 * C[present, 0]
        return len(X) - 1, C

    current_slope = X[next_diff] - present_val
    while i < len(X) - 1 and current_slope * (X[i + 1] - X[i]) >= 0:
        C[i, 0] = 0
        if i == len(X) - 1:
            C[i, 0] = -3 * present_sign
            return len(X) - 1, C
        i += 1

    if i >= len(X) - 1:
        C[i, 0] = 0
        return len(X) - 1, C

    next_vertex = i
    C[i, 0] = np.sign(X[i] - X[i + 1]) if i + 1 < len(X) else np.sign(X[i])
    if present == 0:
        C[0, 0] = -3 * C[i, 0]
        C[1:i, 0] = 2 * np.sign(C[0, 0])

    return next_vertex, C



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



