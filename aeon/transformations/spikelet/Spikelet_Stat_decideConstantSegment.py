import numpy as np
import matplotlib.pyplot as plt

def Spikelet_Stat_decideConstantSegment(data_org, segment_initial, segment_names, param_ef, mag_thr, cl_thr, mag_info):
    """
    Identifies and processes constant segments in a signal based on thresholds.

    Parameters:
        data_org (np.ndarray): Original signal data.
        segment_initial (np.ndarray): Initial segment information.
        segment_names (list): Column names for segment_initial.
        param_ef (dict): Parameter dictionary.
        mag_thr (float): Magnitude threshold.
        cl_thr (float): Length threshold.
        mag_info (dict): Magnitude information.

    Returns:
        segment (np.ndarray): Processed constant segments.
        segment_names (list): Names of the segment columns.
        segment_id_hist (dict): History of segment IDs at each step.
    """
    TERMINAL_MINIMUM_LENGTH = 3

    # Output history dictionary
    segment_id_hist = {"Segment_initial": segment_initial}

    # Extract lengths and check for thresholds
    cld = segment_initial[:, segment_names.index("length")]
    index_cl_thr = cld >= cl_thr
    if index_cl_thr.sum() <= 1:
        return [], segment_names, segment_id_hist

    # Ensure terminal constants are preserved
    if not index_cl_thr[0]:
        if segment_initial[0, segment_names.index("from")] == 1 and segment_initial[0, segment_names.index("length")] >= TERMINAL_MINIMUM_LENGTH:
            index_cl_thr[0] = True

    if not index_cl_thr[-1]:
        if segment_initial[-1, segment_names.index("to")] == len(data_org) and segment_initial[-1, segment_names.index("length")] >= TERMINAL_MINIMUM_LENGTH:
            index_cl_thr[-1] = True

    id_cl_thr_cut = np.where(index_cl_thr)[0]
    segment_id_hist["Id_CLThrCut"] = id_cl_thr_cut
    id_delimiter = id_cl_thr_cut

    # Determine constant separation method
    constant_separated_method = param_ef.get("constant_separated", {}).get("method", np.inf)

    if constant_separated_method == "off":
        return segment_initial[id_cl_thr_cut], segment_names, segment_id_hist

    # Additional checks for "on" method
    supp_max = param_ef.get("supp_max", np.inf)
    supp_min = param_ef.get("supp_min", 0)

    # Step 2: Cut out-band constants
    cld_cl_thr = segment_initial[id_cl_thr_cut, segment_names.index("length")]
    mv_cl_thr = segment_initial[id_cl_thr_cut, segment_names.index("mean_value")]
    center_val_weighted = calculate_center_value(mv_cl_thr, cld_cl_thr)

    center_band = [center_val_weighted - mag_thr / 2, center_val_weighted + mag_thr / 2]
    mv = segment_initial[:, segment_names.index("mean_value")]
    index_in_band = (center_band[0] <= mv) & (mv <= center_band[1]) & index_cl_thr
    index_out_band = ~index_in_band & index_cl_thr

    id_in_band = np.where(index_in_band)[0]
    id_out_band = np.where(index_out_band)[0]
    segment_id_hist["Id_CutOut"] = id_in_band
    id_delimiter = id_in_band

    # Step 3: Cut In-Constant in Pattern
    id_not_delimiter_in_pattern = find_constants_in_pattern(segment_initial, segment_names, id_in_band, mag_thr, cl_thr, data_org, mag_info)
    id_delimiter = np.setdiff1d(id_delimiter, id_not_delimiter_in_pattern)

    # Step 4: Cut In-Constant in Leg
    id_not_delimiter_in_leg = find_constants_in_leg(segment_initial, segment_names, id_delimiter, mag_thr, cl_thr, data_org, mag_info)
    id_delimiter = np.setdiff1d(id_delimiter, id_not_delimiter_in_leg)

    # Step 5: Add neighboring constants
    id_delimiter_neib = find_out_constants_in_neighborhood(segment_initial, segment_names, id_delimiter, id_out_band, mag_thr, cl_thr, supp_max, supp_min, mag_info)
    id_delimiter = np.union1d(id_delimiter, id_delimiter_neib)

    # Step 6: Divide long subsequences
    id_delimiter, _ = find_delimiter_divide(segment_initial, segment_names, id_cl_thr_cut, id_delimiter, mag_thr, cl_thr, supp_max, supp_min, mag_info)

    # Final output
    id_delimiter = np.array(id_delimiter, dtype=int)
    segment = segment_initial[id_delimiter]
    segment_id_hist["Id_Final"] = id_delimiter
    return segment, segment_names, segment_id_hist


def calculate_center_value(mv_cl_thr, cld_cl_thr):
    """Calculates center value for thresholds."""
    weighted_sum = np.histogram(mv_cl_thr, weights=cld_cl_thr, bins=10)
    center_index = np.argmax(weighted_sum[0])
    return weighted_sum[1][center_index]


def find_constants_in_pattern(segment_initial, segment_names, id_in_band, mag_thr, cl_thr, data_org, mag_info):
    """Identifies and removes constants in a specific pattern."""
    not_delimiter_ids = []
    for i in range(1, len(id_in_band) - 1):
        current_id = id_in_band[i]
        prev_id = id_in_band[i - 1]
        next_id = id_in_band[i + 1]

        from_cur = segment_initial[current_id, segment_names.index("from")]
        to_cur = segment_initial[current_id, segment_names.index("to")]
        length_cur = segment_initial[current_id, segment_names.index("length")]

        from_prev = segment_initial[prev_id, segment_names.index("from")]
        to_next = segment_initial[next_id, segment_names.index("to")]

        if (to_next - from_prev) <= mag_thr and length_cur < 2 * cl_thr:
            not_delimiter_ids.append(current_id)
    return not_delimiter_ids


def find_constants_in_leg(segment_initial, segment_names, id_delimiter, mag_thr, cl_thr, data_org, mag_info):
    """Finds and removes constants based on their position in a 'leg' structure."""
    not_delimiter_ids = []
    for current_id in id_delimiter:
        from_cur = segment_initial[current_id, segment_names.index("from")]
        to_cur = segment_initial[current_id, segment_names.index("to")]
        from_cur = int(from_cur)
        to_cur = int(to_cur)
        mean_val = np.mean(data_org[from_cur:to_cur])

        if mean_val > mag_thr:
            not_delimiter_ids.append(current_id)
    return not_delimiter_ids

def find_delimiter_divide(segment_initial, segment_names, id_cl_thr_cut, id_delimiter, mag_thr, cl_thr, supp_max, supp_min, mag_info):
    """Adds delimiters to divide long subsequences."""
    for i in range(len(id_delimiter) - 1):
        from_current = segment_initial[id_delimiter[i], segment_names.index("to")]
        to_next = segment_initial[id_delimiter[i + 1], segment_names.index("from")]
        if (to_next - from_current) > supp_max:
            mid_id = np.argmax(segment_initial[id_cl_thr_cut, segment_names.index("length")])
            id_delimiter = np.append(id_delimiter, id_cl_thr_cut[mid_id])
    return np.sort(id_delimiter), None


def connect_delimiter_constants(segment_initial, segment_names, id_delimiter, mag_thr, cl_thr, supp_max, supp_min, mag_info):
    """
    Find overlapping constants and connect them into pairs.

    Parameters:
        segment_initial (numpy.ndarray): Initial segments array.
        segment_names (list): Names of segment columns.
        id_delimiter (list): Indices of delimiter segments.
        mag_thr (float): Magnitude threshold (not used in this function directly).
        cl_thr (float): Constant length threshold (not used here).
        supp_max (float): Maximum support length (not used here).
        supp_min (float): Minimum support length (not used here).
        mag_info (dict): Magnitude information (not used here).

    Returns:
        id_list_connected_delimiter (numpy.ndarray): Connected overlapping segment pairs.
        id_list_connected_delimiter_names (list): Names of the output columns.
    """
    # Initialize the overlap array
    overlap = np.zeros(len(id_delimiter) - 1, dtype=bool)
    
    # Check for overlapping constants
    for i in range(len(id_delimiter) - 1):
        id_current = id_delimiter[i]
        id_next = id_delimiter[i + 1]
        from_current = segment_initial[id_current, segment_names.index("from")]
        to_current = segment_initial[id_current, segment_names.index("to")]
        from_next = segment_initial[id_next, segment_names.index("from")]
        
        if from_next <= to_current:
            overlap[i] = True
    
    # Identify overlapping pairs
    if np.sum(overlap) >= 1:
        id_pair_rel = np.full((len(id_delimiter), 2), np.nan)
        pre = False
        count = 0

        for i in range(len(overlap)):
            if not pre and overlap[i]:  # Overlapping starts
                from_idx = i
            elif not pre and not overlap[i]:  # No overlap
                count += 1
                id_pair_rel[count - 1, :] = [i, i]
            elif pre and overlap[i]:  # Overlapping continues
                if i == len(overlap) - 1:  # End case
                    count += 1
                    id_pair_rel[count - 1, :] = [from_idx, i]
            elif pre and not overlap[i]:  # Overlapping ends
                to_idx = i
                count += 1
                id_pair_rel[count - 1, :] = [from_idx, to_idx]
            pre = overlap[i]

        id_pair_rel = id_pair_rel[~np.isnan(id_pair_rel[:, 0]), :].astype(int)
    else:
        id_pair_rel = np.column_stack((np.arange(len(id_delimiter)), np.arange(len(id_delimiter))))

    # Build the output connected delimiter list
    id_list_connected_delimiter_names = ["id_from", "id_to", "from", "to"]
    if id_pair_rel.size > 0:
        id_list_connected_delimiter = np.zeros((id_pair_rel.shape[0], len(id_list_connected_delimiter_names)))
        id_list_connected_delimiter[:, 0] = np.array(id_delimiter)[id_pair_rel[:, 0]]
        id_list_connected_delimiter[:, 1] = np.array(id_delimiter)[id_pair_rel[:, 1]]
        id_list_connected_delimiter[:, 2] = segment_initial[id_list_connected_delimiter[:, 0].astype(int), segment_names.index("from")]
        id_list_connected_delimiter[:, 3] = segment_initial[id_list_connected_delimiter[:, 1].astype(int), segment_names.index("to")]
    else:
        id_list_connected_delimiter = np.array([])

    return id_list_connected_delimiter, id_list_connected_delimiter_names

def find_delimiter_divide(segment_initial, segment_names, id_clthr_cut, id_delimiter, mag_thr, cl_thr, supp_max, supp_min, mag_info):
    """
    Repeatedly finds additional delimiters by dividing long sequences without constants.

    Parameters:
        segment_initial (numpy.ndarray): Segment data array.
        segment_names (list): Names of the segment columns.
        id_clthr_cut (list): IDs of segments meeting the CLThr criterion.
        id_delimiter (list): Initial delimiter IDs.
        mag_thr (float): Magnitude threshold.
        cl_thr (float): Constant length threshold.
        supp_max (float): Maximum support length.
        supp_min (float): Minimum support length.
        mag_info (dict): Magnitude information.

    Returns:
        id_delimiter (list): Updated delimiter IDs.
        id_delimiter_add (list): Newly added delimiter IDs.
    """
    id_delimiter_original = id_delimiter[:]
    id_delimiters_add = [0]

    while id_delimiters_add:
        id_delimiter, id_delimiters_add = find_delimiter_divide_step(
            segment_initial, segment_names, id_clthr_cut, id_delimiter, mag_thr, cl_thr, supp_max, supp_min, mag_info
        )
    
    id_delimiter_add = list(set(id_delimiter) - set(id_delimiter_original))
    return id_delimiter, id_delimiter_add

def find_delimiter_divide_step(segment_initial, segment_names, id_clthr_cut, id_delimiters_org, mag_thr, cl_thr, supp_max, supp_min, mag_info):
    """
    Finds additional delimiters by examining gaps between existing delimiters.

    Parameters:
        segment_initial (numpy.ndarray): Segment data array.
        segment_names (list): Names of the segment columns.
        id_clthr_cut (list): IDs of segments meeting the CLThr criterion.
        id_delimiters_org (list): Existing delimiter IDs.
        mag_thr (float): Magnitude threshold.
        cl_thr (float): Constant length threshold.
        supp_max (float): Maximum support length.
        supp_min (float): Minimum support length.
        mag_info (dict): Magnitude information.

    Returns:
        id_delimiters (list): Updated delimiter IDs.
        id_delimiters_add (list): Newly added delimiter IDs.
    """
    id_candidates = list(set(id_clthr_cut) - set(id_delimiters_org))
    index_candidates = np.zeros(len(id_candidates), dtype=bool)

    for i in range(len(id_delimiters_org) - 1):
        id_current = int(id_delimiters_org[i])
        id_next = int(id_delimiters_org[i + 1])



        from_current = segment_initial[id_current, segment_names.index("from")]
        to_current = segment_initial[id_current, segment_names.index("to")]

        from_next = segment_initial[id_next, segment_names.index("from")]

        if (from_next - to_current) > supp_max:
            index_mids = [id_current < id_cand < id_next for id_cand in id_candidates]
            if any(index_mids):
                mids = np.array(id_candidates)[index_mids]
                max_idx = np.argmax(segment_initial[mids, segment_names.index("length")])
                index_candidates[id_candidates.index(mids[max_idx])] = True

    id_delimiters_add = list(np.array(id_candidates)[index_candidates])
    id_delimiters = sorted(set(np.concatenate([id_delimiters_org, id_delimiters_add])))
    return id_delimiters, id_delimiters_add

def find_out_constants_in_neighborhood(segment_initial, segment_names, id_delimiter, id_outband, mag_thr, cl_thr, supp_max, supp_min, mag_info):
    """
    Finds constants in the neighborhood of existing delimiter constants.

    Parameters:
        segment_initial (numpy.ndarray): Segment data array.
        segment_names (list): Names of the segment columns.
        id_initial_delimiters (list): IDs of initial delimiters.
        id_list_connected_delimiter (numpy.ndarray): Connected delimiter list.
        id_list_connected_delimiter_names (list): Names of the delimiter columns.
        id_outband (list): IDs of outband constants.
        mag_thr (float): Magnitude threshold.
        cl_thr (float): Constant length threshold.
        supp_max (float): Maximum support length.
        supp_min (float): Minimum support length.
        mag_info (dict): Magnitude information.

    Returns:
        id_delimiter_outband (list): IDs of delimiters in the neighborhood of existing delimiters.
    """
    data_org = mag_info["data_org"]
    magnitude = mag_info["magnitude"]

    index_delimiter_outband = np.zeros(len(id_outband), dtype=bool)

    for i, id_out in enumerate(id_outband):
        from_out = segment_initial[id_out, segment_names.index("from")]
        to_out = segment_initial[id_out, segment_names.index("to")]
        mean_out = segment_initial[id_out, segment_names.index("mean_value")]

        # Find left and right neighboring constants
        id_left_pos = np.searchsorted(id_delimiter, id_out, side='left') - 1
        id_right_pos = np.searchsorted(id_delimiter, id_out, side='right')

        id_left = id_delimiter[id_left_pos] if id_left_pos >= 0 else None
        id_right = id_delimiter[id_right_pos] if id_right_pos < len(id_delimiter) else None

        if id_left is not None:
            mean_left = segment_initial[id_left, segment_names.index("mean_value")]
            to_left = segment_initial[id_left, segment_names.index("to")]

            if abs(mean_left - mean_out) <= mag_thr / 2 or to_left >= from_out:
                index_delimiter_outband[i] = True

        if id_right is not None:
            mean_right = segment_initial[id_right, segment_names.index("mean_value")]
            from_right = segment_initial[id_right, segment_names.index("from")]

            if abs(mean_right - mean_out) <= mag_thr / 2 or from_right <= to_out:
                index_delimiter_outband[i] = True

    id_delimiter_outband = np.array(id_outband)[index_delimiter_outband].tolist()
    return id_delimiter_outband

def undo_delimiter_in_band(segment_in_band, segment_names, id_in_band_delimiter, supp_max):
    """
    Repeatedly refines delimiters in-band until no further changes are found.

    Parameters:
        segment_in_band (numpy.ndarray): Segment data array.
        segment_names (list): Names of the segment columns.
        id_in_band_delimiter (list): IDs of delimiters in-band.
        supp_max (float): Maximum support length.

    Returns:
        id_in_band_delimiter (list): Updated list of delimiter IDs.
        id_in_band_undo (list): IDs that were re-added.
    """
    id_in_band_undo = []
    found = 1

    while found >= 1:
        id_in_band_delimiter, id_in_band_undo_i = undo_delimiter_in_band_step(
            segment_in_band, segment_names, id_in_band_delimiter, supp_max
        )
        id_in_band_undo = list(set(id_in_band_undo) | set(id_in_band_undo_i))
        found = len(id_in_band_undo_i)

    return id_in_band_delimiter, id_in_band_undo


def undo_delimiter_in_band_step(segment_in_band, segment_names, id_in_band_delimiter, supp_max):
    """
    A single step to refine delimiters by checking between consecutive delimiters.

    Parameters:
        segment_in_band (numpy.ndarray): Segment data array.
        segment_names (list): Names of the segment columns.
        id_in_band_delimiter (list): IDs of delimiters in-band.
        supp_max (float): Maximum support length.

    Returns:
        id_in_band_delimiter (list): Updated list of delimiter IDs.
        id_in_band_undo (list): IDs that were re-added.
    """
    id_in_band_undo = []

    id_in_band = np.arange(segment_in_band.shape[0])
    id_in_band_not_delimiter = list(set(id_in_band) - set(id_in_band_delimiter))

    for i in range(len(id_in_band_delimiter) - 1):
        id_current = id_in_band_delimiter[i]
        id_next = id_in_band_delimiter[i + 1]

        from_current = segment_in_band[id_current, segment_names.index("from")]
        to_current = segment_in_band[id_current, segment_names.index("to")]
        from_next = segment_in_band[id_next, segment_names.index("from")]

        if (from_next - to_current) > supp_max:
            mids = [id for id in id_in_band_not_delimiter if id_current < id < id_next]
            if mids:
                max_id = max(mids, key=lambda x: segment_in_band[x, segment_names.index("length")])
                id_in_band_undo.append(max_id)

    id_in_band_delimiter = sorted(set(id_in_band_delimiter) | set(id_in_band_undo))
    return id_in_band_delimiter, id_in_band_undo

def find_delimiter_outband_step_ver_02(delimiter_outband, segment_initial, segment_names, id_outband, 
                                       id_initial_in_band_delimiter, mag_thr, cl_thr, supp_max, supp_min, mag_info):
    """
    Refines delimiters by identifying constants in outband regions.

    Parameters:
        delimiter_outband (numpy.ndarray): Boolean array indicating current delimiters.
        segment_initial (numpy.ndarray): Segment data array.
        segment_names (list): Names of the segment columns.
        id_outband (list): IDs of outband constants.
        id_initial_in_band_delimiter (list): IDs of in-band delimiters.
        mag_thr (float): Magnitude threshold.
        cl_thr (float): Constant length threshold.
        supp_max (float): Maximum support length.
        supp_min (float): Minimum support length.
        mag_info (dict): Magnitude information.

    Returns:
        delimiter_outband (numpy.ndarray): Updated delimiter array.
        found (int): Number of new delimiters found.
    """
    found = 0
    id_initial_outband = np.array(id_outband)
    id_initial_outband_delimiter = id_initial_outband[delimiter_outband]
    id_initial_delimiters = sorted(set(id_initial_in_band_delimiter) | set(id_initial_outband_delimiter))

    for i in range(len(id_initial_outband)):
        id_current = id_initial_outband[i]
        from_current = segment_initial[id_current, segment_names.index("from")]
        to_current = segment_initial[id_current, segment_names.index("to")]
        mean_value = segment_initial[id_current, segment_names.index("mean_value")]

        # Find left and right neighboring constants
        left_indices = [idx for idx in id_initial_delimiters if idx < id_current]
        right_indices = [idx for idx in id_initial_delimiters if idx > id_current]

        id_left = left_indices[-1] if left_indices else None
        id_right = right_indices[0] if right_indices else None

        if id_left is not None:
            mean_left = segment_initial[id_left, segment_names.index("mean_value")]
            if abs(mean_left - mean_value) <= mag_thr / 2:
                delimiter_outband[i] = True
                found += 1

        if id_right is not None:
            mean_right = segment_initial[id_right, segment_names.index("mean_value")]
            if abs(mean_right - mean_value) <= mag_thr / 2:
                delimiter_outband[i] = True
                found += 1

    return delimiter_outband, found

def find_delimiter_outband_step(delimiter_outband, segment_initial, segment_names, id_outband, id_inband_cipcut, supp_max):
    """
    Refines delimiters by searching for constants between delimiters and outband segments.

    Parameters:
        delimiter_outband (numpy.ndarray): Boolean array indicating current delimiters.
        segment_initial (numpy.ndarray): Segment data array.
        segment_names (list): Names of the segment columns.
        id_outband (list): IDs of outband constants.
        id_inband_cipcut (list): IDs of in-band constants.
        supp_max (float): Maximum allowed gap length.

    Returns:
        delimiter_outband (numpy.ndarray): Updated delimiter array.
        found (int): Number of newly identified delimiters.
    """
    found = 0

    # Combine in-band delimiters and outband delimiters
    id_outband_delimiter = id_outband[delimiter_outband]
    delimiters = sorted(set(id_inband_cipcut) | set(id_outband_delimiter))

    # Check for gaps and find constants in the gaps
    for i in range(len(delimiters) - 1):
        id_current = delimiters[i]
        id_next = delimiters[i + 1]

        from_current = segment_initial[id_current, segment_names.index("from")]
        to_current = segment_initial[id_current, segment_names.index("to")]
        from_next = segment_initial[id_next, segment_names.index("from")]

        if (from_next - to_current) > supp_max:
            mids = [idx for idx in id_outband if id_current < idx < id_next]
            if mids:
                max_id = max(mids, key=lambda x: segment_initial[x, segment_names.index("length")])
                index_outband = np.where(id_outband == max_id)[0][0]
                delimiter_outband[index_outband] = True
                found += 1

    return delimiter_outband, found

def correct_not_delimiter_in_band_step(not_delimiter_in_band, segment_in_band, segment_names, supp_max):
    """
    Corrects delimiters by reintroducing previously removed candidates based on segment gaps.

    Parameters:
        not_delimiter_in_band (numpy.ndarray): Boolean array indicating non-delimiters.
        segment_in_band (numpy.ndarray): Segment data array.
        segment_names (list): Names of the segment columns.
        supp_max (float): Maximum allowed gap length.

    Returns:
        not_delimiter_in_band (numpy.ndarray): Updated non-delimiter array.
        found (int): Number of corrections made.
    """
    found = 0

    # Find indices of deleted and delimiter segments
    deleted_ids = np.where(not_delimiter_in_band)[0]
    delimiter_ids = np.where(~not_delimiter_in_band)[0]

    # Iterate through delimiters and check for gaps
    for i in range(1, len(delimiter_ids) - 1):
        id_current = delimiter_ids[i]
        id_prev = delimiter_ids[i - 1]
        id_next = delimiter_ids[i + 1]

        from_current = segment_in_band[id_current, segment_names.index("from")]
        to_current = segment_in_band[id_current, segment_names.index("to")]
        to_prev = segment_in_band[id_prev, segment_names.index("to")]
        from_next = segment_in_band[id_next, segment_names.index("from")]

        # Check gaps before and after the current delimiter
        if (from_current - to_prev) > supp_max:
            mids = [idx for idx in deleted_ids if id_prev < idx < id_current]
            if mids:
                max_id = max(mids, key=lambda x: segment_in_band[x, segment_names.index("length")])
                not_delimiter_in_band[max_id] = False
                found += 1

        if (from_next - to_current) > supp_max:
            mids = [idx for idx in deleted_ids if id_current < idx < id_next]
            if mids:
                max_id = max(mids, key=lambda x: segment_in_band[x, segment_names.index("length")])
                not_delimiter_in_band[max_id] = False
                found += 1

    return not_delimiter_in_band, found



