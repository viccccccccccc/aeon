import numpy as np
import re

def Spikelet_Word_query(MagInfo, Query):
    # Debugging flags
    DEBUG_Word_support_length = False
    Debug = False

    # Extract necessary data
    Data_org = MagInfo['data_org']
    SpikeDb = MagInfo['spikeDb']
    CMS_str = SpikeDb['alphabet']
    
    # Prepare structures for results
    string_feature = [None] * len(Query)
    string_feature_names = ["peaknum", "leg2norm", "leg1norm", "l2norm", "l1norm", 
                            "leg2norm_mean", "leg1norm_mean", "l2norm_mean", "l1norm_mean"]
    
    string_query = [None] * len(Query)
    string_query_names = ["global_id", "local_id", "query_id", "start_spike_id", "end_spike_id", 
                          "char_length", "from_org", "to_org", "from_percentile", "to_percentile", 
                          "from", "to", "word_length"]

    string_info = [None] * len(Query)
    Op = 'word_query'
    MagInfo['size'][Op] = np.full((len(Query), 2), np.nan)
    MagInfo['time'][Op] = np.full(len(Query), np.nan)

    for i, Query_i in enumerate(Query):
        SuppMax_i = Query_i.get('supp_max', float('inf'))
        SuppMin_i = Query_i.get('supp_min', 0)

        # Determine terminal methods
        method_left = Query_i['terminal_left'].split('_') if '_' in Query_i['terminal_left'] else [Query_i['terminal_left']]
        LRML_left = f"left_{method_left[0]}"
        percentile_left = float(method_left[1]) if len(method_left) > 1 else np.nan

        method_right = Query_i['terminal_right'].split('_') if '_' in Query_i['terminal_right'] else [Query_i['terminal_right']]
        LRML_right = f"right_{method_right[0]}"
        percentile_right = float(method_right[1]) if len(method_right) > 1 else np.nan

        # Query execution
        start_WQt_i = time.time()
        startIndex_i, endIndex_i = Spikelet_regexp(CMS_str, Query_i['pattern'])
        MagInfo['size'][Op][i, 0] = len(startIndex_i)

        if DEBUG_Word_support_length:
            # Plotting for debugging
            plot_Word_support_length(startIndex_i, endIndex_i, CMS_str, Query_i['pattern'], SpikeDb, Data_org)
        
        # Apply constraints
        if Query_i.get('constraint'):
            Scand_i = np.column_stack((startIndex_i, endIndex_i))
            Sindex_i = np.ones(len(Scand_i), dtype=bool)
            for j, (start, end) in enumerate(Scand_i):
                for spike_id_k in range(start, end + 1):
                    symbol_i = CMS_str[start:end + 1]
                    order_ijk = spike_id_k - start
                    constraint_ijk = get_constraint(order_ijk, symbol_i, Query_i['pattern'], Query_i['constraint'])
                    check_ijk = check_constraint(spike_id_k, constraint_ijk, MagInfo)
                    Sindex_i[j] &= check_ijk
                    if not Sindex_i[j]:
                        break
            startIndex_i = Scand_i[Sindex_i, 0]
            endIndex_i = Scand_i[Sindex_i, 1]

        MagInfo['size'][Op][i, 1] = len(startIndex_i)
        MagInfo['time'][Op][i] = time.time() - start_WQt_i
        print(f"{Op}_{i + 1}: size = {MagInfo['size'][Op][i, 0]} -> {MagInfo['size'][Op][i, 1]}")
        print(f"{Op}_{i + 1}: time = {MagInfo['time'][Op][i]}")

        # Setup string for each query
        string_i = np.full((len(startIndex_i), len(string_query_names)), np.nan)
        string_i[:, string_query_names.index("local_id")] = np.arange(1, len(startIndex_i) + 1)
        string_i[:, string_query_names.index("query_id")] = i + 1
        string_i[:, string_query_names.index("start_spike_id")] = startIndex_i
        string_i[:, string_query_names.index("end_spike_id")] = endIndex_i
        string_i[:, string_query_names.index("char_length")] = endIndex_i - startIndex_i + 1
        string_i[:, string_query_names.index("from_org")] = SpikeDb['from'][startIndex_i]
        string_i[:, string_query_names.index("to_org")] = SpikeDb['to'][endIndex_i]
        string_i[:, string_query_names.index("from_percentile")] = extract_terminal(SpikeDb['center'][startIndex_i], LRML_left, percentile_left, MagInfo)
        string_i[:, string_query_names.index("to_percentile")] = extract_terminal(SpikeDb['center'][endIndex_i], LRML_right, percentile_right, MagInfo)
        string_i[:, string_query_names.index("from")] = extract_terminal(SpikeDb['center'][startIndex_i], LRML_left, percentile_left, MagInfo)
        string_i[:, string_query_names.index("to")] = extract_terminal(SpikeDb['center'][endIndex_i], LRML_right, percentile_right, MagInfo)
        string_i[:, string_query_names.index("word_length")] = string_i[:, string_query_names.index("to")] - string_i[:, string_query_names.index("from")] + 1

        # Debugging plot
        if DEBUG_Word_support_length:
            plot_Word_support_length_time(string_i[:, string_query_names.index("from_percentile")], 
                                          string_i[:, string_query_names.index("to_percentile")], Data_org)

        # Length check
        Word_length_i = string_i[:, string_query_names.index("to")] - string_i[:, string_query_names.index("from")] + 1
        Index_SuppCheck_i = np.ones(len(string_i), dtype=bool)
        if 'supp_max' in Query_i:
            Index_SuppCheck_i &= (Word_length_i <= Query_i['supp_max'])
        if 'supp_min' in Query_i:
            Index_SuppCheck_i &= (Word_length_i >= Query_i['supp_min'])

        print("The following are deleted by support length check")
        print(string_i[~Index_SuppCheck_i, [string_query_names.index("from"), string_query_names.index("to")]])

        string_i = string_i[Index_SuppCheck_i]
        startIndex_i = startIndex_i[Index_SuppCheck_i]
        endIndex_i = endIndex_i[Index_SuppCheck_i]

        # Calculate features and store information
        string_info_i = [None] * len(string_i)
        string_feature_i = np.full((len(string_i), len(string_feature_names)), np.nan)
        for j in range(len(string_i)):
            # Gather string info
            string_info_ij = {
                "global_id": np.nan,
                "local_id": j + 1,
                "query_id": i + 1,
                "alphabet": CMS_str[startIndex_i[j]:endIndex_i[j] + 1],
                "pattern": Query_i['pattern'],
                "constraint": Query_i['constraint'],
                "color": Query_i['color'],
                "spike_index": (startIndex_i[j], endIndex_i[j]),
                "time_index": (SpikeDb['from'][startIndex_i[j]], SpikeDb['to'][endIndex_i[j]]),
                "center_list": SpikeDb['center'][startIndex_i[j]:endIndex_i[j] + 1],
                "type_list": SpikeDb['type'][startIndex_i[j]:endIndex_i[j] + 1],
                "terminal_index": (string_i[j, string_query_names.index('from')], string_i[j, string_query_names.index('to')])
            }
            string_info_i[j] = string_info_ij

            # Calculate features
            Peaks_ij = peaks_in_string(string_info_ij)
            string_feature_i[j, :] = Spikelet_String_features(zscore(Data_org), Peaks_ij, string_feature_names)

        string_feature[i] = string_feature_i
        string_query[i] = string_i
        string_info[i] = string_info_i

    if Debug:
        for info in string_info_i:
            print(f"[{info['time_index'][0]} - {info['time_index'][1]}] {info['alphabet']}")

        From, To = 3600, 3900
        Index = np.where((SpikeDb['from'] >= From) & (SpikeDb['from'] <= To))[0]
        for spike_id in Index:
            print(f"[{SpikeDb['from'][spike_id]} {SpikeDb['to'][spike_id]}] {SpikeDb['type'][spike_id]}")

    # Output results
    QueryRslt = {
        "query": Query,
        "symbolic_representation": CMS_str,
        "string_query_names": string_query_names,
        "string_feature_names": string_feature_names,
        "string_query": string_query,
        "string_feature": string_feature,
        "string_info": string_info
    }
    
    MagInfo['word_query_result'] = QueryRslt
    return MagInfo, QueryRslt

def Spikelet_regexp(CMS_str, pattern):
    # Use Python's regular expressions to match patterns
    matches = [(m.start(), m.end()) for m in re.finditer(pattern, CMS_str)]
    start_index = np.array([m[0] for m in matches])
    end_index = np.array([m[1] - 1 for m in matches])  # Adjust for inclusive end index
    return start_index, end_index

def get_constraint(order, symbol, pattern, Constraint_list):
    return Constraint_list[order]

def check_constraint(spike_id, constraint, MagInfo):
    assigned = assign_value(spike_id, constraint, MagInfo)
    return eval(assigned)

def assign_value(spike_id, constraint, MagInfo):
    SpikeDb = MagInfo['spikeDb']
    if 'Center_val' in constraint:
        center = SpikeDb['center'][spike_id]
        val = MagInfo['data_org'][center]
        constraint = constraint.replace('Center_val', str(val))
    if 'Mag' in constraint:
        center = SpikeDb['center'][spike_id]
        val = MagInfo['magnitude'][center]
        constraint = constraint.replace('Mag', str(val))
    supp_pattern_cell = extract_supp_pattern(constraint)
    if supp_pattern_cell:
        for supp_var_i in supp_pattern_cell:
            if supp_var_i in SpikeDb:
                val = SpikeDb[supp_var_i][spike_id]
                constraint = constraint.replace(supp_var_i, str(val))
            else:
                constraint = assign_supp(center, constraint, supp_var_i, MagInfo)
    return constraint

def extract_supp_pattern(constraint):
    pattern_supp = re.compile(r'Supp\w{3}_\d{1,3}')
    return pattern_supp.findall(constraint)

def assign_supp(center, constraint, supp_var, MagInfo):
    supp_split = supp_var.split('_')
    feature = supp_split[0]
    percentile = int(supp_split[1])
    if feature == 'SuppCBM':
        from_idx = Spikelet_Spike_percentile(center, 'left_mag', percentile, MagInfo)
        to_idx = Spikelet_Spike_percentile(center, 'right_mag', percentile, MagInfo)
        val = to_idx - from_idx + 1
    constraint = constraint.replace(supp_var, str(val))
    return constraint

def Spikelet_Spike_percentile(center, LRML, percentile, MagInfo):
    # Placeholder function to calculate the percentile location
    # Implement according to specific data and operations
    return center  # Example return value

def peaks_in_string(string_info):
    from_idx, to_idx = string_info['terminal_index']
    center_list = string_info['center_list']
    type_list = string_info['type_list']
    peak_list_index = np.zeros(len(type_list), dtype=bool)
    for i, type_val in enumerate(type_list):
        if type_val != 0 or (type_val == 0 and i >= 1 and i < len(type_list) - 1 and type_list[i - 1] * type_list[i + 1] == 1):
            peak_list_index[i] = True
    peak_list = center_list[peak_list_index]
    Peaks = peak_list[(from_idx <= peak_list) & (peak_list <= to_idx)]
    if Peaks[0] != from_idx:
        Peaks = np.insert(Peaks, 0, from_idx)
    if Peaks[-1] != to_idx:
        Peaks = np.append(Peaks, to_idx)
    return Peaks

def Spikelet_String_features(D, Times, string_feature_names):
    Vals = D[Times]
    Diff = np.diff(Vals)
    leg2sum = np.sqrt(np.sum(Diff ** 2))
    leg1sum = np.sum(np.abs(Diff))
    peaknum = len(Times)
    l2norm = np.sqrt(np.sum(Vals ** 2))
    l1norm = np.sum(np.abs(Vals))
    leg2sum_mean = leg2sum / len(Diff)
    leg1sum_mean = leg1sum / len(Diff)
    l2norm_mean = l2norm / len(Vals)
    l1norm_mean = l1norm / len(Vals)
    F = [peaknum, leg2sum, leg1sum, l2norm, l1norm, leg2sum_mean, leg1sum_mean, l2norm_mean, l1norm_mean]
    if len(F) != len(string_feature_names):
        print(f"[Spikelet_String_features] Error: mismatch between features and names")
    return F

def extract_terminal(center_list, LRML, percentile, MagInfo):
    Terminal_list = np.full(len(center_list), np.nan)
    for i, center in enumerate(center_list):
        Terminal_list[i] = Spikelet_Spike_percentile(center, LRML, percentile, MagInfo)
    return Terminal_list

def plot_Word_support_length(startIndex, endIndex, CMS_str, patternDef, SpikeDb, Data_org):
    LINEWIDTH = 2
    from_id_list = startIndex + 1 if patternDef[0] == 'C' else startIndex
    to_id_list = endIndex - 1 if patternDef[-1] == 'C' else endIndex
    from_terminal_time = SpikeDb['from'][from_id_list]
    to_terminal_time = SpikeDb['to'][to_id_list]
    plot_Word_support_length_time(from_terminal_time, to_terminal_time, Data_org)

    for i in range(len(from_id_list)):
        from_id = from_id_list[i]
        to_id = to_id_list[i]

        from_center_time = SpikeDb['center'][from_id]
        to_center_time = SpikeDb['center'][to_id]
        from_terminal_time = SpikeDb['from'][from_id]
        to_terminal_time = SpikeDb['to'][to_id]
        CMS_word = CMS_str[startIndex[i]:endIndex[i] + 1]
        MFT_word = np.column_stack((SpikeDb['magnitude'][startIndex[i]:endIndex[i] + 1],
                                    SpikeDb['from'][startIndex[i]:endIndex[i] + 1],
                                    SpikeDb['to'][startIndex[i]:endIndex[i] + 1]))

        # Visualization code commented out for simplification
        # if i % 2 == 0:
        #     Color = 'b'
        # else:
        #     Color = 'r'
        # plot((from_terminal_time:to_terminal_time), Data_org[from_terminal_time:to_terminal_time], Color, 'LineWidth', LINEWIDTH)

def plot_Word_support_length_time(Word_from_i, Word_to_i, Data_org):
    LINEWIDTH = 2
    import matplotlib.pyplot as plt
    plt.plot(range(1, len(Data_org) + 1), Data_org)
    for wsl in range(len(Word_from_i)):
        Range = range(Word_from_i[wsl], Word_to_i[wsl] + 1)
        plt.plot(Range, Data_org[Range], 'r', linewidth=LINEWIDTH)
    plt.show()

