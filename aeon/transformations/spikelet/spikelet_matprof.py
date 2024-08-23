import time

import numpy as np
from scipy.spatial.distance import euclidean

# from fastdtw import fastdtw


def Spikelet_MP_new_ver_02(MagInfo, MP_param):
    func_name = "Spikelet_MP_new_ver_02"

    # Plotting flag
    PLOT_STAT = False

    Op = "matrix_profile"
    OpDetail = "matrix_profile_detail"

    # Arguments
    QR = MagInfo["word_query_result"]
    StringMatrix_names = QR["string_query_names"]
    StringMatrix_cell = QR["string_query"]

    # Indexing
    if "prune_method" in MP_param:
        if MP_param["prune_method"] == "first_spike":
            for i in range(len(StringMatrix_cell)):
                start_time = time.time()
                StringDb = StringMatrix_cell[i]
                SpikeSet = StringDb[:, StringMatrix_names.index("start_spike_id")]
                PQ_array = MP_param["prune_query"]
                StatVarArray = extractVarInArray(PQ_array)

                # Calculate statistics
                MagInfo, SpikeDb = Spikelet_SpikeDb_statistics(
                    MagInfo, StatVarArray, SpikeSet
                )
                for j, pq in enumerate(PQ_array):
                    var_split = pq.split(" ")
                    feature = var_split[0]
                    type_ = var_split[1]
                    MagInfo["mp"]["index"][feature] = SpikeDb[feature][SpikeSet]
                    if type_ == "percentile":
                        feature_string_val = np.sort(MagInfo["mp"]["index"][feature])
                        feature_string_index = np.argsort(
                            MagInfo["mp"]["index"][feature]
                        )
                        MagInfo["mp"]["index"][
                            f"{feature}_sorted_val"
                        ] = feature_string_val
                        MagInfo["mp"]["index"][
                            f"{feature}_sorted_index"
                        ] = feature_string_index

                Op1 = "indexing"
                MagInfo["time"][OpDetail][Op1] = time.time() - start_time

                if PLOT_STAT:
                    for var in StatVarArray:
                        Stat = SpikeDb[var]
                        Stat = Stat[~np.isnan(Stat)]
                        plt.hist(Stat)
                        plt.title(f"{var} mean={np.mean(Stat)} std={np.std(Stat)}")
                        plt.show()

    # Calculate MP for each query
    MP_query_result = [None] * len(StringMatrix_cell)
    MagInfo["size"][Op] = np.full((len(QR["query"]), 2), np.nan)
    MagInfo["time"][Op] = np.full(len(QR["query"]), np.nan)

    for i, StringDb in enumerate(StringMatrix_cell):
        start_SMCi = time.time()

        (
            string_matrix_i,
            string_matrix_names,
            MP_trend_raw_i,
            MP_trend_norm_i,
            dist_calc_num,
            TOPK_norm,
            TOPK_names,
        ) = calculateMP(StringDb, StringMatrix_names, MP_param, MagInfo)

        if string_matrix_i is not None:
            Op2 = "calculate_MP"
            MagInfo["size"][OpDetail][Op2][i, :] = [
                dist_calc_num["before"],
                dist_calc_num["after"],
            ]
            MagInfo["size"][Op][i, :] = [
                dist_calc_num["before"],
                dist_calc_num["after"],
            ]
            MagInfo["time"][OpDetail][Op2][i] = time.time() - start_SMCi

            if "prune_method" in MP_param:
                MagInfo["time"][Op][i] = (
                    MagInfo["time"][OpDetail][Op1][i]
                    + MagInfo["time"][OpDetail][Op2][i]
                )
            else:
                MagInfo["time"][Op][i] = MagInfo["time"][OpDetail][Op2][i]

            print(
                f'MP_[{i}:{QR["query"][i]["pattern"]}] the number of distance calculation {dist_calc_num["before"]} -> {dist_calc_num["after"]}'
            )
            print(
                f'MP_[{i}:{QR["query"][i]["pattern"]}] time = {MagInfo["time"][Op][i]}'
            )
        else:
            print(f"no words\n")
            print(
                f'MP_[{i}:{QR["query"][i]["pattern"]}] time = {MagInfo["time"][Op][i]}'
            )

        Query_result_i = {
            "string_matrix": string_matrix_i,
            "string_matrix_names": string_matrix_names,
            "mp_trend_raw": MP_trend_raw_i,
            "mp_trend_norm": MP_trend_norm_i,
            "topk_norm": TOPK_norm,
            "topk_names": TOPK_names,
        }

        MP_query_result[i] = Query_result_i

    # Output
    MP = {"query_result": MP_query_result}
    MagInfo["mp"] = MP
    return MagInfo, MP


def calculateMP(StringMatrix, StringMatrix_names, MP_param, MagInfo):
    TOPK_names = [
        "score",
        "string_id_1",
        "string_id_2",
        "from_1",
        "to_1",
        "from_2",
        "to_2",
    ]
    TopK = MP_param.get("topk", 1)
    TOPK_norm = np.full((TopK, len(TOPK_names)), np.inf)
    TOPK_pos = 0
    TOPK_thr = np.inf

    # Data structure MP
    MP_names_add = ["nn_dist_raw", "nn_pair_raw", "nn_dist_norm", "nn_pair_norm"]
    MP_names = StringMatrix_names + MP_names_add

    if StringMatrix is None or len(StringMatrix) == 0:
        return None, None, None, None, None, None, None

    # Initialize output
    MP = np.full((len(StringMatrix), len(MP_names)), np.inf)
    MP[:, : len(StringMatrix_names)] = StringMatrix
    MP_trend_raw = np.full(len(MagInfo["data_org"]), np.inf)
    MP_trend_norm = np.full(len(MagInfo["data_org"]), np.inf)

    distcalc_num = {"before": len(StringMatrix) * (len(StringMatrix) - 1), "after": 0}

    # Calculate MP
    from_pos = StringMatrix_names.index("from")
    to_pos = StringMatrix_names.index("to")
    String_Terminal = StringMatrix[:, [from_pos, to_pos]]

    for i in range(len(StringMatrix)):
        from_i, to_i = String_Terminal[i]
        Si = MagInfo["data_org"][int(from_i) : int(to_i) + 1]

        StringId_index_i = StringDb_prune(
            i, StringMatrix, StringMatrix_names, MP_param, MagInfo
        )
        StringId_list_i = np.where(StringId_index_i == True)[0]
        distcalc_num["after"] += len(StringId_list_i)

        if len(StringId_list_i) == 0:
            MP[i, MP_names.index("nn_dist_raw")] = np.inf
            MP[i, MP_names.index("nn_pair_raw")] = np.nan
            MP[i, MP_names.index("nn_dist_norm")] = np.inf
            MP[i, MP_names.index("nn_pair_norm")] = np.nan
        else:
            for string_id_ij in StringId_list_i:
                from_j, to_j = String_Terminal[string_id_ij]
                Sj = MagInfo["data_org"][int(from_j) : int(to_j) + 1]

                if (from_j <= from_i and from_i < to_j) or (
                    from_i <= from_j and from_j < to_i
                ):
                    dist_raw_ij = np.inf
                    dist_norm_ij = np.inf
                else:
                    dist_raw_ij, dist_norm_ij = pair_distance(
                        Si, Sj, MP_param["distance"]
                    )

                if dist_raw_ij < MP[i, MP_names.index("nn_dist_raw")]:
                    MP[i, MP_names.index("nn_dist_raw")] = dist_raw_ij
                    MP[i, MP_names.index("nn_pair_raw")] = string_id_ij

                if dist_norm_ij < MP[i, MP_names.index("nn_dist_norm")]:
                    MP[i, MP_names.index("nn_dist_norm")] = dist_norm_ij
                    MP[i, MP_names.index("nn_pair_norm")] = string_id_ij

                # TopK
                if "TOPK_norm" in locals():
                    if TOPK_thr == np.inf and dist_norm_ij != np.inf:
                        TOPK_pos += 1
                        TOPK_norm[TOPK_pos - 1] = [
                            dist_norm_ij,
                            i,
                            string_id_ij,
                            from_i,
                            to_i,
                            from_j,
                            to_j,
                        ]
                        if TOPK_pos == TopK:
                            TOPK_thr = np.max(TOPK_norm[:, 0])
                    elif dist_norm_ij < TOPK_thr:
                        TOPK_norm[TOPK_pos - 1] = [
                            dist_norm_ij,
                            i,
                            string_id_ij,
                            from_i,
                            to_i,
                            from_j,
                            to_j,
                        ]
                        TOPK_thr = np.max(TOPK_norm[:, 0])

            # Update matrix profile trends
            MP_trend_norm[int(from_i) : int(to_i) + 1] = np.minimum(
                MP_trend_norm[int(from_i) : int(to_i) + 1],
                MP[i, MP_names.index("nn_dist_norm")],
            )
            MP_trend_raw[int(from_i) : int(to_i) + 1] = np.minimum(
                MP_trend_raw[int(from_i) : int(to_i) + 1],
                MP[i, MP_names.index("nn_dist_raw")],
            )

    if "TOPK_norm" in locals() and TOPK_norm.size > 0:
        Sorted_indices = np.argsort(TOPK_norm[:, TOPK_names.index("score")])
        TOPK_norm = TOPK_norm[Sorted_indices]

    return (
        MP,
        MP_names,
        MP_trend_raw,
        MP_trend_norm,
        distcalc_num,
        TOPK_norm,
        TOPK_names,
    )


def StringDb_prune(string_id, StringDb, StringDb_names, Prune_param, MagInfo):
    func_name = "StringDb_prune in Spikelet_MP_new_ver_02"

    # Initialize output
    Bool_list = np.ones(len(StringDb), dtype=bool)
    Bool_list[string_id] = False  # Avoid self-comparison

    if not Prune_param or Prune_param.get("prune_method") != "first_spike":
        return Bool_list

    # Calculate index
    SpikeDb = MagInfo["spikeDb"]
    start_spike = StringDb[string_id, StringDb_names.index("start_spike_id")]
    string_length = StringDb[string_id, StringDb_names.index("char_length")]
    end_spike = start_spike + string_length - 1
    spike_id_list = np.arange(start_spike, end_spike + 1)

    # Main pruning
    if Prune_param["prune_method"] == "first_spike":
        spike_id_all = StringDb[:, StringDb_names.index("start_spike_id")]
        spike_id_all_length = len(spike_id_all)
        for query_i in Prune_param["prune_query"]:
            var_split = query_i.split(" ")
            var = var_split[0]
            IV_i_string = MagInfo["mp"]["index"][var]
            type_ = var_split[1]

            if type_ == "range":
                lower = float(var_split[2])
                upper = float(var_split[3])
                Bool_list_ij = (IV_i_string[string_id] * lower <= IV_i_string) & (
                    IV_i_string[string_id] * upper >= IV_i_string
                )

            elif type_ == "percentile":
                percentile = float(var_split[2])
                pos_sorted = np.where(
                    MagInfo["mp"]["index"][f"{var}_sorted_index"] == string_id
                )[0][0]
                lower_pos_sorted = int(
                    np.ceil(
                        spike_id_all_length
                        * (percentile / 100)
                        * (pos_sorted)
                        / spike_id_all_length
                    )
                )
                upper_pos_sorted = int(
                    np.ceil(
                        spike_id_all_length
                        * (percentile / 100)
                        * (spike_id_all_length - pos_sorted)
                        / spike_id_all_length
                    )
                )
                from_sorted = max(0, pos_sorted - lower_pos_sorted)
                to_sorted = min(spike_id_all_length, pos_sorted + upper_pos_sorted)
                Selected_index = MagInfo["mp"]["index"][f"{var}_sorted_index"][
                    from_sorted:to_sorted
                ]
                Bool_list_ij = np.zeros(spike_id_all_length, dtype=bool)
                Bool_list_ij[Selected_index] = True

            else:
                print(f"[{func_name}] unknown type {type_}")
                continue

            Bool_list_ij[string_id] = False  # Remove self-match
            Bool_list = Bool_list & Bool_list_ij

    return Bool_list


def pair_distance(A, B, Distance):
    if Distance == "dtw":
        A_z = (A - np.mean(A)) / np.std(A)
        B_z = (B - np.mean(B)) / np.std(B)
        # dist_raw, _ = fastdtw(A_z, B_z, dist=euclidean)                            das hier noch wegmachen. es hat deswegen einen fehler gegeben und ich war zu faul den weg zu machen
        dist_norm = dist_raw / (np.sqrt(len(A)) * np.sqrt(len(B)))
    return dist_raw, dist_norm


def extractVarInArray(PQ_array):
    VarList = Spikelet_Feature_varlist()
    Array = []
    for string_ij in PQ_array:
        if isinstance(string_ij, str):
            for var in VarList:
                if var in string_ij:
                    Array.append(var)
    return Array


def Spikelet_Feature_varlist():
    return ["SomeVar1", "SomeVar2", "SomeVar3"]  # Example list of variable names
