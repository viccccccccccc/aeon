def Spikelet_MpParam_generate_ver_02(user):
    # Function name for logging
    func_name = "Spikelet_MpParam_generate"

    # Determine support max and min values
    supp_max = getattr(user, "supp_max", float("inf"))
    supp_min = getattr(user, "supp_min", 3)

    # Initialize default parameters
    param = default_param(supp_min, supp_max)
    if not user:
        return param

    # Symbol mapping rule
    if hasattr(user, "symbol_mapping_rule"):
        param["spikeDb"]["symbol_mapping"]["rule"] = user.symbol_mapping_rule

    # Symbol mapping details
    if hasattr(user, "symbol_mapping"):
        for field in user.symbol_mapping:
            param["spikeDb"]["symbol_mapping"][field] = user.symbol_mapping[field]

    param["operation"]["extractConstantSegment"]["auto"]["knee_find"]["sign"] = "nonzero"

    param["operation"]["restrictSupportByWindowLength"] = {"window_band": [supp_max, supp_max]}

    if hasattr(user, "symbol_mapping_argument"):
        param["spikeDb"]["symbol_mapping"]["argument"] = user.symbol_mapping_argument

    # Query setup
    if hasattr(user, "query"):
        q_cell = user.query
        query = []

        for pattern in q_cell:
            q = {
                "pattern": pattern,
                "constraint": "",
                "color": "red",
                "supp_max": supp_max,
                "supp_min": supp_min,
                "terminal_left": "notincluded" if pattern[0] == "C" else "mag_100",
                "terminal_right": "notincluded" if pattern[-1] == "C" else "mag_100",
                "matrix_profile": {"distance": "dtw"},
                "terminal_constant_both": {
                    "on": True,
                    "constant_val": "middle",  # Options: 'shallow', 'deep', etc.
                },
            }
            query.append(q)

        param["query"] = query

    # Operation sequence setup
    op_c = ["reduceSpikeByMagnitude", "extractConstantSegment"]
    op_nc = [
        "reduceSpikeByMagnitude",
        "restrictSupportByWindowLength",
        "restrictSupportByMagnitudeRatio",
    ]

    if hasattr(user, "query"):
        param["operation"]["operation_sequence"] = [
            "restrictSupportByWindowLength",
            "reduceSpikeByMagnitude",
            "restrictSupportByMagnitudeRatioInitial",
            "extractConstantSegment",
        ]
        param["operation"]["name"] = "custom_sequence"
        # if contains_in_cell(user.query, "C"):
        #     param["operation"]["operation_sequence"] = op_c
        #     param["operation"]["name"] = "standard_C"
        # else:
        #     param["operation"]["operation_sequence"] = op_nc
        #     param["operation"]["name"] = "standard_NC"

    field_1, field_2 = "operation", "operation_sequence"
    if hasattr(user, field_1) and field_2 in getattr(user, field_1):
        param[field_1][field_2] = getattr(user, field_1)[field_2]
        param[field_1]["name"] = "user_defined"

    field_1, field_2 = "operation", "restrictSupportByMagnitudeRatio"
    if hasattr(user, field_1) and field_2 in getattr(user, field_1):
        for field in getattr(user, field_1)[field_2]:
            param[field_1][field_2][field] = getattr(user, field_1)[field_2][field]

    # Plot parameters
    param["plot_param"] = Spikelet_SD_createTrendPlotParam(
        param["operation"]["operation_sequence"]
    )

    # Magnitude threshold
    if hasattr(user, "magnitude_threshold_auto"):
        param["operation"]["reduceSpikeByMagnitude"][
            "method"
        ] = user.magnitude_threshold_auto

    if hasattr(user, "magnitude_threshold"):
        mag_thr = user.magnitude_threshold
        band_query = f"abs(Mag) >= {mag_thr}"

        param["operation"]["reduceSpikeByMagnitude"]["method"] = "manual"
        param["operation"]["reduceSpikeByMagnitude"]["manual"] = {
            "query": band_query,
            "magnitude_threshold": mag_thr,
        }

        print(f"[{func_name}] Magnitude threshold ({mag_thr}) is defined by user")
    else:
        param["operation"]["reduceSpikeByMagnitude"]["method"] = "auto"

    # Constant-length threshold
    if hasattr(user, "constant_length_threshold"):
        cl_thr = user.constant_length_threshold

        param["operation"]["extractConstantSegment"]["lengthThreshold"] = {
            "method": "manual",
            "manual": {"CLThr": cl_thr},
        }
        print(f"[{func_name}] Constant Length threshold ({cl_thr}) is defined by user")
    else:
        param["operation"]["extractConstantSegment"]["lengthThreshold"] = {
            "method": "auto"
        }

    # Restrict support by window length
    if hasattr(user, "supp_max"):
        param["operation"]["reduceSpikeByMagnitude"]["auto"]["op"][
            "restrictSupportByWindowLength"
        ]["window_band"] = [user.supp_max, user.supp_max]

    return param


def contains_in_cell(query, ptr):
    """Check if any of the query patterns contain a given pointer (substring)."""
    for q in query:
        if ptr in q:
            return True
    return False


def default_param(supp_min, supp_max):
    """Define the default parameters for the Spikelet algorithm."""
    # Default operation sequence and parameters
    op = {
        "operation_sequence": [
            "reduceSpikeByMagnitude",
            "restrictSupportByMagnitudeRatio",
            "extractConstantSegment",
            "reduceSpikeByMagnitudeRatio",
        ],
        "name": "standard_C",
        "reduceSpikeByMagnitude": {
            "auto": {
                "method": "SuppMax-SuppMRatio",
                "supp_max": supp_max,
                "supp_min": supp_min,
                "knee_find": {
                    "sign": "nonzero",
                    "type": "2nd_zerocross",
                    "approx_function": ["poly_1_poly_0", "poly_1_poly_0"],
                },
                "op": {
                    "operation_sequence": [
                        "restrictSupportByWindowLength",
                        "restrictSupportByMagnitudeRatio",
                    ],
                    "restrictSupportByMagnitudeRatio": {"magnitude_ratio": 0.5},
                    "restrictSupportByWindowLength": {
                        "window_band": [supp_max, supp_max]
                    },
                },
            }
        },
        "extractConstantSegment": {
            "auto": {
                "constant_separated": {"method": "on"},
                "knee_find": {
                    "type": "2nd_zerocross",
                    "approx_function": ["poly_1_poly_0", "poly_1_poly_0"],
                },
                "supp_max": supp_max,
                "supp_min": supp_min,
            },
            "legRatio_constant": 0.5,
            "legRatio_twoSided": 0.25,
            "legRatio_oneSided": 0.25,
            "slopeRatio_twoSided": 10,
            "magRatio_twoSided": 0.5,
            "magRatio_oneSided": 1,
        },
        "restrictSupportByMagnitudeRatio": {"magnitude_ratio": 0.5},
        "restrictSupportByMagnitudeRatioInitial": {"magnitude_ratio": 0.5},
        "reduceSpikeByMagnitudeRatio": {"magnitude_ratio": 0.5},
    }

    # Default symbol mapping
    symbol_mapping_rule = [
        {"condition": "Type == 2", "symbol": "A"},
        {"condition": "Type == -2", "symbol": "a"},
        {"condition": "Type == 1", "symbol": "L"},
        {"condition": "Type == -1", "symbol": "l"},
        {"condition": "Type == 0", "symbol": "C"},
    ]

    spike_db_param = {
        "symbol_mapping": {
            "rule": symbol_mapping_rule,
            "supp_max": supp_max,
            "supp_min": supp_min,
        },
        "index": [],
        "reduction_query": "",
    }

    # Default query parameters
    query = [
        {
            "pattern": "A",
            "constraint": "",
            "color": "red",
            "terminal_left": "mag_100",
            "terminal_right": "mag_100",
            "matrix_profile": {"distance": "dtw"},
        }
    ]

    # Default parameter structure
    param = {
        "name": "default",
        "dataname": "not assigned",
        "operation": op,
        "spikeDb": spike_db_param,
        "query": query,
        "matrix_profile": {"distance": "dtw"},
        "plot_param": Spikelet_SD_createTrendPlotParam(op["operation_sequence"]),
    }

    return param


def Spikelet_SD_createTrendPlotParam(op_seq):
    """Create trend plot parameters based on the operation sequence."""
    op1 = "generateInitialSpikelet"

    # Initialize trend time series, magnitude, and decomposition
    tt = (
        ["org_tm", f"{op1}_tm"]
        + [f"{op}_tm" for op in op_seq]
        + ["_symbol", "org_tmMotif"]
    )
    tm = ["org_mag", f"{op1}_mag"] + [f"{op}_mag" for op in op_seq]
    td = ["org_tm", f"{op1}_dc"] + [f"{op}_dc" for op in op_seq]

    return {"trend_timeseries": tt, "trend_magnitude": tm, "trend_decomposition": td}

