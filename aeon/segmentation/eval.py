import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from calculations_for_evaluation import (calc_and_print_avg_scores, calc_and_print_avg_times, f1_score, calculate_percentage_difference_clasp_vs_variants, calc_and_print_avg_time_diff, calc_median, calc_average, sum_execution_times_per_dataset, calc_and_print_std_deviation, konfidenzintervall, calc_mse, calc_and_print_median_scores)
from eval_visualization import scatter_ts_length_vs_runtime_multi2, show_boxplot, plot_execution_times, both_boxplots, scatter_num_change_points, plot_downsampling_nth, plot_downsampling_extrema, plot_average_execution_times_nth, plot_average_execution_times_extrema, plot_downsampling_nth_single, plot_downsampling_extrema_single, plot_average_execution_times_nth_single, plot_average_execution_times_extrema_single, scatter_ts_length_vs_runtime_multi, both_boxplots_extended, plot_memory_comparison

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_dir)

all_evaluation_data = {
        "original_time_series_len": [],  # Originale Zeitreihe
        "time_series_name": [],      # Name der Zeitreihe
        "true_change_points": [],      # Positionen der echten Change Points
        "predicted_change_points": {}, # Predicted CPs pro Kombination
        "downsampled_time_series_len": {}, # Gedownsamplete Zeitreihen pro Kombination
        "scores": {},                  # Scores pro Kombination
        "execution_times": {},         # Ausführungszeiten pro Kombination
        "memory_consumption": {},      # Speicherverbrauch pro Kombination
        "parameters": {},
    }


def find_lowest_scores(all_evaluation_data, num, specific_combination=None):
    """
    Find the five lowest scores in all_evaluation_data and return their values
    along with the corresponding time series names.

    Parameters:
    - all_evaluation_data (dict): Dictionary containing evaluation data.

    Returns:
    - list of tuples: A list of tuples with (score, time_series_name) for the five lowest scores.
    """
    scores_with_names = []

    # Iterate through all combinations in scores
    for combination, scores in all_evaluation_data["scores"].items():
        # Skip if specific_combination is given and doesn't match
        if specific_combination and combination != specific_combination:
            continue

        for idx, score in enumerate(scores):
            # Ensure the corresponding time series name exists
            if idx < len(all_evaluation_data["time_series_name"]):
                time_series_name = all_evaluation_data["time_series_name"][idx]
                scores_with_names.append((score, time_series_name, combination))

    # Sort by score (ascending)
    scores_with_names.sort(key=lambda x: x[0])

    # Get the lowest 5 scores
    lowest_5 = scores_with_names[:num]

    # Print and return the results
    for score, name, combination in lowest_5:
        print(f"Score: {score}, Time Series: {name}, Combination: {combination}")

    return lowest_5

def find_highest_scores(all_evaluation_data, num, specific_combination=None):
    """
    Find the five highest scores in all_evaluation_data and return their values
    along with the corresponding time series names.

    Parameters:
    - all_evaluation_data (dict): Dictionary containing evaluation data.
    - specific_combination (str, optional): If provided, only evaluate this combination.

    Returns:
    - list of tuples: A list of tuples with (score, time_series_name, combination) for the five highest scores.
    """
    scores_with_names = []

    # Durch alle Kombinationen iterieren
    for combination, scores in all_evaluation_data["scores"].items():
        # Wenn eine spezifische Kombination angegeben ist, andere ignorieren
        if specific_combination and combination != specific_combination:
            continue

        for idx, score in enumerate(scores):
            # Stelle sicher, dass der Zeitreihenname vorhanden ist
            if idx < len(all_evaluation_data["time_series_name"]):
                time_series_name = all_evaluation_data["time_series_name"][idx]
                scores_with_names.append((score, time_series_name, combination))

    # Sortiere nach Score (absteigend für die höchsten Werte)
    scores_with_names.sort(key=lambda x: x[0], reverse=True)

    # Nimm die 5 höchsten Scores
    highest_5 = scores_with_names[:num]

    # Ergebnisse ausgeben und zurückgeben
    for score, name, combination in highest_5:
        print(f"Score: {score}, Time Series: {name}, Combination: {combination}")

    return highest_5


def convert_to_serializable(obj):
    """
    Konvertiert nicht serialisierbare Objekte (wie Numpy-Arrays) in JSON-kompatible Typen.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Numpy-Array in Liste umwandeln
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

def save_data(output_dir, all_eval_data, file_name="evaluation_results.json"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / file_name
    with file_path.open('w') as json_file:
        json.dump(convert_to_serializable(all_eval_data), json_file, indent=4)

    print(f"Evaluationsdaten wurden in {file_path} gespeichert.")

def load_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data

def print_all_keys(d, parent_key=''):
    """
    Rekursive Funktion, um alle Keys in einem geschachtelten Dictionary zu drucken.

    Parameters:
    - d: dict, das Dictionary
    - parent_key: str, der übergeordnete Key (für verschachtelte Keys)

    Returns:
    - None
    """
    if isinstance(d, dict):
        for key, value in d.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            print(full_key)
            print_all_keys(value, full_key)  # Rekursiver Aufruf
    elif isinstance(d, list):
        for i, item in enumerate(d):
            list_key = f"{parent_key}[{i}]"
            print(list_key)
            print_all_keys(item, list_key)  # Rekursiver Aufruf


def add_evaluation_data(evaluation_data):
    """
    Anstatt ein globales Dictionary zu füllen, bekommt diese Funktion
    jetzt 'all_evaluation_data' als Parameter, in den sie schreibt.
    """
    all_evaluation_data["original_time_series_len"].append(
        evaluation_data["original_time_series_len"]
    )
    all_evaluation_data["time_series_name"].append(
        evaluation_data["time_series_name"]
    )
    all_evaluation_data["true_change_points"].append(
        evaluation_data["true_change_points"]
    )

    lst = [
        "predicted_change_points",
        "downsampled_time_series_len",
        "scores",
        "execution_times",
        "memory_consumption",
        "parameters",
    ]

    for operation in lst:
        for combination_name in evaluation_data[operation]:
            if combination_name not in all_evaluation_data[operation]:
                all_evaluation_data[operation][combination_name] = []
            all_evaluation_data[operation][combination_name].append(
                evaluation_data[operation][combination_name]
            )

def calculate_segmentation_error(predicted_cps, ground_truth_cps, time_series_length):
    """
    Calculate the segmentation error for predicted change points.

    Parameters:
    - predicted_cps: list of predicted change point indices.
    - ground_truth_cps: list of ground truth change point indices.
    - time_series_length: int, length of the time series.

    Returns:
    - float: normalized segmentation error.
    """
    if not predicted_cps or not ground_truth_cps:
        return 1.0  # Maximum error if no change points are predicted or provided.

    error_sum = 0
    for pred_cp in predicted_cps:
        min_distance = min(abs(pred_cp - gt_cp) for gt_cp in ground_truth_cps)
        error_sum += min_distance

    normalized_error = error_sum / (time_series_length * len(predicted_cps))
    return normalized_error

def sort_ts_by_length(data):
    series_lengths = data["original_time_series_len"]
    series_names = data["time_series_name"]  # Entsprechende Namen

    # Kombiniere die Längen und Namen in eine Liste von Tupeln
    combined = list(zip(series_lengths, series_names))

    # Sortiere nach der Länge der Time-Series
    combined_sorted = sorted(combined, key=lambda x: x[0])

    # Entpacke die sortierten Werte zurück in separate Listen
    sorted_lengths, sorted_names = zip(*combined_sorted)

    # Konvertiere die Ergebnisse wieder zu Listen
    sorted_lengths = list(sorted_lengths)
    sorted_names = list(sorted_names)

    data["original_time_series_len"] = sorted_lengths
    data["time_series_name"] = sorted_names

    print("Sortierte Längen:", sorted_lengths)
    print("Sortierte Namen:", sorted_names)

    return data

def sort_ts_by_length(data):
    series_lengths = data["original_time_series_len"]
    series_names = data["time_series_name"]  # Corresponding names

    # Combine lengths, names, and execution times into tuples
    combined = list(zip(series_lengths, series_names))

    # Sort by the length of the time series
    combined_sorted = sorted(combined, key=lambda x: x[0])

    # Unpack sorted values back into separate lists
    sorted_lengths, sorted_names = zip(*combined_sorted)

    # Convert results back to lists
    sorted_lengths = list(sorted_lengths)
    sorted_names = list(sorted_names)

    # Update the data structure
    data["original_time_series_len"] = sorted_lengths
    data["time_series_name"] = sorted_names

    # Sort execution times
    execution_times = data["execution_times"]
    sorted_execution_times = {}
    for combination, times in execution_times.items():
        combined_execution = list(zip(series_lengths, times))
        sorted_execution = sorted(combined_execution, key=lambda x: x[0])
        _, sorted_times = zip(*sorted_execution)
        sorted_execution_times[combination] = list(sorted_times)

    data["execution_times"] = sorted_execution_times

    return data

def runEval(output_dir, all_evaluation_data):
    """
    Schreibt das (lokal gesammelte) all_evaluation_data
    genau ein Mal in eine JSON-Datei.
    """
    save_data(output_dir, all_evaluation_data)

def test():
    file_path = "C:/Users/Victor/Desktop/Uni/Bachelor/output/tssb_spike_nth/evaluation_results.json"
    file_path2 = "C:/Users/Victor/Desktop/Uni/Bachelor/output/tssb_down_nth/evaluation_results.json"

    file_path3 = "C:/Users/Victor/Desktop/Uni/Bachelor/output/tssb_spike_extrema/evaluation_results.json"
    file_path4 = "C:/Users/Victor/Desktop/Uni/Bachelor/output/tssb_down_extrema/evaluation_results.json"

    file_path5 = "C:/Users/Victor/Desktop/Uni/Bachelor/output/floss_spike_nth/evaluation_results.json"
    file_path6 = "C:/Users/Victor/Desktop/Uni/Bachelor/output/floss_down_nth/evaluation_results.json"

    file_path7 = "C:/Users/Victor/Desktop/Uni/Bachelor/output/floss_spike_extrema/evaluation_results.json"
    file_path8 = "C:/Users/Victor/Desktop/Uni/Bachelor/output/floss_down_extrema/evaluation_results.json"

    file_path9 = "C:/Users/Victor/Desktop/Uni/Bachelor/output/Run_20250223_160954/evaluation_results.json"
    file_path10 = "C:/Users/Victor/Desktop/Uni/Bachelor/output/Run_20250223_163211/evaluation_results.json"

    file_path11 = "C:/Users/Victor/Desktop/Uni/Bachelor/output/tssb_mews_nth4/evaluation_results.json"
    file_path12 = "C:/Users/Victor/Desktop/Uni/Bachelor/output/floss_mews_ext13/evaluation_results.json"

    file_path_downsampled_data = "C:/Users/Victor/Desktop/Uni/Bachelor/output/Run_20250127_214814/downsampled_data.json"
    file_path_spikelet_data = "C:/Users/Victor/Desktop/Uni/Bachelor/output/Run_20250127_214814/spikelet_downsampled_data.json"


    data = load_data(file_path)
    data2 = load_data(file_path2)

    data3 = load_data(file_path3)
    data4 = load_data(file_path4)

    data5 = load_data(file_path5)
    data6 = load_data(file_path6)

    data7 = load_data(file_path7)
    data8 = load_data(file_path8)

    data9 = load_data(file_path9)
    data10 = load_data(file_path10)

    data11 = load_data(file_path11)
    data12 = load_data(file_path12)

    data_downsampled = load_data(file_path_downsampled_data)
    data_spikelet = load_data(file_path_spikelet_data)

    data = sort_ts_by_length(data)
    data2 = sort_ts_by_length(data2)
    data3 = sort_ts_by_length(data3)
    data4 = sort_ts_by_length(data4)
    data5 = sort_ts_by_length(data5)
    data6 = sort_ts_by_length(data6)
    data7 = sort_ts_by_length(data7)
    data8 = sort_ts_by_length(data8)
    data9 = sort_ts_by_length(data9)
    data10 = sort_ts_by_length(data10)
    data11 = sort_ts_by_length(data11)
    data12 = sort_ts_by_length(data12)

    try:
        # Diagramme für den ersten Vergleich - kein Spikelet
        #######################################################################
        # plot_downsampling_nth_single(data, data5)
        # plot_downsampling_extrema_single(data3, data7)

        # plot_average_execution_times_nth_single(data, data5)
        # plot_average_execution_times_extrema_single(data3, data7)

        setting_to_compare_for_boxplot_tssb = [
           "[tssb, spikelet_first, mews=True] Spikelet=False, Downsampling=None",
           "[tssb, spikelet_first, mews=True] Spikelet=False, Downsampling=nth, Param=2",
           "[tssb, spikelet_first, mews=True] Spikelet=False, Downsampling=nth, Param=6",
           "[tssb, spikelet_first, mews=True] Spikelet=False, Downsampling=Extrema, Param=3" 
        ]
        setting_to_compare_for_boxplot_floss = [
           "[floss, spikelet_first, mews=True] Spikelet=False, Downsampling=None",
           "[floss, spikelet_first, mews=True] Spikelet=False, Downsampling=nth2, Param=5",
           "[floss, spikelet_first, mews=True] Spikelet=False, Downsampling=nth2, Param=7",
           "[floss, spikelet_first, mews=True] Spikelet=False, Downsampling=Extrema, Param=11" 
        ]
        # both_boxplots(data["scores"][setting_to_compare_for_boxplot_tssb[0]], data["scores"][setting_to_compare_for_boxplot_tssb[1]], data["scores"][setting_to_compare_for_boxplot_tssb[2]], data3["scores"][setting_to_compare_for_boxplot_tssb[3]],
        #                data5["scores"][setting_to_compare_for_boxplot_floss[0]], data5["scores"][setting_to_compare_for_boxplot_floss[1]], data5["scores"][setting_to_compare_for_boxplot_floss[2]], data7["scores"][setting_to_compare_for_boxplot_floss[3]]) 
                    
        curves_names = ['ClaSP', 'DEL: n=2', 'DEL: n=6', 'M2: w_e=3']
        curves_names2 = ['ClaSP', 'MEAN: n=5', 'MEAN: n=7', 'M2: w_e=11']
        # plot_execution_times(data["time_series_name"], data["execution_times"][setting_to_compare_for_boxplot_tssb[0]], data["execution_times"][setting_to_compare_for_boxplot_tssb[1]], data["execution_times"][setting_to_compare_for_boxplot_tssb[2]], data3["execution_times"][setting_to_compare_for_boxplot_tssb[3]], setting_to_compare_for_boxplot_tssb, curves_names)
        # plot_execution_times(data5["time_series_name"], data5["execution_times"][setting_to_compare_for_boxplot_floss[0]], data5["execution_times"][setting_to_compare_for_boxplot_floss[1]], data5["execution_times"][setting_to_compare_for_boxplot_floss[2]], data7["execution_times"][setting_to_compare_for_boxplot_floss[3]], setting_to_compare_for_boxplot_floss, curves_names2)
        
        # calc_and_print_avg_scores(data10)
        # print("\n\n")
        # calc_and_print_avg_times(data10)
        # print("\n\n")
        # calc_and_print_median_scores(data)
        # print("\n\n")
        #######################################################################


        # Diagramme für den zweiten Vergleich - mit Spikelet
        #######################################################################
        # plot_downsampling_nth(data, data5, data2, data6)
        # plot_downsampling_extrema(data3, data7, data4, data8)

        # plot_average_execution_times_nth(data, data2, data5, data6)
        # plot_average_execution_times_extrema(data3, data4, data7, data8)

        setting_to_compare_for_boxplot_tssb = [
           "[tssb, spikelet_first, mews=True] Spikelet=False, Downsampling=None",
           "[tssb, spikelet_first, mews=True] Spikelet=True, Downsampling=nth, Param=4",
           "[tssb, spikelet_first, mews=True] Spikelet=True, Downsampling=nth, Param=15",
           "[tssb, downsampling_first, mews=True] Spikelet=True, Downsampling=Extrema2, Param=4" 
        ]
        setting_to_compare_for_boxplot_floss = [
           "[floss, spikelet_first, mews=True] Spikelet=False, Downsampling=None",
           "[floss, spikelet_first, mews=True] Spikelet=True, Downsampling=nth2, Param=5",
           "[floss, spikelet_first, mews=True] Spikelet=True, Downsampling=Extrema, Param=9",
           "[floss, spikelet_first, mews=True] Spikelet=True, Downsampling=Extrema2, Param=13" 
        ]

        # both_boxplots(data["scores"][setting_to_compare_for_boxplot_tssb[0]], data["scores"][setting_to_compare_for_boxplot_tssb[1]], data["scores"][setting_to_compare_for_boxplot_tssb[2]], data4["scores"][setting_to_compare_for_boxplot_tssb[3]],
        #                data5["scores"][setting_to_compare_for_boxplot_floss[0]], data5["scores"][setting_to_compare_for_boxplot_floss[1]], data7["scores"][setting_to_compare_for_boxplot_floss[2]], data7["scores"][setting_to_compare_for_boxplot_floss[3]])

        # calc_and_print_avg_scores(data2)
        # print("\n\n")
        # calc_and_print_avg_times(data2)
        # print("\n\n")
        # calc_and_print_median_scores(data2)
        # print("\n\n")
        #######################################################################


        # Diagramme für den dritten Vergleich - Manually determining MaT and CoT
        #######################################################################

        setting_to_compare_for_boxplot_tssb = [
           "[tssb, spikelet_first, mews=True] Spikelet=False, Downsampling=None",
           "[tssb, spikelet_first, mews=True] Spikelet=False, Downsampling=nth, Param=6",
           "[tssb, spikelet_first, mews=True] Spikelet=False, Downsampling=Extrema, Param=3",
           "[tssb, spikelet_first, mews=True] Spikelet=True, Downsampling=nth, Param=4",
           "[tssb, downsampling_first, mews=True] Spikelet=True, Downsampling=Extrema2, Param=4",
           "[tssb, spikelet_first, mews=False] Spikelet=True, Downsampling=nth, Param=4" 
        ]
        setting_to_compare_for_boxplot_floss = [
           "[floss, spikelet_first, mews=True] Spikelet=False, Downsampling=None",
           "[floss, spikelet_first, mews=True] Spikelet=False, Downsampling=nth2, Param=5",
           "[floss, spikelet_first, mews=True] Spikelet=False, Downsampling=Extrema2, Param=16",
           "[floss, spikelet_first, mews=True] Spikelet=True, Downsampling=nth2, Param=5",
           "[floss, spikelet_first, mews=True] Spikelet=True, Downsampling=Extrema2, Param=13",
           "[floss, spikelet_first, mews=False] Spikelet=True, Downsampling=Extrema2, Param=13" 
        ]

        both_boxplots_extended(data["scores"][setting_to_compare_for_boxplot_tssb[0]], data["scores"][setting_to_compare_for_boxplot_tssb[1]], data3["scores"][setting_to_compare_for_boxplot_tssb[2]], data["scores"][setting_to_compare_for_boxplot_tssb[3]], data4["scores"][setting_to_compare_for_boxplot_tssb[4]], data9["scores"][setting_to_compare_for_boxplot_tssb[5]],
                       data5["scores"][setting_to_compare_for_boxplot_floss[0]], data5["scores"][setting_to_compare_for_boxplot_floss[1]], data7["scores"][setting_to_compare_for_boxplot_floss[2]], data5["scores"][setting_to_compare_for_boxplot_floss[3]], data7["scores"][setting_to_compare_for_boxplot_floss[4]], data10["scores"][setting_to_compare_for_boxplot_floss[5]])
                        
        curves_names = ["ClaSP", "n=6", "w_e=3", "n=4 (opt.)"]
        curves_names2 = ["ClaSP", "n=5", "w_e=11", "w_e=13 (opt.)"]
        # plot_execution_times(data["time_series_name"], data["execution_times"][setting_to_compare_for_boxplot_tssb[0]], data["execution_times"][setting_to_compare_for_boxplot_tssb[1]], data3["execution_times"][setting_to_compare_for_boxplot_tssb[2]], data9["execution_times"][setting_to_compare_for_boxplot_tssb[5]], setting_to_compare_for_boxplot_tssb, curves_names)
        # plot_execution_times(data5["time_series_name"], data5["execution_times"][setting_to_compare_for_boxplot_floss[0]], data5["execution_times"][setting_to_compare_for_boxplot_floss[1]], data7["execution_times"][setting_to_compare_for_boxplot_floss[2]], data10["execution_times"][setting_to_compare_for_boxplot_floss[5]], setting_to_compare_for_boxplot_floss, curves_names2)
        
        # scatter_ts_length_vs_runtime_multi2(
        #     data11, data11["execution_times"]["[tssb, spikelet_first, mews=True] Spikelet=True, Downsampling=nth, Param=4"], "MeWS: n=4",
        #     data, data["execution_times"]["[tssb, spikelet_first, mews=True] Spikelet=False, Downsampling=nth, Param=5"], "DEL: n=5",
        #     data9, data9["execution_times"]["[tssb, spikelet_first, mews=False] Spikelet=True, Downsampling=nth, Param=4"], "Opt, DEL: n=4",
        #     data, data["execution_times"]["[tssb, spikelet_first, mews=True] Spikelet=False, Downsampling=None"], "ClaSP",
        #     data12, data12["execution_times"]["[floss, spikelet_first, mews=True] Spikelet=True, Downsampling=Extrema2, Param=13"], "MeWS, M3: w_e=13",
        #     data7, data7["execution_times"]["[floss, spikelet_first, mews=True] Spikelet=False, Downsampling=Extrema2, Param=11"], "M3: w_e=13",
        #     data10, data10["execution_times"]["[floss, spikelet_first, mews=False] Spikelet=True, Downsampling=Extrema2, Param=13"], "Opt, M3: w_e=13",
        #     data5, data5["execution_times"]["[floss, spikelet_first, mews=True] Spikelet=False, Downsampling=None"], "ClaSP",
        # )

        # plot_memory_comparison(
        #     r"C:\Users\Victor\Desktop\Uni\Bachelor\output\memory_usage_tssb_mews.csv",
        #     r"C:\Users\Victor\Desktop\Uni\Bachelor\output\memory_usage_tssb_none.csv",
        #     r"C:\Users\Victor\Desktop\Uni\Bachelor\output\memory_usage_floss_mews.csv",
        #     r"C:\Users\Victor\Desktop\Uni\Bachelor\output\memory_usage_floss_none.csv"
        # )
        #######################################################################

        # scatter_num_change_points(data2, data2["scores"]["[floss, spikelet_first, mews=True] Spikelet=False, Downsampling=Extrema, Param=3"])
        
        # setting_to_compare_for_boxplot = [
        #    "[tssb, spikelet_first, mews=True] Spikelet=False, Downsampling=Extrema, Param=3",
        #    "[floss, spikelet_first, mews=True] Spikelet=False, Downsampling=Extrema, Param=3",
        #    "[tssb, spikelet_first, mews=True] Spikelet=False, Downsampling=Extrema, Param=3",
        #    "[floss, spikelet_first, mews=True] Spikelet=False, Downsampling=Extrema, Param=3" 
        # ]

        # show_boxplot(data["scores"][setting_to_compare_for_boxplot[0]], data2["scores"][setting_to_compare_for_boxplot[1]], data["scores"][setting_to_compare_for_boxplot[2]], data2["scores"][setting_to_compare_for_boxplot[3]])

        # both_boxplots(data["scores"][setting_to_compare_for_boxplot[0]], data["scores"][setting_to_compare_for_boxplot[2]], data["scores"][setting_to_compare_for_boxplot[0]], data["scores"][setting_to_compare_for_boxplot[2]],
        #                data2["scores"][setting_to_compare_for_boxplot[1]], data2["scores"][setting_to_compare_for_boxplot[3]], data2["scores"][setting_to_compare_for_boxplot[1]], data2["scores"][setting_to_compare_for_boxplot[3]])

        # calc_and_print_avg_scores(data2)
        # print("\n\n")
        # calc_and_print_avg_times(data2)
        # print("\n\n")
        # find_lowest_scores(data, 10,  "[tssb, spikelet_first, mews=True] Spikelet=False, Downsampling=Extrema, Param=3")
        # print("\n\n")
        # find_highest_scores(data, 10, "[tssb, spikelet_first, mews=True] Spikelet=False, Downsampling=Extrema, Param=3")

        # plot_execution_times(data["time_series_name"], data2, data2, data4, data4, key_strings, curve_labels)

        #result = calculate_percentage_difference_clasp_vs_variants(data)
        #for key in result:
        #    print(f"{key}: {result[key]}")
        #calc_and_print_avg_time_diff(result)
        #calc_and_print_std_deviation(data["scores"]["Spikelet=False, Downsampling=None"], data["original_time_series_len"])

        #std_dev = calc_and_print_std_deviation(data["scores"]["Spikelet=False, Downsampling=None"], data["original_time_series_len"])
        #konfidenzintervall(avg_time, std_dev*avg_time)

        #val_nth, val_ext = calc_mse(data_downsampled, data_spikelet)
        #print(f"val nth: {val_nth}, val extrema: {val_ext}")

    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    test()


############################################### LITERALLY DIE GODRUNS ###############################################
# Run_20250223_160954 - optimale MaT und CoT werte run mit nth n = 4 auf alle TSSB datensätze
# Run_20250221_114401 - optimale MaT und CoT werte run mit extrema w_e = 3 auf alle TSSB datensätze -> stimmt nicht
# Run_20250221_135532, Run_20250223_163211 - optimale MaT und CoT werte run mit extrema2 w_e = 13 auf alle FLOSS datensätze

# Run_20250205_080311 - ohne MeWS, spikelet vor downsampling alle tssb datensätze. (gruenau aber is eig egal)
# Run_20250209_190105 - ohne MeWS, spikelet vor downsampling alle floss datensätze. (gruenau aber is eig egal)

# Run_20250215_001141 - mit MeWS, spikelet vor downsampling alle tssb datensätze.
### Run_20250215_001141 - Kopie - mit MeWS, spikelet vor downsampling alle tssb datensätze. oder Run_20250218_205853
# Run_20250215_060314 - mit MeWS, downsampling vor spikelet alle tssb datensätze.
### Run_20250215_060314 - Kopie - mit MeWS, downsampling vor spikelet alle tssb datensätze. oder Run_20250219_062713
# Run_20250214_194137 - mit MeWS, spikelet vor downsampling alle floss datensätze.
### Run_20250214_194137 - Kopie - mit MeWS, spikelet vor downsampling alle floss datensätze.
# Run_20250215_215015 - mit MeWS, downsampling vor spikelet alle floss datensätze.
### Run_20250215_215015 - Kopie - mit MeWS, downsampling vor spikelet alle floss datensätze.

# Run_20250208_173320 - mit MeWS, spikelet vor downsampling alle floss datensätze. (von gruenau)
# Run_20250209_070417 - mit MeWS, downsampling vor spikelet alle floss datensätze. (von gruenau)

# Run_20250216_112902 - testrun, nur extrema 8-12 um zu gucken wie sich das verhält

############################################ RUNS MIT SPIKELET VOR DOWNSAMPLING ############################################

# Run_20250115_181427 - nth ist 3 und extrema window size ist vollständig w
# Run_20250116_151615 - nth ist 4 und extrema window size ist halbiert
# Run_20250116_160330 - nth ist 5 und extrema window size ist gedrittelt
# Run_20250116_164144 - nth ist 2 und extrema window size ist * 1.5
# Run_20250116_172003 - nth ist 2 und extrema window size ist halbiert
# Run_20250116_184821 - nth ist 2 und extrema window size ist * 0.75
# Run_20250116_191717 - nth ist 2 und extrema window size ist * 1.25
# Run_20250117_131016 - nth ist 2/3 und extrema window size ist * 0.4
# Run_20250117_155533 - nth ist 2/3 und extrema window size ist * 0.35
# Run_20250118_125509 - nth ist 4/5 und extrema wurde neue technik ausprobiert: aber window size ist * 0.3
# Run_20250118_135339 - nth ist 4 und extrema wurde neue technik ausprobiert: aber window size ist * 0.4 - neue technik ist lowkey ass
# Run_20250118_155836 - nth ist 4 und extrema window size ist * 0.45
# Run_20250122_161750 - nth ist 7 und extrema window size ist 10.
# Run_20250122_164809 - nth ist 9 und extrema window size ist 5.
# Run_20250122_173909 - nth ist 11 und extrema window size ist 4
# Run_20250122_183605 - nth ist 15 und extrema window size ist 3
# Run_20250122_192106 - nth ist 10 und extrema window size ist 3 (mean() wurde rausgenommen also wurde die downsampled_ts um 2 immer verlängert)
# Run_20250123_113006 - nth ist 12 und extrema window size ist 4 (mean() wurde rausgenommen also wurde die downsampled_ts um 2 immer verlängert)
# Run_20250123_130840 - nth ist 13 und extrema window size ist 3 (mean() wurde rausgenommen also wurde die downsampled_ts um 2 immer verlängert)
# Run_20250123_133848 - nth ist 8 und extrema window size ist 3 (mean() wurde rausgenommen also wurde die downsampled_ts um 2 immer verlängert)
# Run_20250126_194106 - nth ist 4 und extrema window size ist 3 (mean() wurde rausgenommen also wurde die downsampled_ts um 2 immer verlängert)
# Run_20250126_203727 - nth ist 5 und extrema window size ist 3 (mean() wurde rausgenommen also wurde die downsampled_ts um 2 immer verlängert)


############################################ RUNS MIT SPIKELET NACH DOWNSAMPLING ############################################

# Run_20250122_123617 - nth ist 4 und extrema window size ist * 0.45. 
# Run_20250122_132500 - nth ist 2 und extrema window size ist * 0.35. 
# Run_20250122_140120 - nth ist 3 und extrema window size ist * 0.5. 
# Run_20250122_144047 - nth ist 5 und extrema window size ist * 0.75. 
# Run_20250122_154708 - nth ist 7 und extrema window size ist 10. 
# Run_20250127_140105 - nth ist 8 und extrema window size ist 3 (mean() wurde rausgenommen also wurde die downsampled_ts um 2 immer verlängert)
# Run_20250127_143407 - nth ist 10 und extrema window size ist 4 (mean() wurde rausgenommen also wurde die downsampled_ts um 2 immer verlängert)
# Run_20250127_151224 - nth ist 12 und extrema window size ist 4
# Run_20250127_155430 - nth ist 15 und extrema window size ist 5
# Run_20250127_163423 - nth ist 6 und extrema window size ist 3 (mean() wurde rausgenommen also wurde die downsampled_ts um 2 immer verlängert)
# Run_20250127_170929 - nth ist 11 und extrema window size ist 4 (mean() wurde rausgenommen also wurde die downsampled_ts um 2 immer verlängert)
# Run_20250127_214814 - nth ist 4 und extrema window size ist 3 (mean() wurde rausgenommen also wurde die downsampled_ts um 2 immer verlängert) - ts_down + spikelet_ts wurde gespeichert
# Run_20250127_224039 - nth ist 5 und extrema window size ist 4 (mean() wurde rausgenommen also wurde die downsampled_ts um 2 immer verlängert) - ts_down + spikelet_ts wurde gespeichert
# Run_20250127_232640 - nth ist 8 und extrema window size ist 4 - ts_down + spikelet_ts wurde gespeichert
# Run_20250128_001655 - nth ist 10 und extrema window size ist 5 - ts_down + spikelet_ts wurde gespeichert



############################################ RUNS MIT ITERIERENDE MAT ############################################

# Run_20250123_172440 - tiefgründiger Spikelet run auf "Crop"
# Run_20250126_110710 - tiefgründiger Spikelet run auf "NonInvasiveFetalECGThorax1"
# Run_20250126_154020 - tiefgründiger Spikelet run auf "FaceFour"
# FacesUCR hat verbesserungen gezeigt aber ist abgestürzt
