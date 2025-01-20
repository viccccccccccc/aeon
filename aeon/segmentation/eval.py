import os
import sys
import json
import numpy as np
from datetime import datetime

from calculations_for_evaluation import (calc_and_print_avg_scores, calc_and_print_avg_times, f1_score, calculate_percentage_difference_clasp_vs_variants, calc_and_print_avg_time_diff, calc_median, calc_average, sum_execution_times_per_dataset)
from eval_visualization import visualize_times_bars, show_hist, show_boxplot, plot_execution_times

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

def find_lowest_scores(all_evaluation_data, specific_combination=None):
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
    lowest_5 = scores_with_names[:5]

    # Print and return the results
    for score, name, combination in lowest_5:
        print(f"Score: {score}, Time Series: {name}, Combination: {combination}")

    return lowest_5

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

def save_data(output_dir, all_eval_data,  file_name="evaluation_results.json"):
    """
    Speichert alle gesammelten Evaluationsdaten als JSON-Datei.

    Parameters
    ----------
    output_dir : str
        Verzeichnis, in dem die JSON-Datei gespeichert wird.
    file_name : str, optional
        Name der JSON-Datei (default: "evaluation_results.json").

    Returns
    -------
    None
    """
    local_output_dir = output_dir
    if not os.path.exists(local_output_dir):
        os.makedirs(local_output_dir)

    all_eval_data = convert_to_serializable(all_eval_data)
    file_path = os.path.join(local_output_dir, file_name)
    with open(file_path, "w") as json_file:
        json.dump(all_eval_data, json_file, indent=4)

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


def add_evaluation_data(evaluation_data):###
    """
    Fügt die Evaluationsdaten eines Datensatzes dem globalen Dictionary hinzu.

    Parameters
    ----------
    ts_name : str
        Name des Datensatzes.
    evaluation_data : dict
        Evaluationsdaten für den aktuellen Datensatz.

    Returns
    -------
    None
    """
    all_evaluation_data["original_time_series_len"].append(evaluation_data["original_time_series_len"])
    all_evaluation_data["time_series_name"].append(evaluation_data["time_series_name"])
    all_evaluation_data["true_change_points"].append(evaluation_data["true_change_points"])

    lst = [
        "predicted_change_points",
        "downsampled_time_series_len",
        "scores",
        "execution_times",
        "memory_consumption",
        "parameters"
    ]

    for operation in lst:
        for combination_name in evaluation_data[operation]:
            if combination_name not in all_evaluation_data[operation]:
                all_evaluation_data[operation][combination_name] = []
            all_evaluation_data[operation][combination_name].append(evaluation_data[operation][combination_name])

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

def runEval(output_dir):
    print("\n\n\n")
    save_data(output_dir, all_evaluation_data)

def test():
    file_path = "C:/Users/Victor/Desktop/Uni/Bachelor/output/Run_20250116_164144/evaluation_results.json"
    file_path2 = "C:/Users/Victor/Desktop/Uni/Bachelor/output/Run_20250115_181427/evaluation_results.json"
    file_path3 = "C:/Users/Victor/Desktop/Uni/Bachelor/output/Run_20250116_151615/evaluation_results.json"
    file_path4 = "C:/Users/Victor/Desktop/Uni/Bachelor/output/Run_20250116_160330/evaluation_results.json"

    data = load_data(file_path)
    data2 = load_data(file_path2)
    data3 = load_data(file_path3)
    data4 = load_data(file_path4)

    #data = sort_ts_by_length(data)
    #data2 = sort_ts_by_length(data2)
    #data3 = sort_ts_by_length(data3)
    #data4 = sort_ts_by_length(data4)

    data = sort_ts_by_length(data)
    data2 = sort_ts_by_length(data2)
    data3 = sort_ts_by_length(data3)
    data4 = sort_ts_by_length(data4)

    #results = f1_score(data)
    #results2 = f1_score(data2)
    #results3 = f1_score(data3)
    #results4 = f1_score(data4)

    try:
        #show_hist(data["scores"]["Spikelet=False, Downsampling=nth"])
        #visualize_times_bars(data)
        #find_lowest_scores(data, "Spikelet=False, Downsampling=Extrema")
        
        show_boxplot(data["scores"]["Spikelet=True, Downsampling=nth"], data2["scores"]["Spikelet=True, Downsampling=nth"], data3["scores"]["Spikelet=True, Downsampling=nth"], data4["scores"]["Spikelet=True, Downsampling=nth"])
        
        results = f1_score(data)
        results2 = f1_score(data2)
        results3 = f1_score(data3)
        results4 = f1_score(data4)
        show_boxplot(results["Spikelet=True, Downsampling=nth"], results2["Spikelet=True, Downsampling=nth"], results3["Spikelet=True, Downsampling=nth"], results4["Spikelet=True, Downsampling=nth"])
        
        plot_execution_times(data["time_series_name"], data, data2, data3, data4)
        result = calculate_percentage_difference_clasp_vs_variants(data)
        for key in result:
            print(f"{key}: {result[key]}")
        #calc_and_print_avg_time_diff(result)
        calc_and_print_avg_scores(data4)
        #calc_and_print_avg_times(data)

    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    test()


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


