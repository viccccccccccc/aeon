import numpy as np
import math
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel

def t_test(times1,  times2):
    t_stat, p_value = ttest_rel(times1, times2)
    print("Wenn der Wert unter 0.05 ist, gibt es einen signifikanten unterschied")
    print(f"T-Statistik: {t_stat}, p-Wert: {p_value}")

def wilcoxon_test(times1, times2):
    stat, p_value = wilcoxon(times1, times2)
    print("Wenn der Wert unter 0.05 ist, gibt es einen signifikanten unterschied")
    print(f"Wilcoxon-Statistik: {stat}, p-Wert: {p_value}")

def calc_average(vals):
    arr = np.array(vals)
    return np.sum(arr)/len(arr)

def calc_and_print_avg_scores(data):
    if "scores" in data:
            for comb in data["scores"]:
                averages = calc_average(data["scores"][comb])
                print(f"Score average von {comb}: {averages}")
    else:
        print("Scores is not in the dictionary")

def calc_and_print_avg_time_diff(data):
    for comb in data:
        averages = calc_average(data[comb])
        print(f"Time average von {comb}: {averages}")

def calc_and_print_avg_times(data):
    if "execution_times" in data:
            for comb in data["execution_times"]:
                spikelet_sum = 0
                clasp_sum = 0
                downsampling_sum = 0

                execution_list = data["execution_times"][comb]

                for entry in execution_list:
                    spikelet_sum += entry.get("Spikelet", 0)
                    clasp_sum += entry.get("ClaSP", 0)
                    downsampling_sum += entry.get("Downsampling", 0)

                num_entries = len(execution_list)
                spikelet_avg = spikelet_sum / num_entries if num_entries > 0 else 0
                clasp_avg = clasp_sum / num_entries if num_entries > 0 else 0
                downsampling_avg = downsampling_sum / num_entries if num_entries > 0 else 0
                
                print(f"Time averages for {comb}:")
                print(f"  Spikelet: {spikelet_avg}")
                print(f"  ClaSP: {clasp_avg}")
                print(f"  Downsampling: {downsampling_avg}")
    else:
        print("execution times are not in the dictionary")

def f1_score(data):
    results = {}
    for combination, predicted_lists in data["predicted_change_points"].items():
        results[combination] = []
        print(f"\nCombination: {combination}")
        for predicted, ground_truth in zip(predicted_lists, data["true_change_points"]):
            f1_score = calculate_f1_score(predicted, ground_truth)
            #print(f"Predicted: {predicted}, Ground Truth: {ground_truth}, F1 Score: {f1_score:.2f}")
            results[combination].append(f1_score)
    
    return results

def calculate_f1_score(predicted_cps, ground_truth_cps, tolerance=5):
    """
    Calculate the F1 score for change point detection.

    Parameters:
    - predicted_cps: List of predicted change point indices.
    - ground_truth_cps: List of ground truth change point indices.
    - tolerance: int, acceptable range for matching change points.

    Returns:
    - f1: float, F1 score.
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Match predicted with ground truth within the tolerance range
    matched = set()
    for pred_cp in predicted_cps:
        match_found = False
        for gt_cp in ground_truth_cps:
            if abs(pred_cp - gt_cp) <= tolerance and gt_cp not in matched:
                true_positives += 1
                matched.add(gt_cp)
                match_found = True
                break
        if not match_found:
            false_positives += 1

    false_negatives = len(ground_truth_cps) - len(matched)

    # Calculate precision, recall, and F1
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0

    if precision + recall == 0:
        return 0  # Avoid division by zero

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def sum_execution_times_per_dataset(execution_times):
    result = {}
    for combination in execution_times:
        result[combination] = []
        for entry in execution_times[combination]:
            values = entry.values()
            total = sum(values)
            result[combination].append(total)
         
    return result

def calculate_percentage_difference_clasp_vs_variants(data):
    """
    Berechnet den prozentualen Unterschied der Ausführungszeiten zwischen
    ClaSP und den Varianten ClaSP + nth sowie ClaSP + Extrema.

    Parameters:
    -----------
    execution_times : dict
        Dictionary mit den Ausführungszeiten für jede Kombination.
        Beispiel:
        {
            "Spikelet=False, Downsampling=None": [0.5, 0.6, 0.7],
            "Spikelet=False, Downsampling=nth": [0.4, 0.5, 0.6],
            "Spikelet=False, Downsampling=Extrema": [0.3, 0.4, 0.5],
        }
    dataset_index : int
        Index des Datensatzes, für den der Unterschied berechnet werden soll.

    Returns:
    --------
    dict
        Dictionary mit den prozentualen Unterschieden:
        {
            "nth_difference": value,
            "extrema_difference": value
        }
    """
    results = {
        "nth_difference": [],
        "extrema_difference": [],
    }
    execution_times = sum_execution_times_per_dataset(data["execution_times"])

    for idx in range(len(data["time_series_name"])):
        clasp_time = execution_times["Spikelet=False, Downsampling=None"][idx]
        clasp_nth_time = execution_times["Spikelet=False, Downsampling=nth"][idx]
        clasp_extrema_time = execution_times["Spikelet=False, Downsampling=Extrema"][idx]
    
        nth_difference = ((clasp_nth_time - clasp_time) / clasp_time) * 100 if clasp_time != 0 else 0
        extrema_difference = ((clasp_extrema_time - clasp_time) / clasp_time) * 100 if clasp_time != 0 else 0

        results["nth_difference"].append(nth_difference)
        results["extrema_difference"].append(extrema_difference)

    return results

def calc_median(values):
    values = sorted(values)
    n = len(values)
    if n % 2 == 1:
        return values[n // 2]
    else:
        mid1 = values[n // 2 - 1]
        mid2 = values[n // 2]
        return (mid1 + mid2) / 2


