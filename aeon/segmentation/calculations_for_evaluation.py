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

def calc_and_print_median_scores(data):
    """
    Berechnet und gibt die Medianscores für jede Kombination in 'scores' aus.
    
    Parameters:
    - data (dict): Ein Dictionary, das eine "scores"-Liste mit Werten enthält.
    """
    if "scores" in data:
        for comb in data["scores"]:
            median = np.median(data["scores"][comb])  # Berechnet den Median der Scores
            print(f"Score median von {comb}: {median}")
    else:
        print("Scores is not in the dictionary")

def calc_and_print_avg_time_diff(data):
    for comb in data:
        averages = calc_average(data[comb])
        print(f"Time average von {comb}: {averages}")

def calc_and_print_avg_times(data):
    """
    Berechnet die Gesamtausführungszeit (Summe von Spikelet, ClaSP, Downsampling)
    für jede Kombination und gibt den Durchschnitt aus.
    
    Parameters:
    - data (dict): Ein Dictionary mit "execution_times"
    """
    if "execution_times" in data:
        for comb in data["execution_times"]:
            total_sum = 0.0
            execution_list = data["execution_times"][comb]

            # Summe aller Zeiten (Spikelet, ClaSP, Downsampling)
            for entry in execution_list:
                if isinstance(entry, dict):
                    spikelet_time = entry.get("Spikelet", 0)
                    clasp_time = entry.get("ClaSP", 0)
                    downsampling_time = entry.get("Downsampling", 0)
                    total_sum += spikelet_time + clasp_time + downsampling_time
                
            num_entries = len(execution_list)

            # Durchschnitt aus der Gesamtsumme
            total_avg = total_sum / num_entries if num_entries > 0 else 0
            
            print(f"Time average for {comb}: {total_avg:.6f}")
    else:
        print("execution_times are not in the dictionary")


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
            total_for_entry = sum(entry.values())
            result[combination] = total_for_entry

    return result

def sum_execution_times_per_dataset2(execution_times):
    """
    Sums up execution times for each key within a dataset.
    
    Parameters:
    - execution_times (dict): A dictionary where keys are dataset descriptions 
      and values are lists of execution time components.
    
    Returns:
    - dict: A dictionary with summed execution times per dataset.
    """
    summed_times = []
    
    for dicts in execution_times:
        total_time = sum(dicts.values())
        summed_times.append(total_time)
    
    return summed_times

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

def calc_and_print_std_deviation(times, lengths):
    normalized_runtimes = [rt / l for rt, l in zip(times, lengths)]
    std_dev_normalized = np.std(normalized_runtimes)
    print("standard deviation: ", std_dev_normalized)
    return std_dev_normalized

def konfidenzintervall(mean, stddev_abs):
    z = 1.64  # 1.96 für 95% Konfidenzintervall, 1.64 für 10% Konfidenzintervall
    ci_lower = mean - z * stddev_abs
    ci_upper = mean + z * stddev_abs
    print("Konfidenzintervall:", (ci_lower, ci_upper))

def mse_ts(ts1, ts2):
    ts1 = np.array(ts1)
    ts2 = np.array(ts2)
    return np.mean((ts1-ts2)**2)

def calc_mse(data1, data2):
    nth_mse = 0
    extrema_mse = 0
    for ts_name in data1:
        print("downsampled len: ", len(data1[ts_name]["nth"]["downsampled_ts"]))
        print("downsampled spikelet len: ", len(data2[ts_name]["nth"]["spikelet_downsampled_ts"]))
        nth_mse += mse_ts(data1[ts_name]["nth"]["downsampled_ts"], data2[ts_name]["nth"]["spikelet_downsampled_ts"])
        extrema_mse += mse_ts(data1[ts_name]["Extrema"]["downsampled_ts"], data2[ts_name]["Extrema"]["spikelet_downsampled_ts"])
    return nth_mse/len(data1), extrema_mse/len(data1)

