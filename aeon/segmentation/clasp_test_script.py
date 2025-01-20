import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pandas as pd
import tracemalloc

from datetime import datetime

output_dir = r'C:\Users\Victor\Desktop\Uni\Bachelor\output'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_dir)

from aeon.segmentation import ClaSPSegmenter, find_dominant_window_sizes
from aeon.transformations.spikelet.spikelet import motif_discovery_and_clasp

from timessb.tssb.utils import load_time_series_segmentation_datasets
from timessb.tssb.utils import visualize_time_series
from claspy.segmentation import BinaryClaSPSegmentation
from claspy.data_loader import load_tssb_dataset

from runSpikelet import runSpikelet
from runDownsampling import runDownsampling
from runClaSP import runClaSP
from eval import add_evaluation_data, runEval

def warm_up():
    print("Warm-Up start...\nSadly a must.")
    start_time = datetime.now()

    dummy_data = np.random.rand(100)  
    dummy_clasp = BinaryClaSPSegmentation()
    dummy_clasp.fit_predict(dummy_data)

    warmup_time = (datetime.now() - start_time).total_seconds()
    print("Warm-Up end. Time: ", warmup_time, " seconds")

def save_timings_and_memory_to_file():
    
    with open(combined_output_file, 'w') as f:
        f.write(f"Benchmark: {ts_name}\n\n")
        for idx, result in enumerate(all_results):
            f.write(f"--- Ergebnisse für Kombination ({idx + 1}): Spikelet={result['use_spikelet']}, Downsampling={result['use_downsampling']} ---\n")
            f.write(f"Score: {result['score']}\n")
            f.write(f"Time for ClaSP: {result['clasp_fit_predict_time']:.4f} seconds\n")
            f.write(f"Time for Spikelet: {result['transformation_time']:.4f} seconds\n")
            f.write(f"Time for Downsampling: {result['downsampling_time']:.4f} seconds\n")
            f.write(f"Window Size: {result['window_size']}\n")
            f.write(f"Time Series length: {result['time_series_length']}\n")
            f.write(f"Original Time Series Length: {result['original_time_series_length']}\n")
            f.write(f"Peak memory usage ClaSP: {result['clasp_memory_peak'] / 1024:.2f} KB\n")
            f.write(f"Current memory usage ClaSP: {result['clasp_memory_current'] / 1024:.2f} KB\n")
            f.write(f"Peak memory usage Spikelet: {result['spikelet_memory_peak'] / 1024:.2f} KB\n")
            f.write(f"Current memory usage Spikelet: {result['spikelet_memory_current'] / 1024:.2f} KB\n")
            f.write(f"Peak memory usage Downsampling: {result['downsampling_memory_peak'] / 1024:.2f} KB\n")
            f.write(f"Current memory usage Downsampling: {result['downsampling_memory_peak'] / 1024:.2f} KB\n\n")
            
def execute_combination(use_spikelet, downAlg, pltshow, ts_name, window_size, cps, ts, multiple_iterations_spikelet, spikelet_mat=None, spikelet_cot=None):
    """
    Führt eine bestimmte Kombination aus Spikelet, Downsampling und ClaSP aus.
    """
    tracemalloc.start()

    # dataset, window_size, true_cps, time_series = load_tssb_dataset(names=(ts_name,)).iloc[0, :]
    org_length = len(ts)
    print("Time Series Länge:", len(ts))

    ts_down = []
    downsampled_length = 0

    if pltshow: 
        visualize_time_series(ts, ts_name + ": Datensatz Original", cps)

    transformation_time = 0
    clasp_fit_time = 0
    downsampling_time = 0
    spikelet_memory = {"current_memory": 0, "peak_memory": 0}
    clasp_memory = {"current_memory": 0, "peak_memory": 0}
    downsampling_memory = {"current_memory": 0, "peak_memory": 0}

    # Spikelet
    if use_spikelet:
        if multiple_iterations_spikelet:
            ts, transformation_time, spikelet_memory = runSpikelet(
            ts, ts_name, cps, pltshow, mat=spikelet_mat, cot=spikelet_cot
        )
        else: 
            ts, transformation_time, spikelet_memory = runSpikelet(ts, ts_name, cps, pltshow)
    else:
        print("Spikelet wird nicht ausgeführt.")

    # Downsampling
    if downAlg != "None":
        ts_down, window_size, downsampled_length, downsampling_time, downsampling_memory = runDownsampling(ts, ts_name, downAlg, window_size, pltshow)
    else:
        print("Downsampling wird nicht ausgeführt.")
    
    # ClaSP
    print("Die selbst bestimmte Window Size ist:", window_size)
    score, clasp_fit_time, clasp_memory, predicted_cps = runClaSP(ts, ts_name, cps, window_size, ts_down, downAlg, downsampled_length, pltshow)

    elapsed_times = {
        "use_spikelet": use_spikelet,
        "use_downsampling": downAlg,
        "benchmark": ts_name,
        "score": score,
        "clasp_fit_predict_time": clasp_fit_time,
        "transformation_time": transformation_time,
        "downsampling_time": downsampling_time,  
        "window_size": window_size,
        "time_series_length": downsampled_length,
        "original_time_series_length": org_length,
        "clasp_memory_peak": clasp_memory["peak_memory"],  
        "clasp_memory_current": clasp_memory["current_memory"],
        "spikelet_memory_peak": spikelet_memory["peak_memory"],
        "spikelet_memory_current": spikelet_memory["current_memory"],
        "downsampling_memory_peak": downsampling_memory["peak_memory"],
        "downsampling_memory_current": downsampling_memory["current_memory"],

        "downsampled_ts": ts_down,
        "predicted_cps": predicted_cps,
    }

    tracemalloc.stop()
    return elapsed_times

###############################################################################################

benchmarkNames = ["Crop"]   # Datensatz aus data_loader.py laden
doEverything = False    # Wenn aktiviert, werden alle Kombinationen ausgeführt

use_spikelet = True
downAlg = "Extrema"     # Either "nth" or "Extrema"
pltshow = False  

###############################################################################################

run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = os.path.join(output_dir, f"Run_{run_timestamp}")
os.makedirs(output_dir, exist_ok=True)

edit_string = "_result"

warm_up()

tssb = load_time_series_segmentation_datasets(names=benchmarkNames)
all_eval_data = []
multiple_iterations_spikelet = False

for _, (ts_name, window_size, cps, ts) in tssb.iterrows():
    if doEverything:
        all_results = []
        edit_string = "_all_results"
        combinations = [
            (False, "None", False),          # Original ClaSP
            (False, "nth", False),           # ClaSP + nth
            (False, "Extrema", False),       # ClaSP + Extrema
            (True, "None", False),           # ClaSP + Spikelet
            (True, "nth", False),            # ClaSP + Spikelet + nth
            (True, "Extrema", False)         # ClaSP + Spikelet + Extrema
        ]
    else: 
        multiple_iterations_spikelet = True
        all_results = []
        combinations = [
            (use_spikelet, downAlg, pltshow)
        ]

    evaluation_data = {
        "original_time_series_len": None,  # Originale Zeitreihe
        "time_series_name": None,      # Name der Zeitreihe
        "true_change_points": [],      # Positionen der echten Change Points
        "predicted_change_points": {}, # Predicted CPs pro Kombination
        "downsampled_time_series_len": {}, # Gedownsamplete Zeitreihen pro Kombination
        "scores": {},                  # Scores pro Kombination
        "execution_times": {},         # Ausführungszeiten pro Kombination
        "memory_consumption": {},      # Speicherverbrauch pro Kombination
        "parameters": {},
    }

    evaluation_data["time_series_name"] = ts_name
    evaluation_data["original_time_series_len"] = len(ts)
    evaluation_data["true_change_points"] = cps

    for use_spikelet, downAlg, pltshow in combinations:
        print(f"\n{ts_name}:\nStarte Kombination: Spikelet={use_spikelet}, Downsampling={downAlg}\n")
        combination_name = f"Spikelet={use_spikelet}, Downsampling={downAlg}"
        
        if multiple_iterations_spikelet:
            # Werte für MaT und CoT definieren (anpassen je nach Bedarf)
            spikelet_settings = [
                {"MaT": 1, "CoT": 20},
                {"MaT": 2, "CoT": 20},
            ]
            for settings in spikelet_settings:
                print(f"\nTeste Spikelet mit MaT={settings['MaT']} und CoT={settings['CoT']}")

                # Anpassung von Spikelet-Parametern vor dem Run
                result = execute_combination(
                    use_spikelet, downAlg, pltshow, ts_name, window_size, cps, ts, 
                    multiple_iterations_spikelet, spikelet_mat=settings["MaT"], spikelet_cot=settings["CoT"]
                )

                all_results.append(result)

                # Kombinationsname basierend auf MaT und CoT erstellen
                setting_name = f"MaT={settings['MaT']}, CoT={settings['CoT']}"

                # Ergebnisse speichern
                evaluation_data["predicted_change_points"][setting_name] = result["predicted_cps"]
                evaluation_data["downsampled_time_series_len"][setting_name] = len(result["downsampled_ts"])
                evaluation_data["scores"][setting_name] = result["score"]
                evaluation_data["execution_times"][setting_name] = {
                    "Spikelet": result["transformation_time"],
                    "ClaSP": result["clasp_fit_predict_time"],
                    "Downsampling": result["downsampling_time"]
                }
                evaluation_data["memory_consumption"][setting_name] = {
                    "Spikelet": {
                        "peak": result["spikelet_memory_peak"],
                        "current": result["spikelet_memory_current"]
                    },
                    "ClaSP": {
                        "peak": result["clasp_memory_peak"],
                        "current": result["clasp_memory_current"]
                    },
                    "Downsampling": {
                        "peak": result["downsampling_memory_peak"],
                        "current": result["downsampling_memory_current"]
                    },
                }
                evaluation_data["parameters"][setting_name] = {
                    "window_size": result["window_size"],
                    "n_for_nth_point": 3 if downAlg == "nth" else None,
                    "MaT": settings["MaT"],
                    "CoT": settings["CoT"]
                }
        else:
            result = execute_combination(use_spikelet, downAlg, pltshow, ts_name, window_size, cps, ts, multiple_iterations_spikelet)
            all_results.append(result)

            evaluation_data["predicted_change_points"][combination_name] = result["predicted_cps"]
            evaluation_data["downsampled_time_series_len"][combination_name] = len(result["downsampled_ts"])
            evaluation_data["scores"][combination_name] = result["score"]
            evaluation_data["execution_times"][combination_name] = {
                "Spikelet": result["transformation_time"],
                "ClaSP": result["clasp_fit_predict_time"],
                "Downsampling": result["downsampling_time"]
            }
            evaluation_data["memory_consumption"][combination_name] = {
                "Spikelet": {
                    "peak": result["spikelet_memory_peak"],
                    "current": result["spikelet_memory_current"]
                },
                "ClaSP": {
                    "peak": result["clasp_memory_peak"],
                    "current": result["clasp_memory_current"]
                },
                "Downsampling": {
                    "peak": result["downsampling_memory_peak"],
                    "current": result["downsampling_memory_current"]
                },
            }
            evaluation_data["parameters"][combination_name] = {
                "window_size": result["window_size"],
                "n_for_nth_point": 3 if downAlg == "nth" else None,
            }

    combined_output_file = os.path.join(output_dir, f"{ts_name}{edit_string}.txt")
    save_timings_and_memory_to_file()
    print(f"Alle Ergebnisse wurden in {combined_output_file} gespeichert.")
    all_eval_data.append(evaluation_data)
    add_evaluation_data(evaluation_data)

if doEverything: runEval(output_dir)

