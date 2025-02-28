import sys
import os
import numpy as np
import pdb
import pandas as pd
import tracemalloc
import json
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import signal
import threading
import matplotlib.pyplot as plt

class TimeoutException(Exception):
    pass

output_dir = r'C:\Users\Victor\Desktop\Uni\Bachelor\output'
output_dir2 = Path("C:/Users/Victor/Desktop/Uni/Bachelor/output")
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_dir)
lock = threading.Lock()

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
from save_run_info_txt import save_run_info_txt
from utils import load_floss_dataset, load_ucr_dataset, load_combined_dataset

evaluation_data_template = {                     # Daten in der Json datei am Ende. Falls nicht gespeichert werden soll -> auskommentieren
        "original_time_series_len": None,   # Originale Zeitreihe
        "time_series_name": None,           # Name der Zeitreihe
        "true_change_points": [],           # Positionen der echten Change Points
        "predicted_change_points": {},      # Predicted CPs pro Kombination
        "downsampled_time_series_len": {},  # Gedownsamplete Zeitreihen pro Kombination
        "scores": {},                       # Scores pro Kombination
        "execution_times": {},              # Ausführungszeiten pro Kombination
        "memory_consumption": {},           # Speicherverbrauch pro Kombination
        #"parameters": {},                   # Parameter
    }

def convert_arrays_to_lists(data):
    """Ersetzt in einem beliebig verschachtelten Dict oder Listen
    alle NumPy-Arrays durch Python-Listen und NumPy-Typen durch Python-Typen."""
    if isinstance(data, dict):
        return {k: convert_arrays_to_lists(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_arrays_to_lists(x) for x in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.int64, np.int32, np.float64, np.float32)):  # Hier numpy-Typen in Standard-Python-Typen umwandeln
        return data.item()
    else:
        return data

    
def save_evaluation_data(evaluation_data, output_dir, file_name="evaluation_results.json"):
    """
    Speichert ein Dictionary 'evaluation_data' als JSON unter 'output_dir/file_name'.
    """
    # Falls das Ausgabeverzeichnis noch nicht existiert, legen wir es an
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Ziel-Pfad
    file_path = output_path / file_name

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(convert_arrays_to_lists(evaluation_data), f, indent=4)

    print(f"Evaluation Data wurde gespeichert in: {file_path}")

def save_evaluation_data_incremental(new_data, output_dir2, ts_name, len_ts, cps, file_name="evaluation_results.json"):
    file_path = Path(output_dir2) / file_name
    if file_path.exists():
        with file_path.open("r", encoding="utf-8") as f:
            existing_data = json.load(f)
    else:
        existing_data = deepcopy(evaluation_data_template)

    if "time_series_name" in existing_data:
        if not isinstance(existing_data["time_series_name"], list):
            existing_data["time_series_name"] = []
        existing_data["time_series_name"].append(ts_name)

    if "original_time_series_len" in existing_data:
        if not isinstance(existing_data["original_time_series_len"], list):
            existing_data["original_time_series_len"] = []
        elif isinstance(existing_data["original_time_series_len"], int):
            existing_data["original_time_series_len"] = [existing_data["original_time_series_len"]]
        existing_data["original_time_series_len"].append(len_ts)  

    if "true_change_points" in existing_data:
        if isinstance(existing_data["true_change_points"], np.ndarray):
            existing_data["true_change_points"] = evaluation_data["true_change_points"].tolist() 
        elif not isinstance(existing_data["true_change_points"], list):
            existing_data["true_change_points"] = [evaluation_data["true_change_points"]]
        else:
            existing_data["true_change_points"].append(cps)
    
    if "predicted_change_points" in existing_data:
        for combination_name in new_data["predicted_change_points"]:
            if combination_name not in existing_data["predicted_change_points"]: 
                existing_data["predicted_change_points"][combination_name] = []
            existing_data["predicted_change_points"][combination_name].extend(new_data["predicted_change_points"][combination_name])

    if "downsampled_time_series_len" in existing_data:
        for combination_name in new_data["downsampled_time_series_len"]:
            if combination_name not in existing_data["downsampled_time_series_len"]: 
                existing_data["downsampled_time_series_len"][combination_name] = []
            existing_data["downsampled_time_series_len"][combination_name].extend(new_data["downsampled_time_series_len"][combination_name])

    if "scores" in existing_data:
        for combination_name in new_data["scores"]:
            if combination_name not in existing_data["scores"]: 
                existing_data["scores"][combination_name] = []
            if isinstance(new_data["scores"][combination_name], list):
                existing_data["scores"][combination_name].extend(new_data["scores"][combination_name])
            else:
                existing_data["scores"][combination_name].append(new_data["scores"][combination_name])
    
    if "execution_times" in existing_data:
        for combination_name, new_list_of_dicts in new_data["execution_times"].items():
            if combination_name not in existing_data["execution_times"]:
                existing_data["execution_times"][combination_name] = []
            if isinstance(new_list_of_dicts, dict):
                new_list_of_dicts = [new_list_of_dicts]
            if isinstance(new_list_of_dicts, list) and all(isinstance(d, dict) for d in new_list_of_dicts):
                existing_data["execution_times"][combination_name].extend(new_list_of_dicts)
            else:
                print(f"Invalid execution times format for {combination_name}: {new_list_of_dicts}")

    if "memory_consumption" in existing_data:
        for combination_name, memory_values in new_data["memory_consumption"].items():
            if combination_name not in existing_data["memory_consumption"]:
                existing_data["memory_consumption"][combination_name] = {
                    "Spikelet": {"peak": [], "current": []},
                    "ClaSP": {"peak": [], "current": []},
                    "Downsampling": {"peak": [], "current": []}
                }
                
            for component in ["Spikelet", "ClaSP", "Downsampling"]:
                if component in memory_values:
                    entry = memory_values[component]
                    if isinstance(entry, dict):
                        if "peak" in entry:
                            existing_data["memory_consumption"][combination_name][component]["peak"].append(entry["peak"])
                        else:
                            print(f"⚠️ '{component}.peak' fehlt für {combination_name}")

                        if "current" in entry:
                            existing_data["memory_consumption"][combination_name][component]["current"].append(entry["current"])
                        else:
                            print(f"'{component}.current' fehlt für {combination_name}")
                    else:
                        print(f"'{component}' ist kein Dictionary für {combination_name}: {entry}")
                else:
                    print(f"'{component}' fehlt für {combination_name}")





    # if "parameters" in existing_data:  
    #     for combination_name in new_data["parameters"]:
    #         if combination_name not in existing_data["parameters"]:
    #             existing_data["parameters"][combination_name] = {"window_size": []}
    #         existing_data["parameters"][combination_name]["window_size"].append(new_data["parameters"][combination_name])

    final_data = convert_arrays_to_lists(existing_data)
    with file_path.open("w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4)

    print(f"Inkrementell gespeichert in {file_path}")


def warm_up():
    print("Warm-Up start...\nSadly a must.")
    start_time = datetime.now()

    dummy_data = np.random.rand(100)  
    dummy_clasp = BinaryClaSPSegmentation()
    dummy_clasp.fit_predict(dummy_data)

    warmup_time = (datetime.now() - start_time).total_seconds()
    print("Warm-Up end. Time: ", warmup_time, " seconds")
            
def execute_combination(analysis_mode, use_spikelet, downAlgName, downAlgParam, pltshow, ts_name, window_size, cps, ts, order, mewsImplementation, spikelet_mat=None, spikelet_cot=None):
    """
    Führt eine bestimmte Kombination aus Spikelet, Downsampling und ClaSP aus.
    """
    tracemalloc.start()

    org_ts = ts
    org_length = len(ts)
    print("Time Series Länge:", len(ts))

    if downAlgParam is not None:
        downAlgKey = f"{downAlgName}_{downAlgParam}"
    else:
        downAlgKey = downAlgName  # kein Parameter => nur den Namen

    ts_down = []
    downsampled_length = 0

    # if True: 
    #     fig, ax = visualize_time_series(ts, ts_name + ": Original", cps)
    #     plt.show()

    transformation_time = 0
    clasp_fit_time = 0
    downsampling_time = 0
    spikelet_memory = {"current_memory": 0, "peak_memory": 0}
    clasp_memory = {"current_memory": 0, "peak_memory": 0}
    downsampling_memory = {"current_memory": 0, "peak_memory": 0}
    
    def do_spikelet(current_ts):
        nonlocal transformation_time, spikelet_memory
        if use_spikelet:
            if not mewsImplementation:
                return runSpikelet(analysis_mode, current_ts, ts_name, cps, pltshow, mat=spikelet_mat, cot=spikelet_cot)
            else: 
                return runSpikelet(analysis_mode, current_ts, ts_name, cps, pltshow)
        else:
            print("Spikelet wird nicht ausgeführt.")
            return current_ts, 0, {"current_memory": 0, "peak_memory": 0}

    def do_downsampling(current_ts, current_window_size):
        nonlocal downsampling_time, downsampling_memory, ts_down, downsampled_length
        if downAlgName != "None":
            result_ts, new_window_size, new_length, d_time, d_memory = runDownsampling(
                current_ts, ts_name, downAlgName, downAlgParam, current_window_size, pltshow, cps
            )
            ts_down = result_ts
            downsampled_length = new_length
            downsampling_time = d_time
            downsampling_memory = d_memory
            return result_ts, new_window_size
        else:
            print("Downsampling wird nicht ausgeführt.")
            return current_ts, current_window_size

    if order == "spikelet_first":
        ts, transformation_time, spikelet_memory = do_spikelet(ts)
        ts, window_size = do_downsampling(ts, window_size)
    elif order == "downsampling_first":
        ts, window_size = do_downsampling(ts, window_size)
        ts, transformation_time, spikelet_memory = do_spikelet(ts)
    else:
        raise ValueError("Ungültiger Wert für 'order'. Nutze 'spikelet_first' oder 'downsampling_first'.")

    # ClaSP
    print("Die selbst bestimmte Window Size ist:", window_size)
    score, clasp_fit_time, clasp_memory, predicted_cps = runClaSP(org_ts, ts_name, cps, window_size, ts, downAlgKey, downsampled_length, pltshow)

    elapsed_times = {
        "use_spikelet": use_spikelet,
        "use_downsampling": downAlgKey,
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

    if analysis_mode:
        output_directory = Path("C:\\Users\\Victor\\Desktop\\Uni\\Bachelor\\optimal_output\\processed_ts")
        output_directory.mkdir(parents=True, exist_ok=True)
        output_file = output_directory / f"{ts_name}.json"

        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump({"ts_name": ts_name, "processed_time_series": convert_arrays_to_lists(ts)}, file, indent=4)

    tracemalloc.stop()
    return elapsed_times
    
def timeout_handler(signum, frame):
    print("execute_combination exceeded 120 seconds!")
    raise TimeoutException("execute_combination exceeded 120 seconds!")

def run(analysis_mode, benchmarkSet, benchmarkNames, mewsImplementation, run_settings, evaluation_data, output_dir, order):
    
    if benchmarkSet != "tssb" and benchmarkSet != "floss" and benchmarkSet != "all":
        raise ValueError("benchmarkSet muss entweder 'tssb', 'floss' oder 'all' sein.")
    else:
        if benchmarkSet == "floss":
            if benchmarkNames == []:
                tssb = load_floss_dataset()
            else: 
                tssb = load_floss_dataset(selection=benchmarkNames)
        elif benchmarkSet == "tssb":
            if benchmarkNames == []:
                tssb = load_ucr_dataset()
            else: 
                tssb = load_time_series_segmentation_datasets(names=benchmarkNames)
        elif benchmarkSet == "all":
            tssb = load_combined_dataset()

    if run_settings == []:
        run_settings = [
            # (False, "None", False),          # Original ClaSP
            # (False, ("nth", 2), False),      # ClaSP + nth
            # (False, ("nth", 3), False),
            # (False, ("nth", 4), False),           
            # (False, ("nth", 5), False),
            # (False, ("nth", 6), False),
            # (False, ("nth", 7), False),
            # (False, ("nth", 8), False),
            # (False, ("nth", 9), False),
            # (False, ("nth", 10), False),
            # (False, ("nth", 11), False),
            # (False, ("nth", 12), False),
            # (False, ("nth", 13), False),
            # (False, ("nth", 14), False),
            # (False, ("nth", 15), False),
            # (False, ("nth2", 2), False),
            # (False, ("nth2", 3), False),
            # (False, ("nth2", 4), False),           
            # (False, ("nth2", 5), False),
            # (False, ("nth2", 6), False),
            # (False, ("nth2", 7), False),
            # (False, ("nth2", 8), False),
            # (False, ("nth2", 9), False),
            # (False, ("nth2", 10), False),
            # (False, ("nth2", 11), False),
            # (False, ("nth2", 12), False),
            # (False, ("nth2", 13), False),
            # (False, ("nth2", 14), False),
            # (False, ("nth2", 15), False),
            # (False, ("Extrema", 3), False),       # ClaSP + Extrema
            # (False, ("Extrema", 4), False),
            # (False, ("Extrema", 5), False),
            # (False, ("Extrema", 6), False),
            # (False, ("Extrema", 7), False),
            # (False, ("Extrema", 8), False),
            # (False, ("Extrema", 9), False),
            # (False, ("Extrema", 10), False),
            # (False, ("Extrema", 11), False),
            # (False, ("Extrema", 12), False),
            # (False, ("Extrema", 13), False),
            # (False, ("Extrema", 14), False),
            # (False, ("Extrema", 15), False),
            # (False, ("Extrema", 16), False),
            # (False, ("Extrema", 17), False),
            # (False, ("Extrema", 18), False),
            # (False, ("Extrema", 19), False),
            # (False, ("Extrema2", 3), False),
            # (False, ("Extrema2", 4), False),
            # (False, ("Extrema2", 5), False),
            # (False, ("Extrema2", 6), False),
            # (False, ("Extrema2", 7), False),
            # (False, ("Extrema2", 8), False),
            # (False, ("Extrema2", 9), False),
            # (False, ("Extrema2", 10), False),
            # (False, ("Extrema2", 11), False),
            # (False, ("Extrema2", 12), False),
            # (False, ("Extrema2", 13), False),
            # (False, ("Extrema2", 14), False),
            # (False, ("Extrema2", 15), False),
            # (False, ("Extrema2", 16), False),
            # (False, ("Extrema2", 17), False),
            # (False, ("Extrema2", 18), False),
            # (False, ("Extrema2", 19), False),
            # (True, "None", False),           # ClaSP + Spikelet
            # (True, ("nth", 2), False),          # ClaSP + Spikelet + nth
            # (True, ("nth", 3), False),
            # (True, ("nth", 4), False),            
            # (True, ("nth", 5), False),
            # (True, ("nth", 6), False),
            # (True, ("nth", 7), False),
            # (True, ("nth", 8), False),
            # (True, ("nth", 9), False),
            # (True, ("nth", 10), False),
            # (True, ("nth", 11), False),
            # (True, ("nth", 12), False),
            # (True, ("nth", 13), False),
            # (True, ("nth", 14), False),
            # (True, ("nth", 15), False),
            # (True, ("nth2", 2), False),
            # (True, ("nth2", 3), False),
            # (True, ("nth2", 4), False),            
            # (True, ("nth2", 5), False),
            # (True, ("nth2", 6), False),
            # (True, ("nth2", 7), False),
            # (True, ("nth2", 8), False),
            # (True, ("nth2", 9), False),
            # (True, ("nth2", 10), False),
            # (True, ("nth2", 11), False),
            # (True, ("nth2", 12), False),
            # (True, ("nth2", 13), False),
            # (True, ("nth2", 14), False),
            # (True, ("nth2", 15), False),
            # (True, ("Extrema", 3), False),          # ClaSP + Spikelet + Extrema
            # (True, ("Extrema", 4), False),
            # (True, ("Extrema", 5), False),
            # (True, ("Extrema", 6), False),
            # (True, ("Extrema", 7), False),
            # (True, ("Extrema", 8), False),
            # (True, ("Extrema", 9), False),
            # (True, ("Extrema", 10), False),
            # (True, ("Extrema", 11), False),
            # (True, ("Extrema", 12), False),
            # (True, ("Extrema", 13), False),
            # (True, ("Extrema", 14), False),
            # (True, ("Extrema", 15), False),
            # (True, ("Extrema", 16), False),
            # (True, ("Extrema", 17), False),
            # (True, ("Extrema", 18), False),
            # (True, ("Extrema", 19), False),
            # (True, ("Extrema2", 3), False),
            # (True, ("Extrema2", 4), False),
            # (True, ("Extrema2", 5), False),
            # (True, ("Extrema2", 6), False),
            # (True, ("Extrema2", 7), False),
            # (True, ("Extrema2", 8), False),
            # (True, ("Extrema2", 9), False),
            # (True, ("Extrema2", 10), False),
            # (True, ("Extrema2", 11), False),
            # (True, ("Extrema2", 12), False),
            # (True, ("Extrema2", 13), False),
            # (True, ("Extrema2", 14), False),
            # (True, ("Extrema2", 15), False),
            # (True, ("Extrema2", 16), False),
            # (True, ("Extrema2", 17), False),
            # (True, ("Extrema2", 18), False),
            # (True, ("Extrema2", 19), False),
        ]

    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir2 = Path(f"..") / ".." / ".." /"output"/f"Run_{run_timestamp}"
    output_dir2.mkdir(parents=True, exist_ok=True)
    skip = True
    warm_up()

    for _, (ts_name, window_size, cps, ts) in tssb.iterrows():
        ###########################
        #if skip:              # Skipped Benchmark wenn kleiner als 5000 Datenpunkte
        # if ts_name != 'InsectEPG2': # and ts_name != 'GrandMalSeizures2' and ts_name != 'GrandMalSeizures':
        # #         #skip = False
        #     continue
        ###########################

        all_results = []
        if "time_series_name" in evaluation_data:
            evaluation_data["time_series_name"] = ts_name
        if "original_time_series_len" in evaluation_data:
            evaluation_data["original_time_series_len"] = len(ts)
        if "true_change_points" in evaluation_data:
            evaluation_data["true_change_points"] = cps

        for use_spikelet, downAlgSpec, pltshow in run_settings:
            if use_spikelet != True and use_spikelet != False:
                raise ValueError("Erster Eintrag bei 'run_settings' muss True oder False sein.")

            if isinstance(downAlgSpec, tuple):
                downAlgName, downAlgParam = downAlgSpec
            else:
                downAlgName = downAlgSpec
                downAlgParam = None

            if downAlgName != "" and downAlgName != "None" and downAlgName != "nth" and downAlgName != "Extrema" and downAlgName != "Extrema2" and downAlgName != "nth2":
                raise ValueError("Validen Downsampling Algorithmus angeben oder leeren String eingeben.")

            if pltshow != True and pltshow != False:
                raise ValueError("Dritter Eintrag bei 'run_settings' muss True oder False sein.")

            print(f"\n{ts_name}:\nStarte Kombination: Spikelet={use_spikelet}, Downsampling={downAlgName}, Param={downAlgParam}\n")

            combination_name = f"Spikelet={use_spikelet}, Downsampling={downAlgName}"
            if downAlgParam is not None:
                combination_name = f"{combination_name}, Param={downAlgParam}"

            if not mewsImplementation and analysis_mode == False:
                json_output_dir = output_dir2/"benchmark_jsons"
                json_output_dir.mkdir(parents=True, exist_ok=True)

                dataset_json_name = f"{ts_name}.json"
                output_json_file = json_output_dir / dataset_json_name

                num_iteration = 0

                start_MaT = 0.1                                 # Startwert für MaT
                end_MaT = 3                                     # Endwert für MaT
                step_MaT = 0.01                                 # Schrittweite für MaT
                CoT = 15                                        # Konstanter Wert für CoT

                current_MaT = start_MaT
                while num_iteration <= 300:
                    print(f"\nTeste Spikelet mit MaT={current_MaT} und CoT={CoT}, Benchmark: {ts_name}")
                    # signal.signal(signal.SIGALRM, timeout_handler)
                    # signal.alarm(120)
                    try:
                        tracemalloc.start()
                        result = execute_combination(analysis_mode, use_spikelet, downAlgName, downAlgParam, pltshow, ts_name, window_size, cps, ts, order, mewsImplementation, spikelet_mat=current_MaT, spikelet_cot=CoT)
                        
                        all_results.append(result)

                        setting_name = f"MaT={current_MaT}, CoT={CoT}"

                        # Ergebnisse speichern
                        if "predicted_change_points" in evaluation_data:
                            evaluation_data["predicted_change_points"][setting_name] = result["predicted_cps"]

                        if "downsampled_time_series_len" in evaluation_data:
                            evaluation_data["downsampled_time_series_len"][setting_name] = len(result["downsampled_ts"])

                        if "scores" in evaluation_data:
                            evaluation_data["scores"][setting_name] = result["score"]
                        
                        if "execution_times" in evaluation_data:
                            evaluation_data["execution_times"][setting_name] = {
                                "Spikelet": result["transformation_time"],
                                "ClaSP": result["clasp_fit_predict_time"],
                                "Downsampling": result["downsampling_time"]
                            }
                        
                        if "memory_consumption" in evaluation_data:
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
                        
                        if "parameters" in evaluation_data:
                            evaluation_data["parameters"][setting_name] = {
                                "window_size": result["window_size"],
                                "n_for_nth_point": downAlgParam if downAlgName == "nth" else None,
                                "MaT": current_MaT,
                                "CoT": CoT
                            }

                        evaluation_data = convert_arrays_to_lists(evaluation_data)

                        with output_json_file.open('w') as json_file:
                            json.dump(evaluation_data, json_file, indent=4)

                        all_results.clear()
                        
                        current_MaT += step_MaT
                        current_MaT = round(current_MaT, 2)

                        with lock:
                            num_iteration += 1

                    except TimeoutException:
                        print(f"Abbruch nach 2 Minuten Bearbeitungszeit bei MaT={current_MaT}")
                        current_MaT += step_MaT
                        current_MaT = round(current_MaT, 2)
                        CoT += 10
                        CoT = round(CoT)
                        #num_iteration += 1
                        with lock:
                            num_iteration += 1
                        continue
                    
                    except Exception as e:
                        print(f"Fehler bei MaT={current_MaT}: {e}")
                        current_MaT += step_MaT
                        current_MaT = round(current_MaT, 2)
                        CoT += 10
                        CoT = round(CoT)

                        #num_iteration += 1
                        with lock:
                            num_iteration += 1
                        continue

                    # finally:
                    #     signal.alarm(0)  # Disable the timeout alarm
                    #     tracemalloc.stop()

                    finally:
                        tracemalloc.stop()

                        # Sicherstellen, dass die Variablen existieren, bevor sie gelöscht werden
                        if "predicted_change_points" in evaluation_data and setting_name in evaluation_data["predicted_change_points"]:
                            del evaluation_data["predicted_change_points"][setting_name]

                        if "downsampled_time_series_len" in evaluation_data and setting_name in evaluation_data["downsampled_time_series_len"]:
                            del evaluation_data["downsampled_time_series_len"][setting_name]

                        if "scores" in evaluation_data and setting_name in evaluation_data["scores"]:
                            del evaluation_data["scores"][setting_name]

                        if "execution_times" in evaluation_data and setting_name in evaluation_data["execution_times"]:
                            del evaluation_data["execution_times"][setting_name]

                        if "memory_consumption" in evaluation_data and setting_name in evaluation_data["memory_consumption"]:
                            del evaluation_data["memory_consumption"][setting_name]

                        if "parameters" in evaluation_data and setting_name in evaluation_data["parameters"]:
                            del evaluation_data["parameters"][setting_name]

                        # Sicherstellen, dass setting_name existiert, bevor es gelöscht wird
                        try:
                            del setting_name
                        except NameError:
                            pass

                print(f"Ergebnisse für {ts_name} wurden in {output_json_file} gespeichert.")
            else:
                # signal.signal(signal.SIGALRM, timeout_handler)
                # signal.alarm(120)
                try:
                    tracemalloc.start()
                    result = execute_combination(analysis_mode, use_spikelet, downAlgName, downAlgParam, pltshow, ts_name, window_size, cps, ts, order, mewsImplementation)

                    combination_name = (
                        f"[{benchmarkSet}, {order}, mews={mewsImplementation}] "
                        f"Spikelet={use_spikelet}, Downsampling={downAlgName}"
                    )

                    if downAlgParam is not None:
                        combination_name = f"{combination_name}, Param={downAlgParam}"

                    if "predicted_change_points" in evaluation_data:
                        if combination_name not in evaluation_data["predicted_change_points"]: 
                            evaluation_data["predicted_change_points"][combination_name] = []
                        evaluation_data["predicted_change_points"][combination_name].append(result["predicted_cps"])
                    if "downsampled_time_series_len" in evaluation_data:
                        if combination_name not in evaluation_data["downsampled_time_series_len"]: 
                            evaluation_data["downsampled_time_series_len"][combination_name] = []
                        evaluation_data["downsampled_time_series_len"][combination_name].append(len(result["downsampled_ts"]))

                    if "scores" in evaluation_data:
                        if combination_name not in evaluation_data["scores"]: 
                            evaluation_data["scores"][combination_name] = []
                        evaluation_data["scores"][combination_name].append(result["score"])

                    if "execution_times" in evaluation_data:
                        if combination_name not in evaluation_data["execution_times"]: 
                            evaluation_data["execution_times"][combination_name] = {
                                "Spikelet": result["transformation_time"],
                                "ClaSP": result["clasp_fit_predict_time"],
                                "Downsampling": result["downsampling_time"]
                            }
                        else:
                            evaluation_data["execution_times"][combination_name]["Spikelet"] = result["transformation_time"]
                            evaluation_data["execution_times"][combination_name]["ClaSP"] = result["clasp_fit_predict_time"]
                            evaluation_data["execution_times"][combination_name]["Downsampling"] = result["downsampling_time"]
            
                    if "memory_consumption" in evaluation_data:  
                        if combination_name not in evaluation_data["memory_consumption"]: 
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
                                }
                            }  
                        else:
                            evaluation_data["execution_times"][combination_name]["Spikelet"] = {
                                "peak": result["spikelet_memory_peak"],
                                "current": result["spikelet_memory_current"]
                            }
                            evaluation_data["execution_times"][combination_name]["ClaSP"] = {
                                "peak": result["clasp_memory_peak"],
                                "current": result["clasp_memory_current"]
                            }
                            evaluation_data["execution_times"][combination_name]["Downsampling"] = {
                                "peak": result["downsampling_memory_peak"],
                                "current": result["downsampling_memory_current"]
                            }
                        
                    if "parameters" in evaluation_data:  
                        if combination_name not in evaluation_data["parameters"]: 
                            evaluation_data["parameters"][combination_name] = {
                                "window_size": result["window_size"]
                            }   
                        else:
                            evaluation_data["parameters"][combination_name]["window_size"] = result["window_size"]

                except TimeoutException:
                    print(f"Abbruch nach 2 Minuten Bearbeitungszeit für {ts_name}")
                    error_filename = f"AAA_TIMEOUT_{ts_name}.txt"
                    error_filepath = Path(output_dir) / error_filename
                    with error_filepath.open('w') as error_file:
                        error_file.write(f"Timeout für Datensatz {ts_name}\n\n")
                        error_file.write(f"MeWS implementation war aktiviert.")
                    continue    

                except Exception as e:
                    if ts_name == "Computers":
                        print("ich kommt aber trotzdem raus")
                        time.sleep(10)
                    error_filename = f"AAA_FEHLER_{ts_name}.txt"
                    error_filepath = Path(output_dir) / error_filename
                    with error_filepath.open('w') as error_file:
                        error_file.write(f"Fehler für Datensatz {ts_name}:\n{str(e)}\n\n")
                        error_file.write(f"MeWS implementation war aktiviert.")
                    continue

                finally:
                        #signal.alarm(0)  # Disable the timeout alarm
                        tracemalloc.stop()
        
        if mewsImplementation or analysis_mode:
            save_evaluation_data_incremental(evaluation_data, output_dir2, ts_name, len(ts), cps)
            evaluation_data = deepcopy(evaluation_data_template)

        #combined_output_file = os.path.join(output_dir, f"{ts_name}_all_results.txt")
        #save_run_info_txt(combined_output_file, all_results)

        #print(f"Alle Ergebnisse wurden in {combined_output_file} gespeichert.")

if __name__ == "__main__":
    skip = True
    benchmarkSet = "all"                    # tssb, floss oder all
    benchmarkNames = []                     # [] -> alle, "[x, y,...]" -> spezifisch
    mewsImplementation = True              # False: es wird lange dauern haha, True: MeWS bestimmt MaT und CoT dynamisch, sehr viel schneller

    run_settings = [                        # Settings die für den Run gelten, mehrere sind möglich. (Spikelet_true, (downsampling_algorithm, parameter), visualize)
               # Falls alles (also wirklich alles) gemacht werden soll, leer lassen
    ]

    evaluation_data = deepcopy(evaluation_data_template)
    evaluation_data2 = deepcopy(evaluation_data_template)
    evaluation_data3 = deepcopy(evaluation_data_template)
    evaluation_data4 = deepcopy(evaluation_data_template)
    evaluation_data5 = deepcopy(evaluation_data_template)
    evaluation_data6 = deepcopy(evaluation_data_template)
    evaluation_data7 = deepcopy(evaluation_data_template)
    evaluation_data8 = deepcopy(evaluation_data_template)

    order = "spikelet_first"                # spikelet_first oder downsampling_first
    
    # run(False, "tssb", benchmarkNames, mewsImplementation, run_settings, evaluation_data, output_dir, "spikelet_first")
    # run(False, "tssb", benchmarkNames, mewsImplementation, run_settings, evaluation_data2, output_dir, "downsampling_first")
    # run(False, "tssb", benchmarkNames, False, [(True, ("Extrema", 3), False)], evaluation_data3, output_dir, "spikelet_first")
    
       # run("20250220_173304", False, "floss", benchmarkNames, mewsImplementation, run_settings, evaluation_data5, output_dir, "spikelet_first")
       # run("20250220_204725", False, "floss", benchmarkNames, mewsImplementation, run_settings, evaluation_data6, output_dir, "downsampling_first")
    # run(False, "floss", benchmarkNames, False, [(True, ("Extrema", 3), False)], evaluation_data7, output_dir, "spikelet_first")
    
    # run(True, "tssb", benchmarkNames, False, [(True, ("nth", 4), False)], evaluation_data7, output_dir, "spikelet_first")
    #run(True, "floss", benchmarkNames, False, [(False, ("Extrema2", 13), False)], evaluation_data8, output_dir, "spikelet_first")

    run(False, "floss", benchmarkNames, mewsImplementation, [(True, ("Extrema2", 13), False)], evaluation_data8, output_dir, "spikelet_first")
    run(False, "tssb", benchmarkNames, mewsImplementation, [(True, ("nth", 4), False)], evaluation_data8, output_dir, "spikelet_first")