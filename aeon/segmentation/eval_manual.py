import json
from pathlib import Path

def sum_execution_times_per_dataset(execution_times):
    return {key: sum(methods.values()) for key, methods in execution_times.items()}

def save_top_scores_to_json(output_path, ts_name, max_score, best_entry, execution_times):
    output_directory = Path(output_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_file = output_directory / "top_scores.json"
    
    # Lade vorhandene Daten, falls die Datei existiert
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as file:
            output_data = json.load(file)
    else:
        output_data = {"ts_names": [], "scores": [], "execution_times": [], "parameters": []}
    
    # Werte für den aktuellen Datensatz hinzufügen
    output_data["ts_names"].append(ts_name)
    output_data["scores"].append(max_score)
    output_data["execution_times"].append(execution_times.get(best_entry, 'Unbekannt'))
    output_data["parameters"].append(best_entry)
    
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(output_data, file, indent=4)

def print_top_10_scores(directory_path, ts_names):
    # Konvertiere den Pfad in ein Path-Objekt
    directory = Path(directory_path)
    
    for ts_name in ts_names:
        file_path = directory / f"{ts_name}.json"
        
        if not file_path.exists():
            print(f"Datei {file_path} existiert nicht.")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        scores = {k: data["scores"][k] for k in list(data.get("scores", {}))[:290]}
        execution_times = {k: data["execution_times"][k] for k in list(data.get("execution_times", {}))[:290]}
        
        summed_execution_times = sum_execution_times_per_dataset(execution_times)
        
        sorted_scores = sorted(
            scores.items(),
            key=lambda x: (-x[1], summed_execution_times.get(x[0], float('inf')))
        )
        
        max_score = sorted_scores[0][1] if sorted_scores else None
        top_scores = [entry for entry in sorted_scores if entry[1] == max_score][:10]
        
        if top_scores:
            best_entry = top_scores[0][0]
            save_top_scores_to_json(
                output_path,
                ts_name,
                max_score,
                best_entry,
                summed_execution_times
            )

def print_top_x_scores(directory_path, ts_names, x=10):
    # Konvertiere den Pfad in ein Path-Objekt
    directory = Path(directory_path)
    
    for ts_name in ts_names:
        file_path = directory / f"{ts_name}.json"
        
        if not file_path.exists():
            print(f"Datei {file_path} existiert nicht.")
            continue
        
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        scores = {k: data["scores"][k] for k in list(data.get("scores", {}))[:290]}
        execution_times = {k: data["execution_times"][k] for k in list(data.get("execution_times", {}))[:290]}

        print(scores)
        
        summed_execution_times = sum_execution_times_per_dataset(execution_times)
        
        # Sortiere nach Score (absteigend) und dann nach Laufzeit (aufsteigend)
        sorted_scores = sorted(
            scores.items(),
            key=lambda x: (-x[1], summed_execution_times.get(x[0], float('inf')))
        )
        
        # Bestimme die höchsten einzigartigen Accuracies
        unique_scores = []
        top_scores = []
        for entry in sorted_scores:
            if entry[1] not in unique_scores:
                unique_scores.append(entry[1])
                top_scores.append(entry)
            if len(top_scores) >= x:
                break
        
        # Ausgabe der besten einzigartigen Einträge
        print(f"Top {len(top_scores)} Scores für {ts_name}:")
        for key, value in top_scores:
            exec_time = summed_execution_times.get(key, 'Unbekannt')
            print(f"{key}: Score={value}, Laufzeit={exec_time}")
        print("-" * 40)  # Trennlinie für bessere Lesbarkeit

ts_names_tssb = [
        "Adiac",
        "ArrowHead",
        "Beef",
        "BeetleFly",
        "BirdChicken",
        "Car",
        "CBF",
        "ChlorineConcentration",
        "CinCECGTorso",
        "Coffee",
        "Computers",
        "CricketX",
        "CricketY",
        "CricketZ",
        "DiatomSizeReduction",
        "DistalPhalanxOutlineAgeGroup",
        "DistalPhalanxTW",
        "ECG200",
        "ECGFiveDays",
        "FaceAll",
        "FaceFour",
        "FacesUCR",
        "FiftyWords",
        "Fish",
        "GunPoint",
        "Haptics",
        "InlineSkate",
        "InsectWingbeatSound",
        "ItalyPowerDemand",
        "LargeKitchenAppliances",
        "Lightning2",
        "Lightning7",
        "Mallat",
        "Meat",
        "MedicalImages",
        "MoteStrain",
        "NonInvasiveFetalECGThorax1",
        "NonInvasiveFetalECGThorax2",
        "OliveOil",
        "OSULeaf",
        "Plane",
        "ProximalPhalanxOutlineAgeGroup",
        "ProximalPhalanxTW",
        "ShapesAll",
        "SonyAIBORobotSurface1",
        "SonyAIBORobotSurface2",
        "SwedishLeaf",
        "Symbols",
        "SyntheticControl",
        "ToeSegmentation1",
        "ToeSegmentation2",
        "Trace",
        "TwoLeadECG",
        "UWaveGestureLibraryAll",
        "UWaveGestureLibraryX",
        "UWaveGestureLibraryY",
        "UWaveGestureLibraryZ",
        "WordSynonyms",
        "Yoga",
        "Crop",
        "EOGHorizontalSignal",
        "EOGVerticalSignal",
        "FreezerRegularTrain",
        "Ham",
        "MelbournePedestrian",
        "MiddlePhalanxOutlineAgeGroup",
        "MiddlePhalanxOutlineCorrect",
        "ProximalPhalanxOutlineCorrect",
        "Strawberry",
        "Chinatown",
        "DodgerLoopDay",
        "Herring",
        "MiddlePhalanxTW",
        "ShapeletSim",
        "UMD"
    ]
ts_names_floss = [
        "Cane",
        "DutchFactory",
        "EEGRat",
        "EEGRat2",
        "Fetal2013",
        "GrandMalSeizures",
        "GrandMalSeizures2",
        "GreatBarbet1",
        "GreatBarbet2",
        "InsectEPG1",
        "InsectEPG2",
        "InsectEPG3",
        "NogunGun",
        "PigInternalBleedingDatasetAirwayPressure",
        "PigInternalBleedingDatasetArtPressureFluidFilled",
        "PigInternalBleedingDatasetCVP",
        "Powerdemand",
        "PulsusParadoxusECG1",
        "PulsusParadoxusECG2",
        "PulsusParadoxusSP02",
        "RoboticDogActivityX",
        "RoboticDogActivityY",
        "RoboticDogActivityZ",
        "SimpleSynthetic",
        "SuddenCardiacDeath1",
        "SuddenCardiacDeath2",
        "SuddenCardiacDeath3",
        "TiltABP",
        "TiltECG",
        "WalkJogRun1",
        "WalkJogRun2"
    ]

output_path = r"C:\Users\Victor\Desktop\Uni\Bachelor\optimal_output\tssb"
directory_path = r"C:\Users\Victor\Desktop\Uni\Bachelor\output\nth_tssb_detailliert\benchmark_jsons"
print_top_10_scores(directory_path, ts_names_tssb)
#print_top_x_scores(directory_path, ts_names_tssb)

# output_path = r"C:\Users\Victor\Desktop\Uni\Bachelor\optimal_output\floss"
# directory_path = r"C:\Users\Victor\Desktop\Uni\Bachelor\output\Run_20250217_232811\benchmark_jsons"
# print_top_10_scores(directory_path, ts_names_floss)


# Run_20250205_080311 - tssb benchmark
# Run_20250209_190105 - floss benchmark