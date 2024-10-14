import sys
import os
import time  # Importieren des time-Moduls für die Zeitmessung
from datetime import datetime

output_dir = r'C:\Users\Victor\Desktop\Uni\Bachelor\output'
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_dir)

from aeon.datasets import (
    load_gun_point_segmentation,
    load_psyllid_reduced_segmentation,
    load_psyllid_segmentation,
)
from aeon.segmentation import ClaSPSegmenter, find_dominant_window_sizes
from aeon.transformations.spikelet.spikelet import motif_discovery_and_clasp

def save_timings_to_file(elapsed_times):
    # Sicherstellen, dass das Ausgabe-Verzeichnis existiert
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Aktuelle Zeit für den Dateinamen
    current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Dateiname mit aktuellem Zeitstempel
    output_file_name = f"timings_{current_time_str}.txt"
    output_file_path = os.path.join(output_dir, output_file_name)
    
    # Schreiben der Zeiten in die Textdatei
    with open(output_file_path, 'w') as f:
        f.write(f"Zeit für motif_discovery_and_clasp(X): {elapsed_times['transformation']:.4f} Sekunden\n")
        f.write(f"Zeit für clasp.fit_predict(X): {elapsed_times['clasp_fit_predict']:.4f} Sekunden\n")
    
    print(f"Zeiten wurden in der Datei {output_file_path} gespeichert.")

    
#X, true_period_size, cps = load_gun_point_segmentation()
X, true_period_size, cps = load_psyllid_reduced_segmentation()
#X, true_period_size, cps = load_psyllid_reduced_segmentation()
print("Datensatz geladen.")
start_time_total = time.time()

start_time_transformation = time.time()
X = motif_discovery_and_clasp(X)
end_time_transformation = time.time()

elapsed_time_transformation = end_time_transformation - start_time_transformation
print(f"Zeit für motif_discovery_and_clasp(X): {elapsed_time_transformation:.4f} Sekunden")

dominant_period_size = find_dominant_window_sizes(X)
clasp = ClaSPSegmenter(dominant_period_size, n_cps=1)

start_time_clasp_fit_predict = time.time()
found_cps = clasp.fit_predict(X)
end_time_clasp_fit_predict = time.time()

elapsed_time_clasp_fit_predict = end_time_clasp_fit_predict - start_time_clasp_fit_predict
print(f"Zeit für clasp.fit_predict(X): {elapsed_time_clasp_fit_predict:.4f} Sekunden")

end_time_total = time.time()
total_time = end_time_total - start_time_total

print("Zeit total: ", total_time)

profiles = clasp.profiles
scores = clasp.scores

elapsed_times = {
    'transformation': elapsed_time_transformation,
    'clasp_fit_predict': elapsed_time_clasp_fit_predict
}

# Aufrufen der Funktion zum Speichern der Zeiten
save_timings_to_file(elapsed_times)
