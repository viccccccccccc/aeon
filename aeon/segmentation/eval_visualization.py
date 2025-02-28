import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re

from calculations_for_evaluation import sum_execution_times_per_dataset, sum_execution_times_per_dataset2, t_test, wilcoxon_test, calc_average, calc_median

def show_boxplot(values, values2, values3, values4):
    data = [values, values2, values3, values4]
    labels = ["w_e=w*0.35", "w_e=w*0.45", "w_e=w*0.5", "w_e=w*0.75"]  # Passe die Labels an deine Daten an

    for label, vals in zip(labels, data):
        avg_scores = calc_average(vals)
        median_scores = calc_median(vals)
        print(f"{label}: average score: {avg_scores}")
        print(f"{label}: median score: {median_scores}")

    # Boxplot erstellen
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, patch_artist=True, boxprops=dict(facecolor="lightgray"))

    # Details hinzufügen
    plt.title("Distribution of Scores", fontsize=14)
    plt.xlabel("Extrema Variants", fontsize=12)
    plt.ylabel("Scores", fontsize=12)

    # Gitter hinzufügen und anzeigen
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_execution_times(dataset_names, times1, times2, times3, times4,
                         key_strings, curve_labels):
    """
    Plots a line chart for four lists of execution times with dataset names on the x-axis.
    
    Parameters:
    - dataset_names (list): List of dataset names (x-axis labels).
    - times1 (dict): First dataset's execution times (after summing).
    - times2 (dict): Second dataset's execution times (after summing).
    - times3 (dict): Third dataset's execution times (after summing).
    - times4 (dict): Fourth dataset's execution times (after summing).
    - key_strings (list): List of strings for extracting specific execution times from each dataset.
                          (e.g., ["[tssb,...] Param=5", "[tssb,...] Param=12", ...])
    - curve_labels (list): List of labels for each curve (e.g., ["n=5", "n=12", ...])
    """
    # Vorverarbeitung der Ausführungszeiten
    datasets = [
        sum_execution_times_per_dataset2(times)
        for times in [times1, times2, times3, times4]
    ]
    
    x = np.arange(len(dataset_names))  # X-Achsen-Positionen für Datensätze

    fig, ax = plt.subplots(figsize=(15, 8))

    # Jede Kurve mit passendem Key-String und Label zeichnen
    for i, (key, label) in enumerate(zip(key_strings, curve_labels)):
        # Wähle die Farben konsistent für jedes Paar (z.B. gleiche Farbe für gleiche Parameter)
        color = f"C{i // 2}"  # Farbzyklus basierend auf Paar-Index

        ax.plot(x, datasets[i][:len(x)],  # Kürze die Liste falls nötig
        label=label, linestyle='-', linewidth=2, color=f"C{i}")


    # Rasterlinien (horizontal)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    # Achsentitel und Beschriftungen
    ax.set_xlabel("Datasets (ordered by length)", fontsize=12)
    ax.set_ylabel("Execution Time (s)", fontsize=12)
    ax.set_title("Execution Times per Dataset", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=45, fontsize=8, ha="right")
    ax.tick_params(axis='y', labelsize=12)
    ax.legend()

    # Layout-Anpassung und Plot anzeigen
    plt.tight_layout()
    plt.show()



def both_boxplots(data1, data2, data3, data4, data5, data6, data7, data8):
    # Daten gruppieren: erste 4 für den ersten Plot, letzte 4 für den zweiten Plot
    plot1_data = [data1, data2, data3, data4]
    plot2_data = [data5, data6, data7, data8]
    
    # Labels für die Kategorien
    labels1 = ['ClaSP', 'DEL: n=2', 'DEL: n=6', 'M2: w_e=3']  # Labels für den ersten Plot
    labels2 = ['ClaSP', 'MEAN: n=5', 'MEAN: n=7', 'M2: w_e=11']  # Labels für den zweiten Plot

    for label, vals in zip(labels1, plot1_data):
        avg_scores = calc_average(vals)
        median_scores = calc_median(vals)
        print(f"{label}: average score: {avg_scores}")
        print(f"{label}: median score: {median_scores}")

    for label, vals in zip(labels2, plot2_data):
        avg_scores = calc_average(vals)
        median_scores = calc_median(vals)
        print(f"{label}: average score: {avg_scores}")
        print(f"{label}: median score: {median_scores}")

    # Subplots erstellen
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)  # 1 Reihe, 2 Spalten, gleiche y-Achse teilen

    axes[0].boxplot(plot1_data, patch_artist=True, boxprops=dict(facecolor="lightgrey"))
    axes[0].set_title('TSSB', fontsize=15)  # Titel größer machen
    axes[0].set_xticks(range(1, len(labels1) + 1))
    axes[0].set_xticklabels(labels1, rotation=45, fontsize=12)
    axes[0].set_ylabel('Score', fontsize=14)  # Y-Achsen-Beschriftung größer machen
    axes[0].tick_params(axis='y', labelsize=12)  # 🔹 Y-Achsen-Zahlen größer machen
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)  # Horizontales Grid

    # Boxplots für den zweiten Plot
    axes[1].boxplot(plot2_data, patch_artist=True, boxprops=dict(facecolor="lightgrey"))
    axes[1].set_title('FLOSS', fontsize=15)  # Titel größer machen
    axes[1].set_xticks(range(1, len(labels2) + 1))
    axes[1].set_xticklabels(labels2, rotation=45, fontsize=12)
    axes[1].tick_params(axis='y', labelsize=12)  # 🔹 Y-Achsen-Zahlen größer machen
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)  # Horizontales Grid

    # Layout anpassen und anzeigen
    plt.tight_layout()
    plt.show()

def both_boxplots_extended(data1, data2, data3, data4, data5, data6, 
                           data7, data8, data9, data10, data11, data12):
    """
    Erstellt zwei nebeneinander liegende Boxplots mit jeweils 6 Kategorien.
    
    Parameters:
    - data1 bis data12: Listen mit Scores für jede Kategorie.
    """
    # Daten gruppieren: erste 6 für den ersten Plot, letzte 6 für den zweiten Plot
    plot1_data = [data1, data2, data3, data4, data5, data6]
    plot2_data = [data7, data8, data9, data10, data11, data12]
    
    # Labels für die Kategorien
    labels1 = ['ClaSP', 'DEL: n=6', 'M2: w_e=3', 'DEL: n=4', 'M3: w_e=4', 'DEL: n=4']  # Labels für den ersten Plot
    labels2 = ['ClaSP', 'MEAN: n=4', 'M3: n=16', 'MEAN: n=5', 'M3: w_e=13', 'M3: w_e=13']  # Labels für den zweiten Plot

    for label, vals in zip(labels1, plot1_data):
        avg_scores = calc_average(vals)
        median_scores = calc_median(vals)
        print(f"{label}: average score: {avg_scores}")
        print(f"{label}: median score: {median_scores}")

    for label, vals in zip(labels2, plot2_data):
        avg_scores = calc_average(vals)
        median_scores = calc_median(vals)
        print(f"{label}: average score: {avg_scores}")
        print(f"{label}: median score: {median_scores}")

    # Subplots erstellen
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)  # 1 Reihe, 2 Spalten, gleiche y-Achse teilen

    # Boxplots für den ersten Plot
    axes[0].boxplot(plot1_data, patch_artist=True, boxprops=dict(facecolor="lightgrey"))
    axes[0].set_title('TSSB', fontsize=15)
    axes[0].set_xticks(range(1, len(labels1) + 1))
    axes[0].set_xticklabels(labels1, rotation=45, fontsize=12)
    axes[0].set_ylabel('Score', fontsize=14)
    axes[0].tick_params(axis='y', labelsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)  # Horizontales Grid

    # Boxplots für den zweiten Plot
    axes[1].boxplot(plot2_data, patch_artist=True, boxprops=dict(facecolor="lightgrey"))
    axes[1].set_title('FLOSS', fontsize=15)
    axes[1].set_xticks(range(1, len(labels2) + 1))
    axes[1].set_xticklabels(labels2, rotation=45, fontsize=12)
    axes[1].tick_params(axis='y', labelsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)  # Horizontales Grid
    
    # Layout anpassen und anzeigen
    plt.tight_layout()
    plt.show()


def scatter_num_change_points(data, scores):
    true_change_points_counts = [len(cps) for cps in data["true_change_points"]]

    # Scatterplot erstellen
    plt.figure(figsize=(8, 6))
    plt.scatter(true_change_points_counts, scores, color="blue", alpha=0.7, edgecolor="black")

    # Achsentitel und Diagrammtitel
    plt.xlabel("Number of Change Points")
    plt.ylabel("Covering Score (%)")
    plt.grid(alpha=0.3)

    # Anzeigen
    plt.show()

def scatter_ts_length_vs_runtime_multi(data1_a, runtimes1_a, label1_a, 
                                            data2_a, runtimes2_a, label2_a, 
                                            data3_a, runtimes3_a, label3_a,
                                            data1_b, runtimes1_b, label1_b, 
                                            data2_b, runtimes2_b, label2_b, 
                                            data3_b, runtimes3_b, label3_b):
    """
    Erstellt zwei Scatterplots nebeneinander:
    - Linker Plot für drei Datensätze (A)
    - Rechter Plot für drei Datensätze (B)

    Parameters:
    - data1_a, data2_a, data3_a, data1_b, data2_b, data3_b (dict): Enthalten "original_time_series_len" als Liste der Längen.
    - runtimes1_a, runtimes2_a, runtimes3_a, runtimes1_b, runtimes2_b, runtimes3_b (list): Listen der gemessenen Laufzeiten pro Zeitreihe.
    - label1_a, label2_a, label3_a, label1_b, label2_b, label3_b (str): Labels für die Datensätze.
    """
    ts_lengths1_a = data1_a["original_time_series_len"]
    ts_lengths2_a = data2_a["original_time_series_len"]
    ts_lengths3_a = data3_a["original_time_series_len"]
    
    runtimes1_a = sum_execution_times_per_dataset2(runtimes1_a)
    runtimes2_a = sum_execution_times_per_dataset2(runtimes2_a)
    runtimes3_a = sum_execution_times_per_dataset2(runtimes3_a)
    
    ts_lengths1_b = data1_b["original_time_series_len"]
    ts_lengths2_b = data2_b["original_time_series_len"]
    ts_lengths3_b = data3_b["original_time_series_len"]
    
    runtimes1_b = sum_execution_times_per_dataset2(runtimes1_b)
    runtimes2_b = sum_execution_times_per_dataset2(runtimes2_b)
    runtimes3_b = sum_execution_times_per_dataset2(runtimes3_b)
    
    # Scatterplots erstellen
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Linker Plot (A)
    axes[0].scatter(ts_lengths1_a, runtimes1_a, color="blue", alpha=0.7, edgecolor="black", label=label1_a)
    axes[0].scatter(ts_lengths2_a, runtimes2_a, color="red", alpha=0.7, edgecolor="black", label=label2_a)
    axes[0].scatter(ts_lengths3_a, runtimes3_a, color="green", alpha=0.7, edgecolor="black", label=label3_a)
    axes[0].set_xlabel("Time Series Length")
    axes[0].set_ylabel("Runtime (s)")
    axes[0].set_title("Dataset Group A")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    
    # Rechter Plot (B)
    axes[1].scatter(ts_lengths1_b, runtimes1_b, color="blue", alpha=0.7, edgecolor="black", label=label1_b)
    axes[1].scatter(ts_lengths2_b, runtimes2_b, color="red", alpha=0.7, edgecolor="black", label=label2_b)
    axes[1].scatter(ts_lengths3_b, runtimes3_b, color="green", alpha=0.7, edgecolor="black", label=label3_b)
    axes[1].set_xlabel("Time Series Length")
    axes[1].set_title("Dataset Group B")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    
    # Layout anpassen und anzeigen
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

def scatter_ts_length_vs_runtime_multi2(data1, runtimes1, label1, 
                                       data2, runtimes2, label2, 
                                       data3, runtimes3, label3,
                                       data4, runtimes4, label4,
                                       data5, runtimes5, label5,
                                       data6, runtimes6, label6,
                                       data7, runtimes7, label7,
                                       data8, runtimes8, label8):
    """
    Erstellt vier Scatterplots:
    - Oben links: Datensatz 1 & 2 (TSSB)
    - Oben rechts: Datensatz 5 & 6 (FLOSS)
    - Unten links: Datensatz 1, 3 & 4 (TSSB)
    - Unten rechts: Datensatz 5, 7 & 8 (FLOSS)

    Parameters:
    - data1, ..., data8 (dict): Enthalten "original_time_series_len" als Liste der Längen.
    - runtimes1, ..., runtimes8 (list): Listen der gemessenen Laufzeiten pro Zeitreihe.
    - label1, ..., label8 (str): Labels für die Datensätze.
    """
    # Extrahiere die Zeitreihenlängen
    ts_lengths = [data["original_time_series_len"] for data in [data1, data2, data3, data4, data5, data6, data7, data8]]

    # Laufzeiten summieren (falls notwendig)
    runtimes = [sum_execution_times_per_dataset2(rt) for rt in [runtimes1, runtimes2, runtimes3, runtimes4, runtimes5, runtimes6, runtimes7, runtimes8]]

    # Scatterplot-Farben
    colors = ["orange", "green", "red"]

    # 2x2 Subplots erstellen (TSSB x-achsen teilen, FLOSS x-achsen teilen)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex="col", sharey=True)

    # 🔹 Plot 1: Datensatz 1 & 2 (Oben links - TSSB)
    axes[0, 0].scatter(ts_lengths[0], runtimes[0], color=colors[0], alpha=0.7, edgecolor="black", label=label1)
    axes[0, 0].scatter(ts_lengths[1], runtimes[1], color=colors[1], alpha=0.7, edgecolor="black", label=label2)
    axes[0, 0].set_title("TSSB: MeWS vs. BL")
    axes[0, 0].set_ylabel("Runtime (s)")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    # 🔹 Plot 2: Datensatz 5 & 6 (Oben rechts - FLOSS)
    axes[0, 1].scatter(ts_lengths[4], runtimes[4], color=colors[0], alpha=0.7, edgecolor="black", label=label5)
    axes[0, 1].scatter(ts_lengths[5], runtimes[5], color=colors[1], alpha=0.7, edgecolor="black", label=label6)
    axes[0, 1].set_title("FLOSS: MeWS vs. BL")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    # 🔹 Plot 3: Datensatz 1, 3 & 4 (Unten links - TSSB)
    axes[1, 0].scatter(ts_lengths[0], runtimes[0], color=colors[0], alpha=0.7, edgecolor="black", label=label1)
    axes[1, 0].scatter(ts_lengths[2], runtimes[2], color=colors[1], alpha=0.7, edgecolor="black", label=label3)
    axes[1, 0].scatter(ts_lengths[3], runtimes[3], color=colors[2], alpha=0.7, edgecolor="black", label=label4)
    axes[1, 0].set_title("TSSB")
    axes[1, 0].set_xlabel("Time Series Length")
    axes[1, 0].set_ylabel("Runtime (s)")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    # 🔹 Plot 4: Datensatz 5, 7 & 8 (Unten rechts - FLOSS)
    axes[1, 1].scatter(ts_lengths[4], runtimes[4], color=colors[0], alpha=0.7, edgecolor="black", label=label5)
    axes[1, 1].scatter(ts_lengths[6], runtimes[6], color=colors[1], alpha=0.7, edgecolor="black", label=label7)
    axes[1, 1].scatter(ts_lengths[7], runtimes[7], color=colors[2], alpha=0.7, edgecolor="black", label=label8)
    axes[1, 1].set_title("FLOSS")
    axes[1, 1].set_xlabel("Time Series Length")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    # Layout anpassen und anzeigen
    plt.tight_layout()
    plt.show()






def plot_downsampling_nth(data1, data2, data3, data4):
    """
    Plottet zwei Diagramme:
    - Erster Graph: data1 ("Spikelet zuerst") und data3 ("Downsampling zuerst"), sowie "nth" und "nth2"
    - Zweiter Graph: data2 ("Spikelet zuerst") und data4 ("Downsampling zuerst"), sowie "nth" und "nth2"
    """
    def extract_param_and_avg_scores_nth(data, downsampling_pattern):
        """Extrahiert {Param: Durchschnitts-Score} für den gegebenen Downsampling-Typ."""
        param2avg = {}
        for combo_name, score_list in data["scores"].items():
            if downsampling_pattern not in combo_name or "Spikelet=True" not in combo_name:
                continue

            match = re.search(r"Param=(\d+)", combo_name)
            if match:
                param = int(match.group(1))
            else:
                continue

            if 2 <= param <= 20:
                avg_score = sum(score_list) / len(score_list)
                param2avg[param] = avg_score

        return param2avg

    def plot_four_datasets(ax, data_a, label_a, data_b, label_b, title):
        """Plottet vier Kurven für nth und nth2 für beide Vergleichspaare."""
        p2avg_nth_a = extract_param_and_avg_scores_nth(data_a, "Downsampling=nth,")
        p2avg_nth_b = extract_param_and_avg_scores_nth(data_b, "Downsampling=nth,")
        p2avg_nth2_a = extract_param_and_avg_scores_nth(data_a, "Downsampling=nth2,")
        p2avg_nth2_b = extract_param_and_avg_scores_nth(data_b, "Downsampling=nth2,")

        all_params = sorted(set(p2avg_nth_a.keys()) | set(p2avg_nth_b.keys()) | set(p2avg_nth2_a.keys()) | set(p2avg_nth2_b.keys()))
        y_nth_a = [p2avg_nth_a.get(p, None) for p in all_params]
        y_nth_b = [p2avg_nth_b.get(p, None) for p in all_params]
        y_nth2_a = [p2avg_nth2_a.get(p, None) for p in all_params]
        y_nth2_b = [p2avg_nth2_b.get(p, None) for p in all_params]

        ax.plot(all_params, y_nth_a, marker='o', color='tab:red', label=f'NTH-DEL ({label_a})')
        ax.plot(all_params, y_nth_b, marker='o', color='tab:orange', label=f'NTH-DEL ({label_b})')  # Grün
        ax.plot(all_params, y_nth2_a, marker='s', color='tab:blue', label=f'NTH-MEAN ({label_a})')  # Rot
        ax.plot(all_params, y_nth2_b, marker='s', color='tab:green', label=f'NTH-MEAN ({label_b})')  # Lila


        ax.set_title(title, fontsize=14)  # 🔹 Größerer Titel
        ax.set_xlabel("n-Values (2 to 15)", fontsize=12)  # 🔹 Größere Achsentitel
        ax.set_ylabel("Average Score", fontsize=12)  # 🔹 Größere Achsentitel
        ax.tick_params(axis='x', labelsize=11)  # 🔹 Größere X-Achsen-Werte
        ax.tick_params(axis='y', labelsize=11)  # 🔹 Größere Y-Achsen-Werte
        ax.legend(fontsize=10)  # 🔹 Kleinere Legende für bessere Übersicht
        ax.grid(True)

    # 1x2 Subplots erstellen
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Erster Graph: data1 vs. data3 mit nth und nth2
    plot_four_datasets(
        axs[0], 
        data1, "Spikelet first", 
        data3, "Downsampling first",
        "TSSB"
    )

    # Zweiter Graph: data2 vs. data4 mit nth und nth2
    plot_four_datasets(
        axs[1], 
        data2, "Spikelet first", 
        data4, "Downsampling first",
        "FLOSS"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def plot_downsampling_extrema(data1, data2, data3, data4):
    """
    Erstellt zwei Diagramme:
      1) Extrema und Extrema2 für data1 & data3
      2) Extrema und Extrema2 für data2 & data4
    """

    def get_param2avg(data, downsampling_pattern):
        """Extrahiert {Param: Durchschnitts-Score} für einen Downsampling-Algorithmus."""
        param2avg = {}
        for combo_name, scores_list in data["scores"].items():
            if downsampling_pattern not in combo_name or "Spikelet=True" not in combo_name:
                continue

            match = re.search(r"Param=(\d+)", combo_name)
            if not match:
                continue

            param = int(match.group(1))
            if 2 <= param <= 20:
                avg_score = sum(scores_list) / len(scores_list)
                param2avg[param] = avg_score

        return param2avg

    def plot_four_lines(ax, data_a, label_a, data_b, label_b):
        """Plottet Extrema- und Extrema2-Linien für zwei Datensätze in einem Diagramm."""
        # Extrema
        p2avg_extrema_a = get_param2avg(data_a, "Downsampling=Extrema,")
        p2avg_extrema_b = get_param2avg(data_b, "Downsampling=Extrema,")

        # Extrema2
        p2avg_extrema2_a = get_param2avg(data_a, "Downsampling=Extrema2,")
        p2avg_extrema2_b = get_param2avg(data_b, "Downsampling=Extrema2,")

        # Gemeinsame X-Werte
        all_params_extrema = sorted(set(p2avg_extrema_a.keys()) | set(p2avg_extrema_b.keys()))
        all_params_extrema2 = sorted(set(p2avg_extrema2_a.keys()) | set(p2avg_extrema2_b.keys()))

        # Linien plotten
        ax.plot(all_params_extrema, 
                [p2avg_extrema_a.get(p, None) for p in all_params_extrema], 
                marker='o', color='tab:red', label=f'EXT-M2 (Spikelet first)')

        ax.plot(all_params_extrema, 
                [p2avg_extrema_b.get(p, None) for p in all_params_extrema], 
                marker='o', color='tab:orange', label=f'EXT-M2 (Downsampling first)')

        ax.plot(all_params_extrema2, 
                [p2avg_extrema2_a.get(p, None) for p in all_params_extrema2], 
                marker='s', color='tab:blue', label=f'EXT-M3 (Spikelet first)')

        ax.plot(all_params_extrema2, 
                [p2avg_extrema2_b.get(p, None) for p in all_params_extrema2], 
                marker='s', color='tab:green', label=f'EXT-M3 (Downsampling first)')

        # Achsentitel und Grid
        ax.set_xlabel("w_e-Values (3 to 19)", fontsize=12)  # 🔹 Größere Achsentitel
        ax.set_ylabel("Average Score", fontsize=13)  # 🔹 Größere Achsentitel
        ax.tick_params(axis='x', labelsize=12)  # 🔹 Größere X-Achsen-Werte
        ax.tick_params(axis='y', labelsize=12)  # 🔹 Größere Y-Achsen-Werte
        ax.legend(fontsize=10)  # 🔹 Kleinere Legende für bessere Übersicht
        ax.grid(True)

    # ------------------------
    # Plot erstellen
    # ------------------------
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Erster Graph: Data1 & Data3
    axs[0].set_title("TSSB", fontsize=14)
    plot_four_lines(axs[0], data1, "Data1", data3, "Data3")

    # Zweiter Graph: Data2 & Data4
    axs[1].set_title("FLOSS", fontsize=14)
    plot_four_lines(axs[1], data2, "Data2", data4, "Data4")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def calculate_average_execution_times(data):
    """
    Berechnet den Durchschnitt der Ausführungszeiten für alle vorhandenen Einstellungen (Keys).
    
    Parameters:
    - data (dict): Dictionary mit "execution_times", das die Runtimes pro Key enthält.
    
    Returns:
    - dict: {key_string: average_execution_time}
    """
    averages = {}
    for key, runtimes in data["execution_times"].items():
        if isinstance(runtimes, list) and all(isinstance(entry, dict) for entry in runtimes):
            # Addiere alle Zeiten aus den verschachtelten Dictionaries
            total_time = 0
            total_count = 0
            for entry in runtimes:
                total_time += sum(entry.values())  # Alle Werte im Dictionary summieren
                total_count += 1
            avg_runtime = total_time / total_count if total_count > 0 else 0
            averages[key] = avg_runtime
        else:
            averages[key] = None  # Kein Wert oder falsches Format
    return averages

def plot_average_execution_times_nth(times1, times2, times3, times4):
    """
    Plottet die durchschnittlichen Ausführungszeiten für vier Datensätze,
    aufgeteilt in zwei nebeneinander liegende Subplots:
      - Links: TSSB-Datensätze (nth + nth2, Spikelet zuerst & Downsampling zuerst)
      - Rechts: FLOSS-Datensätze (nth + nth2, Spikelet zuerst & Downsampling zuerst)
    
    Parameters:
    - times1, times2, times3, times4 (dict): Runtimes pro Datensatz.
    """
    avg_times1 = calculate_average_execution_times(times1)
    avg_times2 = calculate_average_execution_times(times2)
    avg_times3 = calculate_average_execution_times(times3)
    avg_times4 = calculate_average_execution_times(times4)

    def extract_param(key):
        match = re.search(r"Param=(\d+)", key)
        return int(match.group(1)) if match else None

    def sort_and_extract_values(avg_times, downsampling_type):
        """Filtert und sortiert Werte nach `Param=X`, basierend auf dem Downsampling-Typ."""
        valid_keys = {key: value for key, value in avg_times.items()
                      if extract_param(key) is not None and f"Downsampling={downsampling_type}" in key}
        
        unique_params = {}
        for key, value in valid_keys.items():
            param = extract_param(key)
            if param not in unique_params:
                unique_params[param] = value
        
        sorted_params = sorted(unique_params.keys())
        return sorted_params, [unique_params[param] for param in sorted_params]

    # Werte für TSSB extrahieren
    params_tssb_nth_s, values_tssb_nth_s = sort_and_extract_values(avg_times1, "nth")
    params_tssb_nth_d, values_tssb_nth_d = sort_and_extract_values(avg_times2, "nth")
    params_tssb_nth2_s, values_tssb_nth2_s = sort_and_extract_values(avg_times1, "nth2")
    params_tssb_nth2_d, values_tssb_nth2_d = sort_and_extract_values(avg_times2, "nth2")

    unique_params_tssb = sorted(set(params_tssb_nth_s) | set(params_tssb_nth_d) |
                                set(params_tssb_nth2_s) | set(params_tssb_nth2_d))
    x_tssb = np.arange(len(unique_params_tssb))
    x_labels_tssb = [f"{param}" for param in unique_params_tssb]

    # Werte für FLOSS extrahieren
    params_floss_nth_s, values_floss_nth_s = sort_and_extract_values(avg_times3, "nth")
    params_floss_nth_d, values_floss_nth_d = sort_and_extract_values(avg_times4, "nth")
    params_floss_nth2_s, values_floss_nth2_s = sort_and_extract_values(avg_times3, "nth2")
    params_floss_nth2_d, values_floss_nth2_d = sort_and_extract_values(avg_times4, "nth2")

    unique_params_floss = sorted(set(params_floss_nth_s) | set(params_floss_nth_d) |
                                 set(params_floss_nth2_s) | set(params_floss_nth2_d))
    x_floss = np.arange(len(unique_params_floss))
    x_labels_floss = [f"{param}" for param in unique_params_floss]

    # ------------------ 🎨 Plot erstellen ------------------ #
    plt.figure(figsize=(14, 6))

    # 1️⃣ Linker Plot: TSSB
    plt.subplot(1, 2, 1)
    plt.plot(x_tssb, values_tssb_nth_s, 'o-', color='tab:red', label="NTH-DEL (Spikelet first)", markersize=6)
    plt.plot(x_tssb, values_tssb_nth_d, 'o-', color='tab:orange', label="NTH-DEL (Downsampling first)", markersize=6)
    plt.plot(x_tssb, values_tssb_nth2_s, 's-', color='tab:blue', label="NTH-MEAN (Spikelet first)", markersize=6)
    plt.plot(x_tssb, values_tssb_nth2_d, 's-', color='tab:green', label="NTH-MEAN (Downsampling first)", markersize=6)
    
    plt.title("TSSB", fontsize=14)
    plt.xlabel("n-Values (2 to 15)", fontsize=12)
    plt.ylabel("Average Execution Time (s)", fontsize=12)
    plt.xticks(x_tssb, x_labels_tssb, fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # 2️⃣ Rechter Plot: FLOSS
    plt.subplot(1, 2, 2)
    plt.plot(x_floss, values_floss_nth_s, 'o-', color='tab:red', label="NTH-DEL (Spikelet first)", markersize=6)
    plt.plot(x_floss, values_floss_nth_d, 'o-', color='tab:orange', label="NTH-DEL (Downsampling first)", markersize=6)
    plt.plot(x_floss, values_floss_nth2_s, 's-', color='tab:blue', label="NTH-MEAN (Spikelet first)", markersize=6)
    plt.plot(x_floss, values_floss_nth2_d, 's-', color='tab:green', label="NTH-MEAN (Downsampling first)", markersize=6)
    
    plt.title("FLOSS", fontsize=14)
    plt.xlabel("n-Values (2 to 15)", fontsize=12)
    plt.ylabel("Average Execution Time (s)", fontsize=12)
    plt.xticks(x_floss, x_labels_floss, fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_average_execution_times_extrema(times1, times2, times3, times4):
    """
    Erstellt zwei nebeneinander liegende Subplots:
      - Links: TSSB-Datensätze (Extrema + Extrema2)
      - Rechts: FLOSS-Datensätze (Extrema + Extrema2)
    Mit allen Kombinationen: Spikelet zuerst & Downsampling zuerst.

    Parameters:
    - times1, times2, times3, times4 (dict): Runtimes pro Datensatz.
    """
    avg_times1 = calculate_average_execution_times(times1)  # TSSB: Spikelet first
    avg_times2 = calculate_average_execution_times(times2)  # TSSB: Downsampling first
    avg_times3 = calculate_average_execution_times(times3)  # FLOSS: Spikelet first
    avg_times4 = calculate_average_execution_times(times4)  # FLOSS: Downsampling first

    def extract_param(key):
        match = re.search(r"Param=(\d+)", key)
        return int(match.group(1)) if match else None

    def sort_and_extract_values(avg_times, downsampling_type, spikelet_first=True):
        """Sortiert die Keys nach `Param=X` und entfernt doppelte Einträge."""
        spikelet_filter = "Spikelet=True" if spikelet_first else "Spikelet=False"
        valid_keys = {
            key: value for key, value in avg_times.items()
            if extract_param(key) is not None and
               spikelet_filter in key and
               f"Downsampling={downsampling_type}" in key
        }

        unique_params = {}
        for key, value in valid_keys.items():
            param = extract_param(key)
            if param not in unique_params:
                unique_params[param] = value

        sorted_params = sorted(unique_params.keys())
        return sorted_params, [unique_params[param] for param in sorted_params]

    # Werte & X-Achsen-Daten generieren (TSSB)
    params_tssb_e_s, values_tssb_e_s = sort_and_extract_values(avg_times1, "Extrema,", True)
    params_tssb_e_d, values_tssb_e_d = sort_and_extract_values(avg_times2, "Extrema,", False)
    params_tssb_e2_s, values_tssb_e2_s = sort_and_extract_values(avg_times1, "Extrema2,", True)
    params_tssb_e2_d, values_tssb_e2_d = sort_and_extract_values(avg_times2, "Extrema2,", False)

    unique_params_tssb = sorted(set(params_tssb_e_s) | set(params_tssb_e_d) |
                                set(params_tssb_e2_s) | set(params_tssb_e2_d))
    x_tssb = np.arange(len(unique_params_tssb))
    x_labels_tssb = [f"{param}" for param in unique_params_tssb]

    # Werte & X-Achsen-Daten generieren (FLOSS)
    params_floss_e_s, values_floss_e_s = sort_and_extract_values(avg_times3, "Extrema,", True)
    params_floss_e_d, values_floss_e_d = sort_and_extract_values(avg_times4, "Extrema,", False)
    params_floss_e2_s, values_floss_e2_s = sort_and_extract_values(avg_times3, "Extrema2,", True)
    params_floss_e2_d, values_floss_e2_d = sort_and_extract_values(avg_times4, "Extrema2,", False)

    unique_params_floss = sorted(set(params_floss_e_s) | set(params_floss_e_d) |
                                 set(params_floss_e2_s) | set(params_floss_e2_d))
    x_floss = np.arange(len(unique_params_floss))
    x_labels_floss = [f"{param}" for param in unique_params_floss]

    # ------------------ 🎨 Plot erstellen ------------------ #
    plt.figure(figsize=(14, 6))

    # 1️⃣ Linker Plot: TSSB
    plt.subplot(1, 2, 1)
    plt.plot(x_tssb, values_tssb_e_s, 'o-', color='tab:red', label="EXT-M2 (Spikelet first)", markersize=6)
    plt.plot(x_tssb, values_tssb_e_d, 'o-', color='tab:orange', label="EXT-M2 (Downsampling first)", markersize=6)
    plt.plot(x_tssb, values_tssb_e2_s, 's-', color='tab:blue', label="EXT-M3 (Spikelet first)", markersize=6)
    plt.plot(x_tssb, values_tssb_e2_d, 's-', color='tab:green', label="EXT-M3 (Downsampling first)", markersize=6)

    plt.title("TSSB", fontsize=14)
    plt.xlabel("w_e-Values (3 to 19)", fontsize=12)
    plt.ylabel("Average Execution Time (s)", fontsize=12)
    plt.xticks(x_tssb, x_labels_tssb, fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # 2️⃣ Rechter Plot: FLOSS
    plt.subplot(1, 2, 2)
    plt.plot(x_floss, values_floss_e_s, 'o-', color='tab:red', label="EXT-M2 (Spikelet first)", markersize=6)
    plt.plot(x_floss, values_floss_e_d, 'o-', color='tab:orange', label="EXT-M2 (Downsampling first)", markersize=6)
    plt.plot(x_floss, values_floss_e2_s, 's-', color='tab:blue', label="EXT-M3 (Spikelet first)", markersize=6)
    plt.plot(x_floss, values_floss_e2_d, 's-', color='tab:green', label="EXT-M3 (Downsampling first)", markersize=6)

    plt.title("FLOSS", fontsize=14)
    plt.xlabel("w_e-Values (3 to 19)", fontsize=12)
    plt.ylabel("Average Execution Time (s)", fontsize=12)
    plt.xticks(x_floss, x_labels_floss, fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_downsampling_nth_single(data1, data2):
    """
    Plottet zwei Diagramme:
    - Erster Graph: data1 ("Spikelet=False") für nth und nth2
    - Zweiter Graph: data2 ("Spikelet=False") für nth und nth2
    """

    def extract_param_and_avg_scores_nth(data, downsampling_pattern):
        """Extrahiert {Param: Durchschnitts-Score} für den gegebenen Downsampling-Typ."""
        param2avg = {}
        for combo_name, score_list in data["scores"].items():
            if downsampling_pattern not in combo_name or "Spikelet=False" not in combo_name:
                continue

            match = re.search(r"Param=(\d+)", combo_name)
            if match:
                param = int(match.group(1))
            else:
                continue

            if 2 <= param <= 20:
                avg_score = sum(score_list) / len(score_list)
                param2avg[param] = avg_score

        return param2avg

    def plot_two_lines(ax, data, title):
        """Plottet nth und nth2 für einen Datensatz."""
        p2avg_nth = extract_param_and_avg_scores_nth(data, "Downsampling=nth,")
        p2avg_nth2 = extract_param_and_avg_scores_nth(data, "Downsampling=nth2,")

        all_params_nth = sorted(p2avg_nth.keys())
        all_params_nth2 = sorted(p2avg_nth2.keys())

        ax.plot(all_params_nth, 
                [p2avg_nth.get(p, None) for p in all_params_nth], 
                marker='o', color='tab:red', label='NTH-DEL')

        ax.plot(all_params_nth2, 
                [p2avg_nth2.get(p, None) for p in all_params_nth2], 
                marker='s', color='tab:orange', label='NTH-MEAN')

        ax.set_title(title, fontsize=15)
        ax.set_xlabel("n-Values (2 to 15)", fontsize=12)
        ax.set_ylabel("Average Score", fontsize=14)
        ax.legend()
        ax.grid(True)

    # 1x2 Subplots erstellen
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("", fontsize=14)

    # Erster Graph: nth und nth2 für data1 (TSSB Benchmark)
    axs[0].tick_params(axis='x', labelsize=12)  # X-Achsen-Beschriftung anpassen
    axs[0].tick_params(axis='y', labelsize=12)  # Y-Achsen-Beschriftung anpassen
    axs[0].grid(True, linestyle='--', alpha=0.6)
    plot_two_lines(axs[0], data1, "TSSB")

    # Zweiter Graph: nth und nth2 für data2 (FLOSS Benchmark)
    axs[1].tick_params(axis='x', labelsize=12)  # X-Achsen-Beschriftung anpassen
    axs[1].tick_params(axis='y', labelsize=12)  # Y-Achsen-Beschriftung anpassen
    axs[1].grid(True, linestyle='--', alpha=0.6)
    plot_two_lines(axs[1], data2, "FLOSS")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_downsampling_extrema_single(data1, data2):
    """
    Erstellt zwei Diagramme:
      1) Extrema und Extrema2 für TSSB (Data1)
      2) Extrema und Extrema2 für FLOSS (Data2)
    Nur Scores mit "Spikelet=False" werden berücksichtigt.
    """

    def get_param2avg(data, downsampling_patterns):
        """
        Extrahiert {Param: Durchschnitts-Score} für die gegebenen Downsampling-Algorithmen 
        und filtert nach 'Spikelet=False'.
        """
        param2avg = {}
        for combo_name, scores_list in data["scores"].items():
            # Filter: Downsampling & Spikelet=False
            if ("Spikelet=False" not in combo_name) or \
               not any(pattern in combo_name for pattern in downsampling_patterns):
                continue

            match = re.search(r"Param=(\d+)", combo_name)
            if not match:
                continue

            param = int(match.group(1))
            if 2 <= param <= 20:
                avg_score = sum(scores_list) / len(scores_list)
                param2avg[param] = avg_score

        return param2avg

    def plot_two_lines(ax, data, label):
        """
        Plottet Extrema- und Extrema2-Linien für einen Datensatz in einem Diagramm,
        wobei nur 'Spikelet=False' berücksichtigt wird.
        """
        # Werte für Extrema & Extrema2
        p2avg_extrema = get_param2avg(data, ["Downsampling=Extrema,"])
        p2avg_extrema2 = get_param2avg(data, ["Downsampling=Extrema2,"])

        # Gemeinsame X-Werte
        all_params_extrema = sorted(p2avg_extrema.keys())
        all_params_extrema2 = sorted(p2avg_extrema2.keys())

        # Linien plotten
        ax.plot(all_params_extrema, 
                [p2avg_extrema.get(p, None) for p in all_params_extrema], 
                marker='o', color='tab:red', label=f'EXT-M2')

        ax.plot(all_params_extrema2, 
                [p2avg_extrema2.get(p, None) for p in all_params_extrema2], 
                marker='s', color='tab:orange', label=f'EXT-M3')

        # Achsentitel, Grid und Legende
        ax.set_xlabel("w_e-Values (2 to 19)", fontsize=12)
        ax.set_ylabel("Average Score", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=9)

    # ------------------------
    # Plot erstellen
    # ------------------------
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("", fontsize=14)

    axs[0].set_title("TSSB", fontsize=15)
    axs[0].tick_params(axis='x', labelsize=12)  # X-Achsen-Beschriftung anpassen
    axs[0].tick_params(axis='y', labelsize=12)  # Y-Achsen-Beschriftung anpassen
    axs[0].grid(True, linestyle='--', alpha=0.6)
    plot_two_lines(axs[0], data1, "TSSB")

    # Zweiter Graph: FLOSS (Data2)
    axs[1].set_title("FLOSS", fontsize=15)
    axs[1].tick_params(axis='x', labelsize=12)  # X-Achsen-Beschriftung anpassen
    axs[1].tick_params(axis='y', labelsize=12)  # Y-Achsen-Beschriftung anpassen
    axs[1].grid(True, linestyle='--', alpha=0.6)
    plot_two_lines(axs[1], data2, "FLOSS")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_average_execution_times_nth_single(times1, times2):
    """
    Erstellt zwei Diagramme NEBENEINANDER:
      - Links: TSSB-Datensätze (nth + nth2)
      - Rechts: FLOSS-Datensätze (nth + nth2)
    Mit allen Kombinationen: Spikelet zuerst & Downsampling zuerst.

    Parameters:
    - times1, times2 (dict): Runtimes pro Datensatz.
    """
    avg_times1 = calculate_average_execution_times(times1)  # TSSB
    avg_times2 = calculate_average_execution_times(times2)  # FLOSS

    def extract_param(key):
        """Extrahiert die Param-Zahl aus einem Key."""
        match = re.search(r"Param=(\d+)", key)
        return int(match.group(1)) if match else None

    def sort_and_extract_values(avg_times, downsampling_type):
        """Sortiert die Keys nach `Param=X` und filtert `Spikelet=False`."""
        spikelet_filter = "Spikelet=False"
        valid_keys = {
            key: value for key, value in avg_times.items()
            if extract_param(key) is not None and
               spikelet_filter in key and
               f"Downsampling={downsampling_type}" in key
        }

        unique_params = {}
        for key, value in valid_keys.items():
            param = extract_param(key)
            if param not in unique_params:
                unique_params[param] = value

        sorted_params = sorted(unique_params.keys())
        return sorted_params, [unique_params[param] for param in sorted_params]

    # -------------------- 🎨 TSSB-Werte -------------------- #
    params_tssb_nth, values_tssb_nth = sort_and_extract_values(avg_times1, "nth")
    params_tssb_nth2, values_tssb_nth2 = sort_and_extract_values(avg_times1, "nth2")

    unique_params_tssb = sorted(set(params_tssb_nth) | set(params_tssb_nth2))
    x_tssb = np.arange(len(unique_params_tssb))
    x_labels_tssb = [f"{param}" for param in unique_params_tssb]

    # -------------------- 🎨 FLOSS-Werte -------------------- #
    params_floss_nth, values_floss_nth = sort_and_extract_values(avg_times2, "nth")
    params_floss_nth2, values_floss_nth2 = sort_and_extract_values(avg_times2, "nth2")

    unique_params_floss = sorted(set(params_floss_nth) | set(params_floss_nth2))
    x_floss = np.arange(len(unique_params_floss))
    x_labels_floss = [f"{param}" for param in unique_params_floss]

    # ------------------ 🎨 Side-by-Side Plot ------------------ #
    plt.figure(figsize=(14, 6))

    # 1️⃣ Linker Plot: TSSB
    plt.subplot(1, 2, 1)
    plt.plot(x_tssb, values_tssb_nth, 'o-', label="NTH-DEL", markersize=6, color='tab:blue')
    plt.plot(x_tssb, values_tssb_nth2, 's-', label="NTH-MEAN", markersize=6, color='tab:green')

    plt.title("TSSB", fontsize=15)
    plt.xlabel("n-Values (2 to 15)", fontsize=12)
    plt.ylabel("Average Execution Time (s)", fontsize=14)
    plt.xticks(x_tssb, x_labels_tssb, rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # 2️⃣ Rechter Plot: FLOSS
    plt.subplot(1, 2, 2)
    plt.plot(x_floss, values_floss_nth, 'o-', label="NTH-DEL", markersize=6, color='tab:blue')
    plt.plot(x_floss, values_floss_nth2, 's-', label="NTH-MEAN", markersize=6, color='tab:green')

    plt.title("FLOSS", fontsize=15)
    plt.xlabel("n-Values (2 to 15)", fontsize=12)
    plt.ylabel("Average Execution Time (s)", fontsize=14)
    plt.xticks(x_floss, x_labels_floss, rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_average_execution_times_extrema_single(times1, times2):
    """
    Erstellt zwei Diagramme NEBENEINANDER:
      - Links: TSSB-Datensätze (Extrema + Extrema2)
      - Rechts: FLOSS-Datensätze (Extrema + Extrema2)
    Mit allen Kombinationen: Spikelet zuerst & Downsampling zuerst.

    Parameters:
    - times1, times2 (dict): Runtimes pro Datensatz.
    """
    avg_times1 = calculate_average_execution_times(times1)  # TSSB
    avg_times2 = calculate_average_execution_times(times2)  # FLOSS

    def extract_param(key):
        """Extrahiert die Param-Zahl aus einem Key."""
        match = re.search(r"Param=(\d+)", key)
        return int(match.group(1)) if match else None

    def sort_and_extract_values(avg_times, downsampling_type):
        """Sortiert die Keys nach `Param=X` und filtert `Spikelet=False`."""
        spikelet_filter = "Spikelet=False"
        valid_keys = {
            key: value for key, value in avg_times.items()
            if extract_param(key) is not None and
               spikelet_filter in key and
               f"Downsampling={downsampling_type}" in key
        }

        unique_params = {}
        for key, value in valid_keys.items():
            param = extract_param(key)
            if param not in unique_params:
                unique_params[param] = value

        sorted_params = sorted(unique_params.keys())
        return sorted_params, [unique_params[param] for param in sorted_params]

    # -------------------- 🎨 TSSB-Werte -------------------- #
    params_tssb_e, values_tssb_e = sort_and_extract_values(avg_times1, "Extrema")
    params_tssb_e2, values_tssb_e2 = sort_and_extract_values(avg_times1, "Extrema2")

    unique_params_tssb = sorted(set(params_tssb_e) | set(params_tssb_e2))
    x_tssb = np.arange(len(unique_params_tssb))
    x_labels_tssb = [f"{param}" for param in unique_params_tssb]

    # -------------------- 🎨 FLOSS-Werte -------------------- #
    params_floss_e, values_floss_e = sort_and_extract_values(avg_times2, "Extrema")
    params_floss_e2, values_floss_e2 = sort_and_extract_values(avg_times2, "Extrema2")

    unique_params_floss = sorted(set(params_floss_e) | set(params_floss_e2))
    x_floss = np.arange(len(unique_params_floss))
    x_labels_floss = [f"{param}" for param in unique_params_floss]

    # ------------------ 🎨 Side-by-Side Plot ------------------ #
    plt.figure(figsize=(14, 6))

    # 1️⃣ Linker Plot: TSSB
    plt.subplot(1, 2, 1)
    plt.plot(x_tssb, values_tssb_e, 'o-', label="EXT-M2", markersize=6, color='tab:blue')
    plt.plot(x_tssb, values_tssb_e2, 's-', label="EXT-M3", markersize=6, color='tab:green')

    plt.title("TSSB", fontsize=15)
    plt.xlabel("w_e-Values (2 to 19)", fontsize=12)
    plt.ylabel("Average Execution Time (s)", fontsize=14)
    plt.xticks(x_tssb, x_labels_tssb, rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # 2️⃣ Rechter Plot: FLOSS
    plt.subplot(1, 2, 2)
    plt.plot(x_floss, values_floss_e, 'o-', label="EXT-M2", markersize=6, color='tab:blue')
    plt.plot(x_floss, values_floss_e2, 's-', label="EXT-M3", markersize=6, color='tab:green')

    plt.title("FLOSS", fontsize=15)
    plt.xlabel("w_e-Values (2 to 19)", fontsize=12)
    plt.ylabel("Average Execution Time (s)", fontsize=14)
    plt.xticks(x_floss, x_labels_floss, rotation=0, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_memory_comparison(csv_file_tssb_spikelet, csv_file_tssb_no_spikelet, 
                           csv_file_floss_spikelet, csv_file_floss_no_spikelet):
    """
    Erstellt Scatterplots für Peak Memory, Average Memory und Memory Footprint
    in Abhängigkeit von der Datensatzgröße für zwei Benchmarks (TSSB & FLOSS),
    jeweils mit und ohne Spikelet.

    Parameters:
    - csv_file_tssb_spikelet (str): Pfad zur CSV-Datei mit Spikelet (TSSB).
    - csv_file_tssb_no_spikelet (str): Pfad zur CSV-Datei ohne Spikelet (TSSB).
    - csv_file_floss_spikelet (str): Pfad zur CSV-Datei mit Spikelet (FLOSS).
    - csv_file_floss_no_spikelet (str): Pfad zur CSV-Datei ohne Spikelet (FLOSS).
    """
    # CSV-Dateien laden
    df_tssb_spikelet = pd.read_csv(csv_file_tssb_spikelet)
    df_tssb_no_spikelet = pd.read_csv(csv_file_tssb_no_spikelet)
    df_floss_spikelet = pd.read_csv(csv_file_floss_spikelet)
    df_floss_no_spikelet = pd.read_csv(csv_file_floss_no_spikelet)

    # Prüfen, ob die notwendigen Spalten existieren
    required_columns = ["Dataset Size", "Peak Memory (MB)", "Average Memory (MB)"]
    for df, name in zip([df_tssb_spikelet, df_tssb_no_spikelet, df_floss_spikelet, df_floss_no_spikelet], 
                         ["TSSB Spikelet", "TSSB No Spikelet", "FLOSS Spikelet", "FLOSS No Spikelet"]):
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV-Datei ({name}) muss die Spalten {required_columns} enthalten!")

    # 🔹 Subplots erstellen (TSSB links, FLOSS rechts)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # 🔹 Linker Plot: TSSB
    axes[0].scatter(df_tssb_spikelet["Dataset Size"], df_tssb_spikelet["Peak Memory (MB)"], 
                    color="orange", edgecolors="black", label="Peak Memory (Spikelet)", alpha=0.7)
    axes[0].scatter(df_tssb_no_spikelet["Dataset Size"], df_tssb_no_spikelet["Peak Memory (MB)"], 
                    color="green", edgecolors="black", label="Peak Memory (No Spikelet)", alpha=0.7)
    axes[0].scatter(df_tssb_spikelet["Dataset Size"], df_tssb_spikelet["Average Memory (MB)"], 
                    color="red", edgecolors="black", label="Average Memory (Spikelet)", alpha=0.7)
    axes[0].scatter(df_tssb_no_spikelet["Dataset Size"], df_tssb_no_spikelet["Average Memory (MB)"], 
                    color="blue", edgecolors="black", label="Average Memory (No Spikelet)", alpha=0.7)
    axes[0].set_xlabel("Dataset Size")
    axes[0].set_ylabel("Memory Usage (MB)")
    axes[0].set_title("TSSB")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 🔹 Rechter Plot: FLOSS
    axes[1].scatter(df_floss_spikelet["Dataset Size"], df_floss_spikelet["Peak Memory (MB)"], 
                    color="orange", edgecolors="black", label="Peak Memory (Spikelet)", alpha=0.7)
    axes[1].scatter(df_floss_no_spikelet["Dataset Size"], df_floss_no_spikelet["Peak Memory (MB)"], 
                    color="green", edgecolors="black", label="Peak Memory (No Spikelet)", alpha=0.7)
    axes[1].scatter(df_floss_spikelet["Dataset Size"], df_floss_spikelet["Average Memory (MB)"], 
                    color="red", edgecolors="black", label="Average Memory (Spikelet)", alpha=0.7)
    axes[1].scatter(df_floss_no_spikelet["Dataset Size"], df_floss_no_spikelet["Average Memory (MB)"], 
                    color="blue", edgecolors="black", label="Average Memory (No Spikelet)", alpha=0.7)
    axes[1].set_xlabel("Dataset Size")
    axes[1].set_title("FLOSS")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Layout optimieren und anzeigen
    plt.tight_layout()
    plt.show()



