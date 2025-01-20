import matplotlib.pyplot as plt
import numpy as np

from calculations_for_evaluation import sum_execution_times_per_dataset, t_test, wilcoxon_test, calc_average, calc_median

def show_hist(values):
    plt.hist(values, bins=50)
    plt.show()

def show_boxplot(values, values2, values3, values4):
    data = [values, values2, values3, values4]
    labels = ["n=2", "n=3", "n=4", "n=5"]  # Passe die Labels an deine Daten an

    for label, vals in zip(labels, data):
        avg_scores = calc_average(vals)
        median_scores = calc_median(vals)
        print(f"{label}: average score: {avg_scores}")
        print(f"{label}: median score: {median_scores}")

    # Boxplot erstellen
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, patch_artist=True, boxprops=dict(facecolor="lightblue"))

    # Details hinzufügen
    plt.title("Distribution of F1-Scores", fontsize=14)
    plt.xlabel("nth-point Selection Variants", fontsize=12)
    plt.ylabel("F1-Scores", fontsize=12)

    # Gitter hinzufügen und anzeigen
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def visualize_times_bars(data):
    # Example data (replace with your actual data)
    datasets = data["time_series_name"]  # List of dataset names
    combinations = [
        "Spikelet=False, Downsampling=None",
        "Spikelet=False, Downsampling=nth",
        "Spikelet=False, Downsampling=Extrema",
        #"Spikelet=True, Downsampling=None",
        #"Spikelet=True, Downsampling=nth",
        #"Spikelet=True, Downsampling=Extrema",
    ]
    execution_times = sum_execution_times_per_dataset(data["execution_times"])

    x = np.arange(len(datasets)) * 1.2 # Dataset indices
    width = 0.3  # Width of bars for each combination

    # Plot setup
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot bars for each combination
    for i, combination in enumerate(combinations):
        ax.bar(x + (i - len(combinations) / 2) * width, 
               execution_times[combination], 
               width, 
               label=combination)

    # Add labels and title
    ax.set_xlabel("Datasets", fontsize=12)
    ax.set_ylabel("Execution Time (s)", fontsize=12)
    ax.set_title("Execution Times per Dataset and Combination", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=90, fontsize=8)
    ax.legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_execution_times(dataset_names, times1, times2, times3, times4):
    """
    Plots a line chart for four lists of execution times with dataset names on the x-axis.

    Parameters:
    - dataset_names (list): List of dataset names (x-axis labels).
    - times1 (list): First list of execution times.
    - times2 (list): Second list of execution times.
    - times3 (list): Third list of execution times.
    - times4 (list): Fourth list of execution times.
    """
    times1 = sum_execution_times_per_dataset(times1["execution_times"])
    times2 = sum_execution_times_per_dataset(times2["execution_times"])
    times3 = sum_execution_times_per_dataset(times3["execution_times"])
    times4 = sum_execution_times_per_dataset(times4["execution_times"])

    lists = [times1, times2, times3, times4]
    labels = ["data1", "data2", "data3", "data4"]

    for label, data in zip(labels, lists):
        avg_time = calc_average(data["Spikelet=False, Downsampling=None"])
        print(f"{label}: average time: {avg_time}")

    x = np.arange(len(dataset_names))  # X-axis positions for datasets

    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot each line with corresponding label
    ax.plot(x, times1["Spikelet=True, Downsampling=nth"], label="n=2", linestyle='-', linewidth=2)
    ax.plot(x, times2["Spikelet=True, Downsampling=nth"], label="n=3", linestyle='-', linewidth=2)
    ax.plot(x, times3["Spikelet=True, Downsampling=nth"], label="n=4", linestyle='-', linewidth=2)
    ax.plot(x, times4["Spikelet=True, Downsampling=nth"], label="n=5", linestyle='-', linewidth=2)

    # Add gridlines (horizontal lines)
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

    # Set labels, title, and legend
    ax.set_xlabel("Datasets (ordered by length)", fontsize=12)
    ax.set_ylabel("Execution Time (s)", fontsize=12)
    ax.set_title("Execution Times per Dataset", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=90, fontsize=8)
    ax.legend()

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
