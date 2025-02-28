import os
ABS_PATH = os.path.dirname(os.path.abspath(__file__))


import numpy as np
import pandas as pd


def load_floss_dataset(selection=None):
    desc_filename = ABS_PATH + "/../datasets/FLOSS/desc.txt"
    desc_file = np.genfromtxt(fname=desc_filename, delimiter=',', filling_values=[None], dtype=None, encoding='utf8')

    df = []

    if selection is None:
        selection = slice(len(desc_file))
    else:
        selection = slice(selection, selection+1)

    for ts_name, window_size, floss_score, cp_1, cp_2 in desc_file[selection]:
        change_points = [cp_1]
        if cp_2 != -1: change_points.append(cp_2)

        ts = np.loadtxt(fname=os.path.join(ABS_PATH, '../datasets/FLOSS/', ts_name + '.txt'), dtype=np.float64)
        df.append((ts_name, int(window_size), np.array(change_points), ts))

    return pd.DataFrame.from_records(df, columns=["name", "window_size", "change_points", "time_series"])


def load_ucr_dataset():
    desc_filename = ABS_PATH + "/../datasets/UCR/desc.txt"
    desc_file = []

    with open(desc_filename, 'r') as file:
        for line in file.readlines(): desc_file.append(line.split(","))

    df = []

    for row in desc_file:
        (ts_name, window_size), change_points = row[:2], row[2:]

        ts = np.loadtxt(fname=os.path.join(ABS_PATH, '../datasets/UCR/', ts_name + '.txt'), dtype=np.float64)
        df.append((ts_name, int(window_size), np.array([int(_) for _ in change_points]), ts))

    return pd.DataFrame.from_records(df, columns=["name", "window_size", "change_points", "time_series"])


def load_combined_dataset():
    df = pd.concat([load_floss_dataset(), load_ucr_dataset()])
    df.sort_values(by="name", inplace=True)
    return df