import sys
import os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_dir)

from aeon.datasets import (
    load_gun_point_segmentation,
    load_psyllid_reduced_segmentation,
    load_psyllid_segmentation,
)
from aeon.segmentation import ClaSPSegmenter, find_dominant_window_sizes

#X, true_period_size, cps = load_gun_point_segmentation()
X, true_period_size, cps = load_psyllid_segmentation()
#X, true_period_size, cps = load_psyllid_reduced_segmentation()

# hier sollte die transformation von der Timeseries (X) stattfinden - man sollte auch an anderen orten experimentieren
# bsp:

from aeon.transformations.spikelet.spikelet import motif_discovery_and_clasp

X = motif_discovery_and_clasp(X)

dominant_period_size = find_dominant_window_sizes(X)
clasp = ClaSPSegmenter(dominant_period_size, n_cps=1)
found_cps = clasp.fit_predict(X)
profiles = clasp.profiles
scores = clasp.scores
