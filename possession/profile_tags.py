"""Aggregate profile of tracking and of our own predictions around Smart Tagging set pieces.

Prints only means over all set-piece tags per time bin -- no individual tag, frame or value.
"""
import sys
import warnings

import numpy as np

from possession import features
from possession.data import ROOT, load_competition

warnings.filterwarnings("ignore")
sys.path.insert(0, ROOT)
from starter.load_smart_tagging import load as load_tags  # noqa: E402

BINS = np.arange(-30, 31, 3)  # seconds relative to the anchor


def main():
    m = load_competition()
    f = features.build(m)
    pred = np.loadtxt(ROOT + "/evaluation/results/possession_v2.csv", delimiter=",", skiprows=1,
                      dtype=str)[:, 1]
    dead_pred = (pred == "D").astype(float)
    tags = [t for t in load_tags() if t["code"].lower() == "set pieces"]
    lens = np.array([t["end_frame"] - t["start_frame"] for t in tags]) / 25
    print("set-piece tags {}, duration s q10/50/90 {}".format(len(tags), np.percentile(lens, [10, 50, 90]).round(1).tolist()))
    for anchor in ("start_frame", "end_frame"):
        print("\nanchor = tag {}   (mean over tags; t in seconds)".format(anchor.split("_")[0]))
        print("   t   ball_seen  ball_speed  players_spd  v2_pred_dead")
        for lo in BINS[:-1]:
            rows = []
            for t in tags:
                a = t[anchor] + int(lo * 25)
                b = a + 75
                if a < 0 or b > m.n_frames:
                    continue
                rows.append((f["seen"].iloc[a:b].mean(), f["ball_speed"].iloc[a:b].clip(lower=0).mean(),
                             f["spd_all"].iloc[a:b].mean(), dead_pred[a:b].mean()))
            r = np.mean(rows, 0)
            print("{:>4}   {:9.2f}  {:10.1f}  {:11.2f}  {:12.2f}".format(lo, *r))


if __name__ == "__main__":
    main()
