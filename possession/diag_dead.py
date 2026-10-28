"""Is v3's dead-ball share plausible? Aggregates only, no labels from the competition match.

Compares what predicted-dead frames look like on the competition match with what truly dead frames
look like on the external matches (degraded the same way as in validation), plus ball-gap shapes.
"""
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from possession import features, model, sources  # noqa: E402
from possession.data import ROOT, load_competition  # noqa: E402
from possession.describe import runs  # noqa: E402


def profile(f, dead, idx):
    f = f.iloc[idx]
    dead = dead[idx]
    seen = f["seen"].values > 0
    spd = f["spd_all"].values
    fast_ball = seen & (f["ball_speed"].values > 8)
    return [dead.mean(),                          # dead share
            dead[~seen].mean(), dead[seen].mean(),  # P(dead | ball missing / seen)
            np.median(spd[dead]), np.median(spd[~dead]),
            fast_ball[dead].mean(),               # dead frames with a fast, visible ball
            (spd[dead] > 2.5).mean()]             # dead frames with players running


def gaps(match):
    g = np.concatenate([runs(np.isnan(match.ball[a:b, 0])) for a, b in match.periods])
    miss = np.mean(np.concatenate([np.isnan(match.ball[a:b, 0]) for a, b in match.periods]))
    return [miss, len(g)] + np.percentile(g, [25, 50, 75, 90]).round(0).tolist()


HEAD = "{:22s} dead  P(D|miss) P(D|seen) spdD spdAlive fastballD runD"


def main():
    rng = np.random.default_rng(model.SEED + 1)
    print(HEAD)
    rows = []
    for m in sources.all_matches():
        d = model.degrade(m, rng, 0.25, 0.07, 0.6)
        idx = np.concatenate([np.arange(a, b) for a, b in d.periods])
        idx = idx[d.labels[idx] < 3]
        r = profile(features.build(d), d.labels == 2, idx)
        rows.append(r)
        print("{:22s} ".format(d.name + " TRUE") + " ".join("{:8.2f}".format(v) for v in r), "| gaps", gaps(d))
    comp = load_competition()
    f = features.build(comp)
    idx = np.concatenate([np.arange(a, b) for a, b in comp.periods])
    for v in ("possession_v2", "possession_v3", "possession_v4"):
        if not os.path.exists("{}/evaluation/results/{}.csv".format(ROOT, v)):
            continue
        lab = np.loadtxt("{}/evaluation/results/{}.csv".format(ROOT, v), delimiter=",", skiprows=1, dtype=str)[:, 1]
        r = profile(f, lab == "D", idx)
        print("{:22s} ".format("BAR-ATH " + v[-2:] + " PRED") + " ".join("{:8.2f}".format(x) for x in r))
    print("BAR-ATH raw ball gaps  | gaps", gaps(comp))


if __name__ == "__main__":
    main()
