"""Aggregate statistics for each match. Prints only summary numbers, never per-frame values."""
import sys
import time

import numpy as np

from possession.data import load_competition, load_sample


def runs(mask):
    """Lengths of True runs."""
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(m)
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def describe(m):
    idx = np.concatenate([np.arange(a, b) for a, b in m.periods])
    ball = m.ball[idx]
    seen = ~np.isnan(ball[:, 0])
    L, W = m.pitch
    inside = seen & (ball[:, 0] >= 0) & (ball[:, 0] <= L) & (ball[:, 1] >= 0) & (ball[:, 1] <= W)
    gaps = np.concatenate([runs(np.isnan(m.ball[a:b, 0])) for a, b in m.periods])
    nplay = (~np.isnan(m.players[idx, :, 0])).sum(1)
    out = {"frames_in_play_periods": len(idx), "ball_seen": seen.mean().round(3),
           "ball_seen_outside_pitch": (seen & ~inside).mean().round(4),
           "gap_count": len(gaps), "gap_len_q50/90/99": np.percentile(gaps, [50, 90, 99]).round(0).tolist() if len(gaps) else [],
           "players_tracked_mean": nplay.mean().round(1)}
    if m.labels is not None:
        lab = m.labels[idx]
        out["label_share_H/A/D/X"] = [round(float((lab == k).mean()), 3) for k in range(4)]
        out["ball_seen_when_alive/dead"] = [round(float(seen[lab < 2].mean()), 3), round(float(seen[lab == 2].mean()), 3)]
        spells = sum(len(runs(lab == k)) for k in (0, 1))
        out["possessions"] = spells
        out["mean_possession_s"] = round(float((lab < 2).sum() / max(spells, 1) / 25), 1)
    print(m.name, out)


if __name__ == "__main__":
    for which in sys.argv[1:]:
        t = time.time()
        describe(load_competition() if which == "comp" else load_sample(int(which)))
        print("  ({:.0f}s)".format(time.time() - t))
