"""Smart Tagging set pieces as evidence for the decoder.

Every set piece is a restart, and every restart ends a dead-ball stretch. Profiling the tracking
around the competition's tags (possession/profile_tags.py, aggregates only) shows a tag starts a few
seconds before the ball is struck, with a stoppage in the preceding ~15 s; the tag's end carries
no signal. So for each tag:

  1. find the restart in the tracking: the first frame after the tag start where a seen ball moves
     fast, falling back to a fixed offset;
  2. add log-evidence for D over the `dead_s` seconds before it;
  3. add log-evidence against D for the `live_s` seconds after it, and for the tagged team when
     the tag names one (throw-ins are tagged N/A).

The sample games have no Smart Tagging, so the weights are fitted on simulated tags built from
their real restarts (simulate), degraded to match what the competition tags look like: only about
half the restarts are tagged, the tag starts 2-6 s early, and some carry no team.
"""

import os
import sys

import numpy as np

from possession.data import ROOT

H, A, D, X = 0, 1, 2, 3
FPS = 25
SEARCH = 10 * FPS        # look for the strike up to 10 s after the tag start
FALLBACK = 4 * FPS       # restart assumed 4 s after the tag start when no strike is found
STRIKE = 4.0             # m/s


def competition_tags():
    """[(start_frame, team or None)] for the set pieces in the competition's Smart Tagging."""
    sys.path.insert(0, ROOT)
    from starter.load_smart_tagging import load
    out = []
    for t in load():
        if t["code"].lower() != "set pieces":
            continue
        v = (t["tags"].get("Team") or "").lower()
        team = H if v.startswith("barcelona") else A if v.startswith("atl") else None
        out.append((t["start_frame"], team))
    return out


def simulate(match, rng, keep=0.55, lead=(2, 6), no_team=0.55):
    """Tags built from a labelled game's restarts (D -> H/A), degraded like the real ones."""
    lab = match.labels
    out = []
    for a, b in match.periods:
        seg = lab[a:b]
        for i in np.flatnonzero((seg[:-1] == D) & (seg[1:] < D)) + 1:
            if rng.random() > keep:
                continue
            start = a + i - int(rng.uniform(*lead) * FPS)
            team = None if rng.random() < no_team else int(seg[i])
            out.append((max(start, a), team))
    return out


def restart_frame(start, feats, n):
    seen = feats["seen"].values
    speed = feats["ball_speed"].values
    hi = min(start + SEARCH, n)
    hits = np.flatnonzero((seen[start:hi] > 0) & (speed[start:hi] >= STRIKE))
    return start + int(hits[0]) if len(hits) else min(start + FALLBACK, n - 1)


def apply(logp, tags, feats, w_dead, dead_s, w_live, live_s=3):
    """Copy of logp with the tag evidence added."""
    out = logp.copy()
    n = len(out)
    for start, team in tags:
        r = restart_frame(start, feats, n)
        out[max(r - int(dead_s * FPS), 0):r, D] += w_dead
        lo, hi = r, min(r + int(live_s * FPS), n)
        out[lo:hi, D] -= w_live
        if team is not None:
            out[lo:hi, team] += w_live
            out[lo:hi, 1 - team] -= w_live
    return out
