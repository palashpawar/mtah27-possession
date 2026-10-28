"""Fit the Smart Tagging weights by leave-one-game-out on the sample games, with simulated tags.

Held-out probabilities come from the same classifier and degradation as crossval.py and are cached,
so the grid only re-runs the decoder. The competition match is never scored here.
"""
import itertools
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from possession import features, model, tags  # noqa: E402
from possession.crossval import score  # noqa: E402
from possession.data import CACHE, load_sample  # noqa: E402

COST_TEAM, COST_DEAD = 10, 40
GRID = list(itertools.product((0, 1.5, 3), (10, 20), (0, 2)))  # w_dead, dead_s, w_live
SEEDS = (11, 12)


def held_out_probs(games):
    rng = np.random.default_rng(model.SEED + 1)
    out = {}
    for held in games:
        test = games[held]
        deg = model.degrade(test, rng, 0.25, 0.07, 0.6)
        path = os.path.join(CACHE, "cv_probs_{}.npz".format(held))
        if os.path.exists(path):
            z = np.load(path)
            pc, pd_ = z["clean"], z["degraded"]
        else:
            clf = model.fit([games[g] for g in games if g != held])
            pc, pd_ = model.frame_probs(clf, test), model.frame_probs(clf, deg)
            np.savez(path, clean=pc, degraded=pd_)
        out[(held, "clean")] = (test, pc)
        out[(held, "degraded")] = (deg, pd_)
    return out


def decode_logp(logp, m):
    out = np.full(m.n_frames, 3, np.int8)
    for a, b in m.periods:
        out[a:b] = model.viterbi(logp[a:b], COST_TEAM, COST_DEAD)
    return out


def main():
    games = {g: load_sample(g) for g in (1, 2, 3)}
    cases = held_out_probs(games)
    results = {k: [] for k in GRID}
    for (held, variant), (m, probs) in cases.items():
        feats = features.build(m)
        logp = np.log(np.clip(probs, 1e-6, 1))
        for seed in SEEDS:
            sim = tags.simulate(m, np.random.default_rng(seed))
            for k in GRID:
                pred = decode_logp(tags.apply(logp, sim, feats, k[0], k[1], k[2]), m)
                results[k].append((variant,) + score(m.labels, pred))
        base = results[(0, 10, 0)][-1]
        print("game {} {:9s} no tags spell {:.3f} deadF1 {:.3f}".format(held, variant, base[1], base[5]), flush=True)
    print("\nw_dead dead_s w_live | spellF1 all / degraded | deadF1 all")
    for k in sorted(GRID, key=lambda k: -np.mean([r[1] for r in results[k]])):
        r = results[k]
        print("{:>6} {:>6} {:>6} | {:.3f} / {:.3f} | {:.3f}".format(
            k[0], k[1], k[2], np.mean([x[1] for x in r]),
            np.mean([x[1] for x in r if x[0] == "degraded"]), np.mean([x[5] for x in r])))


if __name__ == "__main__":
    main()
