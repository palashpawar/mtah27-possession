"""Leave-one-game-out validation on Metrica's public sample games.

This is where every modelling choice is made. The competition match is never scored here.
For each held-out game: train on the other two, predict the held-out game both clean and
degraded, and score with the competition's own metric library across a grid of decoder costs.
"""

import itertools
import os
import sys

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

from possession import model  # noqa: E402
from possession.data import load_sample  # noqa: E402

COST_TEAM = (10, 20, 40)
COST_DEAD = (20, 40, 80)


def score(gt, pred):
    r = pm.evaluate(gt, pred, bootstrap_n=0)
    return (r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"],
            r["per_class"]["D"]["f1"], r["transitions"][25]["f1"])


def baseline(m, window):
    """The repository's reference method on a sample game: nearest player's team, majority vote."""
    from possession.features import build
    f = build(m)
    codes = np.where(f["seen"] > 0, np.where(f["d_home"] <= f["d_away"], 0, 1), -1)
    c = np.concatenate(([0], np.cumsum(codes == 0))), np.concatenate(([0], np.cumsum(codes == 1)))
    i = np.arange(len(codes)); lo = np.maximum(0, i - window); hi = np.minimum(len(codes), i + window + 1)
    h, a = c[0][hi] - c[0][lo], c[1][hi] - c[1][lo]
    out = np.where(h + a == 0, 2, np.where(h >= a, 0, 1)).astype(np.int8)
    return np.where(m.labels == 3, 3, out)


def main():
    games = {g: load_sample(g) for g in (1, 2, 3)}
    rng = np.random.default_rng(model.SEED + 1)
    results = {}  # (held, variant) -> {(ct, cd): (spell, macro, npred, ngt)}
    for held in games:
        clf = model.fit([games[g] for g in games if g != held], show_dead="--no-show-dead" not in sys.argv)
        test = games[held]
        for variant, m in (("clean", test), ("degraded", model.degrade(test, rng, 0.25, 0.07, 0.6))):
            probs = model.frame_probs(clf, m)
            raw = probs.argmax(1).astype(np.int8)
            table = {"argmax": score(m.labels, np.where(m.labels == 3, 3, raw)),
                     "baseline50": score(m.labels, baseline(m, 50))}
            for ct, cd in itertools.product(COST_TEAM, COST_DEAD):
                table[(ct, cd)] = score(m.labels, model.decode(probs, m.periods, m.n_frames, ct, cd))
            results[(held, variant)] = table
            best = max((k for k in table if isinstance(k, tuple)), key=lambda k: table[k][0])
            print("game {} {:9s} baseline50 spell {:.3f} | argmax spell {:.3f} | best {} spell {:.3f} macro {:.3f} pred {} gt {} deadF1 {:.3f}".format(
                held, variant, table["baseline50"][0], table["argmax"][0], best, *table[best]), flush=True)

    keys = [k for k in next(iter(results.values())) if not isinstance(k, str)]
    for name in ("baseline50", "argmax"):
        print("{:<10} mean spellF1 {:.3f}".format(name, np.mean([results[r][name][0] for r in results])))
    mean = {k: np.mean([results[r][k][0] for r in results]) for k in keys}
    deg = {k: np.mean([results[r][k][0] for r in results if r[1] == "degraded"]) for k in keys}
    print("\ntop cost settings by mean spellF1 over 3 held-out games x {clean, degraded}:")
    for k in sorted(keys, key=lambda k: -mean[k])[:8]:
        print("  cost_team {:>4} cost_dead {:>4}  mean {:.3f}  degraded-only {:.3f}".format(k[0], k[1], mean[k], deg[k]))


if __name__ == "__main__":
    main()
