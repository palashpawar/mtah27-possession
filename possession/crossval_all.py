"""Five-fold cross-validation over all 10 public matches (2 held out per fold).

Trains on every match outside the fold, once per degradation style, and scores both on identical
held-out copies degraded in the ATD-calibrated style:
  v3   -- short ball gaps, per-frame jitter (what v3 shipped with)
  atd  -- long ball gaps and smooth position error, shaped like the competition ATD
Each held-out match is scored clean and degraded, across a grid of decoder costs.
"""
import itertools
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from possession import model, sources  # noqa: E402
from possession.crossval import score  # noqa: E402

COST_TEAM = (5, 10, 20, 40)
COST_DEAD = (20, 40, 80)
STRIDE = 3
SETUPS = ("v3", "atd")  # training degradation style; test copies are always "atd"


def main():
    ms = sources.all_matches()
    names = [m.name for m in ms]
    folds = [(0, 3), (1, 4), (2, 5), (6, 7), (8, 9)]
    rng = np.random.default_rng(model.SEED + 1)
    tests = {i: model.degrade(ms[i], rng, 0.25, 0.07, 0.6) for i in range(len(ms))}
    res = {}  # (setup, match, variant) -> {costs: score tuple}
    for fold in folds:
        train_all = [ms[i] for i in range(len(ms)) if i not in fold]
        for setup in SETUPS:
            clf = model.fit(train_all, stride=STRIDE, style=setup)
            for i in fold:
                for variant, m in (("clean", ms[i]), ("degraded", tests[i])):
                    probs = model.frame_probs(clf, m)
                    table = {c: score(m.labels, model.decode(probs, m.periods, m.n_frames, *c))
                             for c in itertools.product(COST_TEAM, COST_DEAD)}
                    res[(setup, names[i], variant)] = table
                    best = max(table, key=lambda c: table[c][0])
                    print("{:8s} {:14s} {:9s} best {} spell {:.3f} deadF1 {:.3f}".format(
                        setup, names[i], variant, best, table[best][0], table[best][4]), flush=True)

    costs = list(itertools.product(COST_TEAM, COST_DEAD))
    for setup in SETUPS:
        keys = [k for k in res if k[0] == setup]
        mean = {c: np.mean([res[k][c][0] for k in keys]) for c in costs}
        c = max(mean, key=mean.get)
        def avg(pred):
            sel = [k for k in keys if pred(k)]
            return np.mean([res[k][c][0] for k in sel]), np.mean([res[k][c][4] for k in sel])
        print("\n{} training, best costs {} -> mean spellF1 {:.3f}".format(setup, c, mean[c]))
        for label, pred in (("metrica held-out", lambda k: k[1].startswith("metrica")),
                            ("idsse held-out", lambda k: k[1].startswith("idsse")),
                            ("degraded only", lambda k: k[2] == "degraded")):
            print("   {:16s} spellF1 {:.3f}  deadF1 {:.3f}".format(label, *avg(pred)))


if __name__ == "__main__":
    main()
