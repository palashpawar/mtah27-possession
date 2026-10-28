"""Train on all ten public matches (Metrica samples + IDSSE) and label the competition match.

    .venv/bin/python -m possession.make_submission            # -> evaluation/results/possession_v4.csv

Deterministic: the same data and code reproduce the same file. Prints aggregates only.
"""

import argparse
import os
import warnings

import numpy as np

from possession import features, model, sources, tags
from possession.data import ROOT, load_competition

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Chosen by 5-fold cross-validation over the ten public matches (possession/crossval_all.py).
COST_TEAM = 20
COST_DEAD = 40
STRIDE = 3
# Smart Tagging evidence (w_dead, dead_s, w_live), fitted on simulated tags (possession/tag_crossval.py).
TAG_WEIGHTS = (3, 20, 0)
LETTERS = np.array(["H", "A", "D", "X"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="possession_v4")
    ap.add_argument("--no-tags", action="store_true", help="decode without Smart Tagging")
    args = ap.parse_args()

    clf = model.fit(sources.all_matches(), stride=STRIDE)
    match = load_competition()
    probs = model.frame_probs(clf, match)
    logp = np.log(np.clip(probs, 1e-6, 1))
    if not args.no_tags:
        logp = tags.apply(logp, tags.competition_tags(), features.build(match), *TAG_WEIGHTS)
    codes = np.full(match.n_frames, 3, np.int8)
    for a, b in match.periods:
        codes[a:b] = model.viterbi(logp[a:b], COST_TEAM, COST_DEAD)

    path = os.path.join(ROOT, "evaluation", "results", args.name + ".csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, LETTERS[c]) for i, c in enumerate(codes))

    graded = np.concatenate([np.arange(a, b) for a, b in match.periods])
    g = codes[graded]
    n_poss = sum(int(((g[1:] != g[:-1]) & (g[1:] < 2)).sum()) for _ in [0]) + int(g[0] < 2)
    print("wrote", os.path.relpath(path, ROOT))
    print("graded frames H {:.3f}  A {:.3f}  D {:.3f}".format(*[(g == k).mean() for k in range(3)]))
    print("possessions ~{}  mean {:.1f} s".format(n_poss, (g < 2).sum() / max(n_poss, 1) / 25))


if __name__ == "__main__":
    main()
