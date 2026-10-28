"""How well do touches and control track true possession? Public matches; competition = counts only."""
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
from possession import features, model, sources, touches  # noqa: E402
from possession.data import load_competition  # noqa: E402


def agree(sign, labels, idx):
    alive = idx[labels[idx] < 2]
    truth = np.where(labels[alive] == 0, 1, -1)
    s = sign[alive]
    return (s == truth).mean(), (s != 0).mean()


def main():
    rng = np.random.default_rng(5)
    print("match                  variant   touches/match  ctrl acc  last-touch acc  nearest acc  (coverage)")
    for m in sources.metrica()[:2] + sources.dfl()[:3]:
        for variant, x in (("clean", m), ("degraded", model.degrade(m, rng, 0.25, 0.07, 0.6))):
            t0 = time.time()
            tf = touches.features(x)
            f = features.build(x)
            idx = np.concatenate([np.arange(a, b) for a, b in x.periods])
            n_t = len(touches.detect(x)[0])
            near = np.sign(f["d_away"].values - f["d_home"].values)
            c_acc, c_cov = agree(tf["ctrl_last"], x.labels, idx)
            l_acc, _ = agree(tf["tch_last"], x.labels, idx)
            n_acc, _ = agree(near, x.labels, idx)
            print("{:22s} {:9s} {:13d}  {:8.3f}  {:14.3f}  {:11.3f}  ({:.2f})".format(
                m.name, variant, n_t, c_acc, l_acc, n_acc, c_cov))
    comp = load_competition()
    print("BAR-ATH touches:", len(touches.detect(comp)[0]))


if __name__ == "__main__":
    main()
