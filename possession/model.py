"""Frame classifier + switching-cost decoder.

1. A gradient-boosted classifier gives P(H), P(A), P(D) per frame from features.build().
   It is trained only on Metrica's public sample games (labels derived from their event data),
   made to look like the competition's uncorrected tracking by degrade(), and mirrored with
   swap_teams() so it cannot learn a home/away bias.
2. A Viterbi decoder turns those probabilities into a label sequence, paying a fixed cost for
   every change of state. The costs are what make the output a sequence of whole possessions
   rather than a per-frame guess; they are chosen by leave-one-game-out cross-validation on the
   sample games, never on the competition match.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from possession import features
from possession.data import Match

H, A, D, X = 0, 1, 2, 3
SEED = 2027


# -- making clean sample tracking look like automatic tracking -------------------------------------

def _gaps(n, frac, median, rng, sigma=0.9, short_share=0.0):
    """Boolean mask with random gaps covering roughly `frac` of n frames.

    With short_share > 0, that share of gaps are brief blips (median 5 frames) and the rest follow
    the main lognormal -- the bimodal shape the competition ATD's ball gaps have.
    """
    mask = np.zeros(n, bool)
    target = int(frac * n)
    covered = 0
    while covered < target:
        med, sig = (5, 0.6) if rng.random() < short_share else (median, sigma)
        length = int(min(max(rng.lognormal(np.log(med), sig), 1), 12 * med))
        start = rng.integers(0, max(n - length, 1))
        covered += length - mask[start:start + length].sum()
        mask[start:start + length] = True
    return mask


def _smooth_noise(shape, sigma, rng, knot=25):
    """Position error that drifts over about a second rather than jittering every frame.

    Independent per-frame noise of 0.2 m reads as ~1.4 m/s of fake speed after differencing, which
    makes live play look faster than the competition ATD (whose speeds match clean tracking).
    """
    n = shape[0]
    k = n // knot + 2
    knots = rng.normal(0, sigma, (k,) + shape[1:]).astype(np.float32)
    t = np.arange(n) / knot
    i = np.floor(t).astype(int)
    w = (t - i).reshape((-1,) + (1,) * (len(shape) - 1)).astype(np.float32)
    return (1 - w) * knots[i] + w * knots[i + 1]


def _show_dead_ball(ball, labels, rng, visible):
    """Put a ball back into dead-ball stretches, the way automatic tracking keeps seeing one.

    Metrica's sample data blanks the ball whenever play is stopped; automatic tracking does not --
    it follows the ball out of play, while a player carries it back, and while it sits on the spot
    for the restart. For a `visible` share of stoppages, rebuild that: the ball rests where it left
    play, is optionally carried towards the restart point, then sits at the restart point.
    """
    clean = ball.copy()
    seen = ~np.isnan(clean[:, 0])
    dead = labels == D
    m = np.concatenate(([0], dead.astype(np.int8), [0]))
    starts, stops = np.flatnonzero(np.diff(m) == 1), np.flatnonzero(np.diff(m) == -1)
    for a, b in zip(starts, stops):
        if rng.random() > visible:
            # hidden: blank whatever the source tracked here (DFL keeps the ball through
            # stoppages), keeping a moment at the start as the ball goes out of play
            ball[a + int(rng.uniform(0, 0.2) * (b - a)):b] = np.nan
            continue
        if seen[a:b].mean() > 0.5:
            continue  # the source already tracks the ball through this stoppage
        before = np.flatnonzero(seen[:a])
        after = np.flatnonzero(seen[b:])
        if not len(before) or not len(after):
            continue
        out_pos, restart = clean[before[-1]], clean[b + after[0]]
        length = b - a
        rest = int(rng.uniform(0, 0.3) * length)
        placed = int(rng.uniform(0.2, 0.7) * length)
        ball[a:a + rest] = out_pos
        if rng.random() < 0.5:
            mid = np.arange(a + rest, b - placed)
            if len(mid):
                w = ((mid - mid[0]) / max(len(mid), 1))[:, None]
                ball[mid] = (1 - w) * out_pos + w * restart
        ball[b - placed:b] = restart
    return ball


def degrade(match, rng, ball_drop=None, player_drop=None, dead_visible=None, style="atd"):
    """Copy of a clean match made to look like automatic tracking: extra ball gaps during play,
    a ball that stays visible through stoppages, player dropouts and position noise."""
    if ball_drop is None:
        ball_drop = rng.uniform(0.15, 0.45) if style == "atd" else rng.uniform(0.10, 0.35)
    player_drop = rng.uniform(0.03, 0.10) if player_drop is None else player_drop
    dead_visible = rng.uniform(0.3, 0.9) if dead_visible is None else dead_visible
    n, s = match.players.shape[:2]
    ball = match.ball.copy()
    players = match.players.copy()

    if match.labels is not None:
        ball = _show_dead_ball(ball, match.labels, rng, dead_visible)
    alive = np.isin(match.labels, (H, A)) if match.labels is not None else np.ones(n, bool)
    # Ball gaps shaped like the competition ATD's (unlabelled aggregate: ~10% blips of a few
    # frames, the rest long, median ~270 frames): long losses happen during live play too.
    # style="v3" keeps the earlier short gaps and per-frame jitter, for comparison only.
    if style == "atd":
        drop = _gaps(n, ball_drop, median=270, rng=rng, sigma=0.7, short_share=0.1) & alive
    else:
        drop = _gaps(n, ball_drop, median=40, rng=rng) & alive
    ball[drop] = np.nan
    noise = _smooth_noise if style == "atd" else (lambda shape, sd, r: r.normal(0, sd, shape).astype(np.float32))
    ball += noise(ball.shape, 0.3, rng)

    for k in range(s):
        players[_gaps(n, player_drop, median=60, rng=rng), k] = np.nan
    players += noise(players.shape, 0.2, rng)
    return Match(name=match.name + "_deg", pitch=match.pitch, ball=ball, players=players,
                 team=match.team, periods=match.periods, labels=match.labels)


def _graded(match):
    idx = np.concatenate([np.arange(a, b) for a, b in match.periods])
    return idx[match.labels[idx] < X] if match.labels is not None else idx


# -- classifier ------------------------------------------------------------------------------------

def training_table(matches, copies=2, seed=SEED, show_dead=True, stride=1, style="atd"):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for m in matches:
        variants = [m] + [degrade(m, rng, dead_visible=None if show_dead else 0.0, style=style)
                          for _ in range(copies)]
        for v in variants:
            idx = _graded(v)[::stride]
            df = features.build(v).iloc[idx]
            y = v.labels[idx]
            xs += [df, features.swap_teams(df)]
            ys += [y, np.where(y == H, A, np.where(y == A, H, y))]
    import pandas as pd
    return pd.concat(xs, ignore_index=True), np.concatenate(ys)


def fit(matches, copies=2, seed=SEED, show_dead=True, stride=1, style="atd"):
    x, y = training_table(matches, copies, seed, show_dead, stride, style)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=seed)
    clf.fit(x, y)
    return clf


def frame_probs(clf, match):
    """(T, 3) probabilities, symmetrised over the team swap so H and A are treated alike."""
    df = features.build(match)
    p = clf.predict_proba(df)
    q = clf.predict_proba(features.swap_teams(df))[:, [1, 0, 2]]
    return (p + q) / 2


# -- decoder ---------------------------------------------------------------------------------------

def viterbi(logp, cost_team, cost_dead):
    """Most likely H/A/D path under per-frame log-probs and fixed switching costs."""
    trans = np.array([[0, cost_team, cost_dead],
                      [cost_team, 0, cost_dead],
                      [cost_dead, cost_dead, 0]], np.float64)
    n = len(logp)
    back = np.zeros((n, 3), np.int8)
    score = logp[0].astype(np.float64).copy()
    for t in range(1, n):
        cand = score[:, None] - trans          # from i to j
        back[t] = cand.argmax(0)
        score = cand.max(0) + logp[t]
    path = np.empty(n, np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(probs, periods, n_frames, cost_team, cost_dead):
    out = np.full(n_frames, X, np.int8)
    logp = np.log(np.clip(probs, 1e-6, 1))
    for a, b in periods:
        out[a:b] = viterbi(logp[a:b], cost_team, cost_dead)
    return out
