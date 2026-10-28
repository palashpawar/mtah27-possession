"""Per-frame features from tracking alone. No identities, no direction of play.

Everything is team-relative in a symmetric way (home/away pairs), so a match can be mirrored by
swapping the pairs and the H/A labels -- see swap_teams. Distances are in metres, speeds in m/s.
"""

import numpy as np
import pandas as pd

FPS = 25
INTERP_MAX = 50          # interpolate the ball across gaps up to 2 s
NEAR = 3.0               # metres: "near the ball"
CAP = 500                # frames: cap for time-since / time-until features
WINDOWS = (12, 50, 125)  # half-widths, frames, for centred rolling summaries
TOUCHES = True           # add touch / pass-chain features (possession/touches.py)

# (home column, away column) pairs that swap when the teams do
PAIRS = [("d_home", "d_away"), ("near_home", "near_away"), ("spd_home", "spd_away"),
         ("n_home", "n_away"), ("last_d_home", "last_d_away"), ("next_d_home", "next_d_away")]


def _run_lengths(mask):
    """For each frame, the length of the True run containing it (0 where False)."""
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(m)
    starts, stops = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    out = np.zeros(len(mask), np.int32)
    for a, b in zip(starts, stops):
        out[a:b] = b - a
    return out


def _since(mask):
    """Frames since mask was last True (0 where True), capped."""
    idx = np.where(mask, np.arange(len(mask)), -10 ** 9)
    last = np.maximum.accumulate(idx)
    return np.minimum(np.arange(len(mask)) - last, CAP)


def _fill(arr, mask, forward=True):
    """Last (or next) observed row of arr where mask is True."""
    n = len(mask)
    if forward:
        idx = np.where(mask, np.arange(n), 0)
        idx = np.maximum.accumulate(idx)
    else:
        idx = np.where(mask, np.arange(n), n - 1)
        idx = np.minimum.accumulate(idx[::-1])[::-1]
    return arr[idx]


def _interp_ball(ball, seen):
    out = ball.copy()
    gap = _run_lengths(~seen)
    t = np.arange(len(ball))
    if seen.sum() >= 2:
        for k in range(2):
            out[:, k] = np.interp(t, t[seen], ball[seen, k])
    short = (~seen) & (gap <= INTERP_MAX)
    out[~seen & ~short] = np.nan
    return out


def _nearest(ball, players, team, side):
    """Distance to the nearest player of one side, and how many are within NEAR."""
    p = players[:, team == side, :]
    d = np.sqrt(((p - ball[:, None, :]) ** 2).sum(-1))
    with np.errstate(all="ignore"):
        dmin = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
    dmin[~np.isfinite(dmin)] = 60.0
    near = (d < NEAR).sum(1)
    return dmin.astype(np.float32), near.astype(np.float32)


def _speeds(players, lag=5):
    """Per-slot speed in m/s from a lag-frame difference; slot jumps are clipped away."""
    v = np.full(players.shape[:2], np.nan, np.float32)
    dp = players[lag:] - players[:-lag]
    v[lag:] = np.sqrt((dp ** 2).sum(-1)) * FPS / lag
    v[v > 12] = np.nan
    return v


def build(match):
    """DataFrame of features, one row per frame of the match."""
    ball, players, team = match.ball, match.players, match.team
    L, W = match.pitch
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])
    f = {}

    f["seen"] = seen.astype(np.float32)
    f["gap_len"] = np.minimum(_run_lengths(~seen), 3000).astype(np.float32)
    f["since_seen"] = _since(seen).astype(np.float32)
    f["until_seen"] = _since(seen[::-1])[::-1].astype(np.float32)

    bi = _interp_ball(ball, seen)
    avail = ~np.isnan(bi[:, 0])
    f["avail"] = avail.astype(np.float32)

    last = _fill(ball, seen, True)
    nxt = _fill(ball, seen, False)
    for name, pos in (("last", last), ("next", nxt)):
        out_d = np.maximum.reduce([-pos[:, 0], pos[:, 0] - L, -pos[:, 1], pos[:, 1] - W])
        f[name + "_outside"] = np.nan_to_num(out_d, nan=0.0).astype(np.float32)
        f[name + "_to_goalline"] = np.nan_to_num(np.minimum(np.abs(pos[:, 0]), np.abs(L - pos[:, 0])), nan=50).astype(np.float32)
        f[name + "_to_touch"] = np.nan_to_num(np.minimum(np.abs(pos[:, 1]), np.abs(W - pos[:, 1])), nan=30).astype(np.float32)

    # ball speed from the interpolated track, centred 4-frame difference
    bs = np.full(n, np.nan, np.float32)
    bs[2:-2] = np.sqrt(((bi[4:] - bi[:-4]) ** 2).sum(-1)) * FPS / 4
    f["ball_speed"] = np.nan_to_num(np.minimum(bs, 40), nan=-1).astype(np.float32)

    use = np.where(avail[:, None], bi, last)
    f["d_home"], f["near_home"] = _nearest(use, players, team, 0)
    f["d_away"], f["near_away"] = _nearest(use, players, team, 1)
    f["last_d_home"], _ = _nearest(last, players, team, 0)
    f["last_d_away"], _ = _nearest(last, players, team, 1)
    f["next_d_home"], _ = _nearest(nxt, players, team, 0)
    f["next_d_away"], _ = _nearest(nxt, players, team, 1)

    v = _speeds(players)
    with np.errstate(all="ignore"):
        f["spd_home"] = np.nan_to_num(np.nanmean(v[:, team == 0], 1), nan=0).astype(np.float32)
        f["spd_away"] = np.nan_to_num(np.nanmean(v[:, team == 1], 1), nan=0).astype(np.float32)
        f["spd_all"] = np.nan_to_num(np.nanmean(v, 1), nan=0).astype(np.float32)
        f["spd_p90"] = np.nan_to_num(np.nanpercentile(np.where(np.isnan(v), np.nan, v), 90, axis=1), nan=0).astype(np.float32)
    f["n_home"] = (~np.isnan(players[:, team == 0, 0])).sum(1).astype(np.float32)
    f["n_away"] = (~np.isnan(players[:, team == 1, 0])).sum(1).astype(np.float32)

    # set-piece geometry: a ball resting on a restart spot with players standing around it
    still = avail & (f["ball_speed"] >= 0) & (f["ball_speed"] < 1.0)
    f["still_run"] = np.minimum(_run_lengths(still), 3000).astype(np.float32)
    f["still_since"] = np.minimum(_since(~still), CAP).astype(np.float32) * still
    bx, by = use[:, 0], use[:, 1]
    f["d_centre"] = np.nan_to_num(np.hypot(bx - L / 2, by - W / 2), nan=60).astype(np.float32)
    f["d_corner"] = np.nan_to_num(np.hypot(np.minimum(bx, L - bx), np.minimum(by, W - by)), nan=60).astype(np.float32)
    f["d_penspot"] = np.nan_to_num(np.hypot(np.minimum(np.abs(bx - 11), np.abs(L - 11 - bx)), by - W / 2), nan=60).astype(np.float32)
    dists = np.sqrt(((players - use[:, None, :]) ** 2).sum(-1))
    f["n_within9"] = (dists < 9.15).sum(1).astype(np.float32)
    f["n_within20"] = (dists < 20).sum(1).astype(np.float32)

    df = pd.DataFrame(f)
    df["closer"] = np.sign(df["d_away"] - df["d_home"]).astype(np.float32)   # +1 home closer
    df["dd"] = (df["d_away"] - df["d_home"]).clip(-20, 20)

    for w in WINDOWS:
        roll = df[["seen", "ball_speed", "spd_all", "closer", "dd", "last_outside", "spd_p90",
                   "n_within9"]].rolling(
            2 * w + 1, center=True, min_periods=1)
        mean = roll.mean().add_suffix("_m{}".format(w))
        df = pd.concat([df, mean], axis=1)
        df["spd_all_min{}".format(w)] = df["spd_all"].rolling(2 * w + 1, center=True, min_periods=1).min()
    if TOUCHES:
        from possession import touches
        df = pd.concat([df, pd.DataFrame(touches.features(match))], axis=1)
    return df.astype(np.float32)


SIGNED = ["closer", "dd"] + ["closer_m{}".format(w) for w in WINDOWS] + ["dd_m{}".format(w) for w in WINDOWS]


def swap_teams(df):
    """The same frames seen with home and away exchanged."""
    from possession.touches import SIGNED as TOUCH_SIGNED
    out = df.copy()
    for a, b in PAIRS:
        out[a], out[b] = df[b].values, df[a].values
    for c in SIGNED + [c for c in TOUCH_SIGNED if c in df]:
        out[c] = -df[c]
    return out
