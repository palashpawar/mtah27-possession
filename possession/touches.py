"""Ball touches and pass chains from tracking alone.

CHALLENGE.md defines a change of possession as the other team gaining *controlled* possession --
"a player keeps the ball for a few seconds" or "a team completes more than two consecutive passes"
-- and says a touch, a deflection or a lost aerial duel does not count. This module finds touches
and chains them, so the classifier can see control rather than mere proximity:

  touch    the ball, seen, within REACH of a player while its direction or speed changes sharply
           (a kick, a reception, a deflection), or held at a player's feet for DRIBBLE frames;
  chain    consecutive touches by the same team, ending at a touch;
  control  a touch whose chain is at least 2 long, or that lasted DWELL frames or more.

Per-frame features say which team touched last and next, how long ago, how long its chain is, and
which team last had control. Team is signed (+1 home, -1 away) so features.swap_teams can mirror it.
"""

import numpy as np

FPS = 25
REACH = 1.5          # m: ball within this of a player counts as contact
LAG = 3              # frames each side for the in / out velocity
TURN = np.cos(np.radians(35))   # direction change of more than 35 degrees
KICK = 4.0           # m/s change in speed
MOVING = 0.15        # m per LAG frames (~1.25 m/s): both sides must move for a turn to count
DRIBBLE = 10         # frames of continuous contact = a touch even without a kink
MERGE = 6            # candidate frames this close belong to one touch
DWELL = 12           # frames: a touch this long is control by itself
CAP = 500


def _interp_short(ball, max_gap=6):
    out = ball.copy()
    seen = ~np.isnan(ball[:, 0])
    if seen.sum() < 2:
        return out
    t = np.arange(len(ball))
    m = np.concatenate(([0], (~seen).astype(np.int8), [0]))
    d = np.diff(m)
    starts, stops = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    for a, b in zip(starts, stops):
        if b - a <= max_gap and a > 0 and b < len(ball):
            for k in range(2):
                out[a:b, k] = np.interp(t[a:b], [a - 1, b], [ball[a - 1, k], ball[b, k]])
    return out


def _runs(mask):
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def detect(match):
    """(frames, teams, dwells): one entry per touch, team 0 home / 1 away, sorted by frame."""
    ball = _interp_short(match.ball)
    n = len(ball)
    seen = ~np.isnan(ball[:, 0])

    # nearest player and its team
    d = np.sqrt(((match.players - ball[:, None, :]) ** 2).sum(-1))
    d = np.where(np.isnan(d), np.inf, d)
    slot = d.argmin(1)
    dmin = d[np.arange(n), slot]
    team = match.team[slot]
    contact = seen & (dmin < REACH)

    # kinks: direction or speed change between the LAG frames before and after
    vin = np.full((n, 2), np.nan, np.float32)
    vout = np.full((n, 2), np.nan, np.float32)
    vin[LAG:] = ball[LAG:] - ball[:-LAG]
    vout[:-LAG] = ball[LAG:] - ball[:-LAG]
    sin, sout = np.hypot(*vin.T), np.hypot(*vout.T)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (vin * vout).sum(1) / (sin * sout)
        turn = (sin > MOVING) & (sout > MOVING) & (cos < TURN)
        kick = np.abs(sout - sin) * FPS / LAG > KICK
    kink = np.nan_to_num(turn | kick, nan=False).astype(bool)

    cand = contact & kink
    a, b = _runs(contact)
    for s, e in zip(a, b):
        if e - s >= DRIBBLE:
            cand[s] = True

    # merge nearby candidates into touches; each touch's team = the side nearest at its first frame
    idx = np.flatnonzero(cand)
    frames, teams, dwells = [], [], []
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] - idx[j] <= MERGE:
            j += 1
        f = idx[i]
        end = f
        while end + 1 < n and contact[end + 1] and team[end + 1] == team[f]:
            end += 1
        frames.append(f)
        teams.append(int(team[f]))
        dwells.append(end - f + 1)
        i = j + 1
    return np.array(frames, int), np.array(teams, int), np.array(dwells, int)


def _signed(team):
    return np.where(team == 0, 1.0, -1.0)


def _last_index(frames, n):
    """For each frame, the index of the last touch at or before it (-1 if none)."""
    out = np.full(n, -1, int)
    out[frames] = np.arange(len(frames))
    return np.maximum.accumulate(out)


def _next_index(frames, n):
    out = np.full(n, len(frames), int)
    out[frames] = np.arange(len(frames))
    return np.minimum.accumulate(out[::-1])[::-1]


def features(match):
    """Dict of per-frame touch features (float32 arrays)."""
    n = match.n_frames
    frames, teams, dwells = detect(match)
    k = len(frames)
    t = np.arange(n)
    out = {}
    if k == 0:
        for name in ("tch_last", "tch_since", "tch_next", "tch_until", "tch_chain_last",
                     "tch_chain_next", "ctrl_last", "ctrl_since", "ctrl_next", "ctrl_until",
                     "tch_bal_m50", "tch_rate_m50"):
            out[name] = np.zeros(n, np.float32)
        return out

    chain = np.ones(k, int)
    for i in range(1, k):
        if teams[i] == teams[i - 1]:
            chain[i] = chain[i - 1] + 1
    fwd = np.ones(k, int)
    for i in range(k - 2, -1, -1):
        if teams[i] == teams[i + 1]:
            fwd[i] = fwd[i + 1] + 1
    ctrl = (chain >= 2) | (dwells >= DWELL)

    s = _signed(teams)
    li, ni = _last_index(frames, n), _next_index(frames, n)
    have_l, have_n = li >= 0, ni < k
    lc, nc = np.clip(li, 0, k - 1), np.clip(ni, 0, k - 1)
    out["tch_last"] = np.where(have_l, s[lc], 0)
    out["tch_since"] = np.where(have_l, np.minimum(t - frames[lc], CAP), CAP)
    out["tch_next"] = np.where(have_n, s[nc], 0)
    out["tch_until"] = np.where(have_n, np.minimum(frames[nc] - t, CAP), CAP)
    out["tch_chain_last"] = np.where(have_l, s[lc] * np.minimum(chain[lc], 6), 0)
    out["tch_chain_next"] = np.where(have_n, s[nc] * np.minimum(fwd[nc], 6), 0)

    cf, ct = frames[ctrl], teams[ctrl]
    if len(cf):
        cs = _signed(ct)
        cli, cni = _last_index(cf, n), _next_index(cf, n)
        hl, hn = cli >= 0, cni < len(cf)
        cl, cn = np.clip(cli, 0, len(cf) - 1), np.clip(cni, 0, len(cf) - 1)
        out["ctrl_last"] = np.where(hl, cs[cl], 0)
        out["ctrl_since"] = np.where(hl, np.minimum(t - cf[cl], CAP), CAP)
        out["ctrl_next"] = np.where(hn, cs[cn], 0)
        out["ctrl_until"] = np.where(hn, np.minimum(cf[cn] - t, CAP), CAP)
    else:
        for name in ("ctrl_last", "ctrl_next"):
            out[name] = np.zeros(n)
        for name in ("ctrl_since", "ctrl_until"):
            out[name] = np.full(n, CAP)

    # touch balance and rate over +/-2 s: who is touching the ball right now
    home = np.zeros(n)
    away = np.zeros(n)
    home[frames[teams == 0]] = 1
    away[frames[teams == 1]] = 1
    w = 101
    ker = np.ones(w)
    h = np.convolve(home, ker, "same")
    a = np.convolve(away, ker, "same")
    out["tch_bal_m50"] = np.where(h + a > 0, (h - a) / np.maximum(h + a, 1), 0)
    out["tch_rate_m50"] = h + a
    return {k2: np.asarray(v, np.float32) for k2, v in out.items()}


SIGNED = ["tch_last", "tch_next", "tch_chain_last", "tch_chain_next", "ctrl_last", "ctrl_next",
          "tch_bal_m50"]
