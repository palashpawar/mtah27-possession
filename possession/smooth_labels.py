"""Turn touch-level possession flags into controlled possession, per CHALLENGE.md.

DFL's BallPossession flips at nearly every touch. The challenge counts a change only when the new
team gains control; touches, deflections and contested balls leave possession with the last team.
merge_short() implements that: inside each alive stretch, a spell of one team shorter than
`min_frames` is handed back to the team that had the ball before it (or, at the start of a
stretch, to the team after it).
"""
import numpy as np

H, A, D, X = 0, 1, 2, 3


def merge_short(labels, min_frames):
    out = labels.copy()
    if min_frames <= 0:
        return out
    changed = True
    while changed:
        changed = False
        m = np.concatenate(([-1], out, [-1]))
        edges = np.flatnonzero(m[1:] != m[:-1])
        for a, b in zip(edges[:-1], edges[1:]):
            code = out[a]
            if code not in (H, A) or b - a >= min_frames:
                continue
            prev = out[a - 1] if a > 0 else -1
            nxt = out[b] if b < len(out) else -1
            fill = prev if prev in (H, A) else nxt if nxt in (H, A) else None
            if fill is not None and fill != code:
                out[a:b] = fill
                changed = True
    return out
