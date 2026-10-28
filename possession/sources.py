"""Every labelled training match, from both public sources.

  - Metrica Sample Games 1-3: labels derived from event data (possession/event_labels.py).
  - IDSSE, 7 DFL matches: labels from the tracking's own BallStatus / BallPossession flags,
    with touch-level flips under MIN_SPELL_S merged back (possession/smooth_labels.py). 2 s makes
    their spell lengths match the Metrica labels (mean 11.4 s vs 11-13 s).
"""
from possession import idsse
from possession.data import load_sample
from possession.smooth_labels import merge_short

MIN_SPELL_S = 2


def metrica():
    return [load_sample(g) for g in (1, 2, 3)]


def dfl():
    out = []
    for k in idsse.MATCHES:
        m = idsse.load(k)
        m.labels = merge_short(m.labels, int(MIN_SPELL_S * 25))
        out.append(m)
    return out


def all_matches():
    return metrica() + dfl()
