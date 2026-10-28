"""H/A/D labels for Metrica's public sample games, derived mechanically from their event data.

These games are not the competition match; RULES.md allows labelled data from other games. The
derivation mirrors CHALLENGE.md's possession rules as closely as event data allows:

  - a controlling action (pass, carry, recovery, shot, set piece, or the start of a ball loss --
    the team had it) gives that team the ball from the action's start frame;
  - a challenge changes nothing: contested balls stay with the last team that had them;
  - a ball out, a foul or a goal makes the ball dead until the next set piece restarts play.

Frames before a period's first action or after its last are X (not scored).
"""

import csv
import json
import os
import xml.etree.ElementTree as ET

import numpy as np

H, A, D, X = 0, 1, 2, 3
CONTROL = {"PASS", "CARRY", "RECOVERY", "SHOT", "BALL LOST", "SET PIECE"}
STOPS = {"BALL OUT", "FAULT RECEIVED"}


def _events(game):
    from possession.data import EXTERNAL
    d = os.path.join(EXTERNAL, "Sample_Game_{}".format(game))
    out = []  # (period, start, end, team, type, subtype)
    if game == 3:
        home = next(ET.parse(os.path.join(d, "Sample_Game_3_metadata.xml")).getroot().iter("Team")).get("id")
        for e in json.load(open(os.path.join(d, "Sample_Game_3_events.json")))["data"]:
            sub = e.get("subtypes") or []
            if isinstance(sub, dict):
                sub = [sub]
            subtype = "-".join(s["name"] for s in sub)
            out.append((e["period"], e["start"]["frame"] - 1, e["end"]["frame"] - 1,
                        H if e["team"]["id"] == home else A, e["type"]["name"], subtype))
    else:
        with open(os.path.join(d, "Sample_Game_{0}_RawEventsData.csv".format(game))) as fh:
            for r in csv.DictReader(fh):
                out.append((int(r["Period"]), int(r["Start Frame"]) - 1, int(r["End Frame"]) - 1,
                            H if r["Team"] == "Home" else A, r["Type"], r["Subtype"] or ""))
    return sorted(out, key=lambda e: (e[0], e[1]))


def derive_labels(events, n_frames):
    codes = np.full(n_frames, X, np.int8)
    periods = []
    for p in sorted({e[0] for e in events}):
        evs = [e for e in events if e[0] == p]
        start = min(e[1] for e in evs)
        stop = min(max(e[2] for e in evs) + 1, n_frames)
        changes, dead = [], False
        for _, s, e, team, kind, sub in evs:
            goal = kind == "SHOT" and "GOAL" in sub.upper() and "OUT" not in sub.upper()
            if kind == "SET PIECE":
                changes.append((s, team))
                dead = False
            elif kind in STOPS:
                if not dead:
                    changes.append((s, D))
                dead = True
            elif goal:
                changes.append((s, team))
                changes.append((e, D))
                dead = True
            elif kind in CONTROL:
                changes.append((s, team))
                dead = False
        changes.sort(key=lambda c: c[0])
        for i, (f, code) in enumerate(changes):
            nxt = changes[i + 1][0] if i + 1 < len(changes) else stop
            lo, hi = max(f, start), min(nxt, stop)
            if hi > lo:
                codes[lo:hi] = code
        periods.append((start, stop))
    return codes, periods


def attach_labels(match, game):
    match.labels, match.periods = derive_labels(_events(game), match.n_frames)
