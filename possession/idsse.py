"""IDSSE: seven 2022/23 Bundesliga / 2. Bundesliga matches of official DFL tracking (CC BY 4.0).

Bassek, Rein, Weber & Memmert (2025), "An integrated dataset of spatiotemporal and event data in
elite soccer", Scientific Data. Mirrored at huggingface.co/datasets/pysport/idsse-data.

The DFL ball track carries a per-frame BallStatus (1 alive, 0 dead) and BallPossession (1 home,
2 guest). Labels are read straight off them -- D where the ball is dead, otherwise the possessing
team -- which is the same derivation CHALLENGE.md describes for the competition's ground truth.
No event data is needed.

    .venv/bin/python -m possession.idsse --download     # ~2.6 GB into external/idsse
"""

import os
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

from possession.data import CACHE, ROOT, Match

DIR = os.path.join(ROOT, "external", "idsse")
URL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
MATCHES = {
    "J03WMX": "DFL-COM-000001", "J03WN1": "DFL-COM-000001", "J03WOH": "DFL-COM-000002",
    "J03WOY": "DFL-COM-000002", "J03WPY": "DFL-COM-000002", "J03WQQ": "DFL-COM-000002",
    "J03WR9": "DFL-COM-000002",
}
H, A, D, X = 0, 1, 2, 3
HALF_GAP = 250  # frames of X between the halves, so rolling features never span half-time


def _files(match_id):
    stem = "{}_DFL-MAT-{}.xml".format(MATCHES[match_id], match_id)
    return (os.path.join(DIR, "DFL_02_01_matchinformation_" + stem),
            os.path.join(DIR, "DFL_04_03_positions_raw_observed_" + stem))


def download():
    os.makedirs(DIR, exist_ok=True)
    for m in MATCHES:
        for path in _files(m):
            if not os.path.exists(path):
                subprocess.check_call(["curl", "-sfL", "-o", path, URL + os.path.basename(path)])


def _meta(path):
    root = ET.parse(path).getroot()
    general = next(root.iter("General"))
    env = next(root.iter("Environment"), None)
    pitch = (float(env.get("PitchX", 105)), float(env.get("PitchY", 68))) if env is not None else (105.0, 68.0)
    return general.get("HomeTeamId"), general.get("GuestTeamId"), pitch


def _parse(match_id):
    meta_path, pos_path = _files(match_id)
    home, guest, pitch = _meta(meta_path)
    L, W = pitch

    # pass 1: frame range per half and the people of each team
    ranges, people = {}, {}
    current = None
    for event, el in ET.iterparse(pos_path, events=("start", "end")):
        if event == "start" and el.tag == "FrameSet":
            current = (el.get("GameSection"), el.get("TeamId"), el.get("PersonId"))
            if current[1] in (home, guest):
                people.setdefault(current[2], H if current[1] == home else A)
        elif event == "end" and el.tag == "Frame":
            if current[1] == "BALL":
                n = int(el.get("N"))
                lo, hi = ranges.get(current[0], (n, n))
                ranges[current[0]] = (min(lo, n), max(hi, n))
            el.clear()

    halves = sorted(ranges, key=lambda s: ranges[s][0])
    offset, periods, base = 0, [], {}
    for s in halves:
        lo, hi = ranges[s]
        base[s] = offset - lo
        periods.append((offset, offset + hi - lo + 1))
        offset += hi - lo + 1 + HALF_GAP
    n = offset
    slots = {p: i for i, p in enumerate(people)}
    team = np.array([people[p] for p in people], np.int8)
    ball = np.full((n, 2), np.nan, np.float32)
    players = np.full((n, len(slots), 2), np.nan, np.float32)
    labels = np.full(n, X, np.int8)

    # pass 2: positions (centred metres -> 0..L, 0..W) and the ball's status / possession
    for event, el in ET.iterparse(pos_path, events=("start", "end")):
        if event == "start" and el.tag == "FrameSet":
            current = (el.get("GameSection"), el.get("TeamId"), el.get("PersonId"))
        elif event == "end" and el.tag == "Frame":
            sec, tid, pid = current
            if sec in base and (tid == "BALL" or pid in slots):
                i = int(el.get("N")) + base[sec]
                if 0 <= i < n:
                    x, y = float(el.get("X")) + L / 2, float(el.get("Y")) + W / 2
                    if tid == "BALL":
                        ball[i] = (x, y)
                        if el.get("BallStatus") == "0":
                            labels[i] = D
                        else:
                            labels[i] = H if el.get("BallPossession") == "1" else A
                    else:
                        players[i, slots[pid]] = (x, y)
            el.clear()
    return Match(name="idsse_" + match_id, pitch=pitch, ball=ball, players=players, team=team,
                 periods=periods, labels=labels)


def load(match_id):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "idsse_{}.npz".format(match_id))
    if os.path.exists(path):
        z = np.load(path)
        return Match(name="idsse_" + match_id, pitch=tuple(z["pitch"]), ball=z["ball"],
                     players=z["players"], team=z["team"],
                     periods=[tuple(p) for p in z["periods"]], labels=z["labels"])
    m = _parse(match_id)
    np.savez(path, pitch=np.array(m.pitch), ball=m.ball, players=m.players, team=m.team,
             periods=np.array(m.periods), labels=m.labels)
    return m


if __name__ == "__main__":
    if "--download" in sys.argv:
        download()
