"""Load tracking into one shape, whatever the source.

A Match is positions in metres on a fixed set of slots, one row per frame, with NaN where nothing
was observed. Two sources produce it:

  - FIFA/EPTS (the competition ATD, and Metrica's public Sample Game 3): metadata XML + raw text.
  - Metrica's public CSV format (Sample Games 1 and 2): one file per team.

Nothing here prints or returns raw values to the caller beyond the arrays themselves; scripts that
run on the competition data report aggregates only.
"""

import csv
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAMES, require  # noqa: E402

CACHE = os.path.join(ROOT, "cache")
EXTERNAL = os.path.join(ROOT, "external", "metrica")
SAMPLE_PITCH = (105.0, 68.0)
HOME, AWAY = 0, 1


@dataclass
class Match:
    name: str
    pitch: tuple                      # (length, width) in metres
    ball: np.ndarray                  # (T, 2) metres, NaN where unseen
    players: np.ndarray               # (T, S, 2) metres, NaN where unseen
    team: np.ndarray                  # (S,) HOME / AWAY per slot
    periods: list = field(default_factory=list)   # [(start, stop)] frames of play, half-open
    labels: np.ndarray = None         # (T,) codes 0=H 1=A 2=D 3=X, external games only

    @property
    def n_frames(self):
        return len(self.ball)


# -- EPTS ------------------------------------------------------------------------------------------

def _epts_specs(meta_path):
    """[(start_frame, end_frame, [group, ...])], each group a list of (player_id|'ball', channel)."""
    root = ET.parse(meta_path).getroot()
    channel_owner = {}
    for ch in root.iter("PlayerChannel"):
        channel_owner[ch.get("id")] = (ch.get("playerId"), ch.get("channelId"))
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        start, end = int(spec.get("startFrame")), int(spec.get("endFrame"))
        registers = list(spec)
        # registers: frameCount, players SplitRegister(;), ball SplitRegister(,)
        player_groups, ball_group = [], []
        for reg in registers[1:]:
            subs = [r for r in reg if r.tag == "SplitRegister"]
            if subs:
                for sub in subs:
                    player_groups.append([channel_owner[r.get("playerChannelId")]
                                          for r in sub if r.tag == "PlayerChannelRef"])
            else:
                ball_group = [r.get("channelId") for r in reg if r.tag == "BallChannelRef"]
        specs.append((start, end, player_groups, ball_group))
    teams = [t.get("id") for t in root.iter("Team")]
    player_team = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    return specs, teams, player_team


def load_epts(name, meta_path, raw_path, pitch):
    specs, teams, player_team = _epts_specs(meta_path)
    players_seen = []
    for _, _, groups, _ in specs:
        for g in groups:
            pid = g[0][0]
            if pid not in players_seen:
                players_seen.append(pid)
    slot = {pid: i for i, pid in enumerate(players_seen)}
    team = np.array([HOME if player_team[p] == teams[0] else AWAY for p in players_seen], np.int8)

    n = max(end for _, end, _, _ in specs)
    ball = np.full((n, 2), np.nan, np.float32)
    players = np.full((n, len(players_seen), 2), np.nan, np.float32)
    length, width = pitch

    spec_i = 0
    with open(raw_path) as fh:
        for line in fh:
            head, body, tail = line.rstrip("\n").split(":")
            f = int(head)
            while not (specs[spec_i][0] <= f <= specs[spec_i][1]):
                spec_i = (spec_i + 1) % len(specs)
            groups = specs[spec_i][2]
            row = f - 1
            for g, text in zip(groups, body.split(";")):
                if text[0] == "N":
                    continue
                vals = text.split(",")
                s = slot[g[0][0]]
                players[row, s, 0] = float(vals[0]) * length
                players[row, s, 1] = float(vals[1]) * width
            if tail[0] != "N":
                vals = tail.split(",")
                ball[row, 0] = float(vals[0]) * length
                ball[row, 1] = float(vals[1]) * width
    return Match(name=name, pitch=pitch, ball=ball, players=players, team=team)


# -- Metrica CSV -----------------------------------------------------------------------------------

def _read_team_csv(path):
    with open(path) as fh:
        rows = list(csv.reader(fh))
    header = rows[2]
    body = np.array([[float(v) if v not in ("", "NaN") else np.nan for v in r] for r in rows[3:]],
                    dtype=np.float64)
    cols = [i for i, h in enumerate(header) if h.startswith("Player")]
    ball_col = header.index("Ball")
    return body, cols, ball_col


def load_metrica_csv(game):
    base = os.path.join(EXTERNAL, "Sample_Game_{0}".format(game), "Sample_Game_{0}_RawTrackingData_".format(game))
    home, hcols, hball = _read_team_csv(base + "Home_Team.csv")
    away, acols, _ = _read_team_csv(base + "Away_Team.csv")
    length, width = SAMPLE_PITCH
    n = len(home)
    frames = home[:, 1].astype(int) - 1
    def xy(body, cols):
        out = np.stack([body[:, cols], body[:, [c + 1 for c in cols]]], axis=-1)
        out[..., 0] *= length
        out[..., 1] *= width
        return out
    players = np.concatenate([xy(home, hcols), xy(away[:n], acols)], axis=1).astype(np.float32)
    ball = np.stack([home[:, hball] * length, home[:, hball + 1] * width], axis=1).astype(np.float32)
    full_p = np.full((frames.max() + 1,) + players.shape[1:], np.nan, np.float32)
    full_b = np.full((frames.max() + 1, 2), np.nan, np.float32)
    full_p[frames], full_b[frames] = players, ball
    team = np.array([HOME] * len(hcols) + [AWAY] * len(acols), np.int8)
    return Match(name="metrica{}".format(game), pitch=SAMPLE_PITCH, ball=full_b, players=full_p,
                 team=team)


# -- entry points with caching ---------------------------------------------------------------------

def _cached(key, build):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, key + ".npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return Match(name=str(z["name"]), pitch=tuple(z["pitch"]), ball=z["ball"],
                     players=z["players"], team=z["team"])
    m = build()
    np.savez(path, name=m.name, pitch=np.array(m.pitch), ball=m.ball, players=m.players,
             team=m.team)
    return m


def load_competition():
    """The BAR-ATH tactical ATD. Periods are the two graded halves, as published in DATA.md."""
    from games import half_video_blocks
    meta, raw = require(["atd_meta", "atd_raw"])
    m = _cached("BAR-ATH_tactical", lambda: load_epts("BAR-ATH", meta, raw,
                                                       tuple(GAMES["BAR-ATH"]["pitch"])))
    m.periods = [(a, b + 1) for _, a, b in half_video_blocks("BAR-ATH")]
    return m


def load_sample(game):
    """Metrica public Sample Game 1, 2 or 3, with H/A/D labels derived from its event data."""
    from possession.event_labels import attach_labels
    if game == 3:
        d = os.path.join(EXTERNAL, "Sample_Game_3")
        m = _cached("metrica3", lambda: load_epts(
            "metrica3", os.path.join(d, "Sample_Game_3_metadata.xml"),
            os.path.join(d, "Sample_Game_3_tracking.txt"), SAMPLE_PITCH))
    else:
        m = _cached("metrica{}".format(game), lambda: load_metrica_csv(game))
    attach_labels(m, game)
    return m
