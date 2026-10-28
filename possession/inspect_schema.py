"""Structural summary of the raw ATD: field counts per group. Prints no values."""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from games import require  # noqa: E402

raw = require(["atd_raw"])[0]
parts_n = collections.Counter()
ball_fields = collections.Counter()
player_fields = collections.Counter()
groups = collections.Counter()
with open(raw) as fh:
    for i, line in enumerate(fh):
        p = line.rstrip("\n").split(":")
        parts_n[len(p)] += 1
        if len(p) >= 3:
            ball_fields[len(p[2].split(","))] += 1
            g = p[1].split(";")
            groups[len(g)] += 1
            player_fields[len(g[0].split(","))] += 1
print("colon-separated parts per line:", dict(parts_n))
print("fields in ball group:", dict(ball_fields))
print("fields per player group:", dict(player_fields))
print("player groups per line:", dict(groups))
