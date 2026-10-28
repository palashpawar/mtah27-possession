# Ball possession from automatic tracking: More than a Hack 2027

An entry for [More than a Hack 2027, Phase 1](https://github.com/metrica-sports/more-than-a-hack-2027):
label every frame of FC Barcelona vs Athletic Club `H`, `A` or `D` from automatic, uncorrected
tracking, ranked on possession-level F1 (`spellF1`).

The model is a frame classifier trained only on **other matches**, followed by a decoder that turns
per-frame probabilities into whole possessions. Every modelling choice was made by cross-validation
on those matches. The competition match is only ever predicted. Its public 5-minute ground truth
was used to check that frames and labels line up, never to train, tune or choose between versions.

## Reproduce the submission

```bash
python3 -m venv .venv          # Python 3.9 && .venv/bin/pip install -r requirements.txt
PATH=.venv/bin:$PATH scripts/download_public_data.sh     # ~2.8 GB of public training matches
# put the competition match in data/ -- see data/README.md
.venv/bin/python -m possession.make_submission           # -> evaluation/results/possession_v4.csv
.venv/bin/python evaluation/validate_submission.py evaluation/results/possession_v4.csv
```

The build is deterministic (fixed seeds, pinned versions): it regenerates the submitted file byte
for byte. It takes about 10 minutes on a laptop, most of it parsing the tracking once (cached in
`cache/`).

## Method

**1. Training data: ten public matches, none of them the competition match.**

| Source | Matches | Labels |
|---|---|---|
| [Metrica Sports sample data](https://github.com/metrica-sports/sample-data) | 3 | Derived from the event data (`possession/event_labels.py`): a controlling action (pass, carry, recovery, shot, set piece, ball lost) gives its team the ball; challenges change nothing; a ball out, foul or goal is dead until the next set piece. |
| [IDSSE](https://huggingface.co/datasets/pysport/idsse-data), DFL Bundesliga 2022/23 | 7 | Read from the tracking's own per-frame `BallStatus` and `BallPossession`, the same kind of derivation as the competition's ground truth. DFL possession flips at almost every touch, so spells shorter than 2 s are handed back to the previous team (`possession/smooth_labels.py`). That matches the challenge's "controlled possession" rule and makes spell lengths agree with the Metrica labels (mean 11.4 s vs 11–13 s). |

**2. Make clean tracking look like the competition's automatic tracking** (`model.degrade`).
The public matches have near-perfect tracking. The competition's has long ball losses during play,
missing players, and a ball that is sometimes tracked through stoppages. Training copies are
roughened to match **unlabelled aggregate statistics** of the competition tracking:
- long ball gaps during live play (median ~270 frames, plus ~10% short blips), the same shape as the competition ATD's;
- per stoppage, the ball randomly either blanked or shown resting where it left play, carried back, and sitting on the restart spot;
- player dropouts;
- position error that drifts smoothly. Per-frame jitter would fake running speed, a key stoppage cue.

**3. Frame classifier** (`features.py`, `model.py`). A gradient-boosted classifier
(scikit-learn `HistGradientBoostingClassifier`) gives P(H), P(A), P(D) from identity-free features:
- ball visibility and gap structure, and ball speed;
- distance from the ball to the nearest player of each side, and players within 3 / 9.15 / 20 m;
- player speeds, the ball's distance off the pitch and to restart spots (centre, corners, penalty spots), and how long the ball has been still;
- centred rolling summaries of all of these.

Features and labels are mirrored (home ↔ away) during training and averaged at prediction, so the
model has no home/away bias.

**4. Smart Tagging** (`tags.py`). Each set-piece tag marks a restart. The restart is located in the
tracking (the first ball strike after the tag starts), and the 20 s before it are nudged towards
`D`. The weight was fitted on simulated tags built from the public matches' real restarts.

**5. Decoder** (`model.viterbi`). A Viterbi pass over H/A/D with a fixed cost for every change of
team (20) and every alive↔dead change (40). This is what turns flickering frame guesses into whole
possessions, which is what `spellF1` rewards.

## Validation

5-fold cross-validation over the ten public matches, two held out per fold (`possession/crossval_all.py`),
scored with the organisers' own metric library. Held-out matches are scored clean and degraded.

| | spellF1, all held-out | degraded only | dead-ball F1, degraded |
|---|---|---|---|
| Repository baseline (nearest player, ±2 s vote), Metrica games | 0.32 | 0.24 | – |
| This model, final configuration | **0.589** | **0.539** | **0.869** |

The repository baseline scores 0.24 on degraded held-out matches, close to its official full-match
score (0.2433), which suggests the degradation is realistic.

How each step was decided:

| Change | Evidence (cross-validation) |
|---|---|
| Show the ball through stoppages | degraded spellF1 0.411 → 0.475 |
| Add the 7 IDSSE matches | 0.483 → 0.591 |
| Smart Tagging dead-ball nudge | dead F1 0.654 → 0.699, spellF1 flat |
| Degradation matched to the competition ATD | 0.549 → 0.589 (degraded 0.464 → 0.539) |

On the full competition match the model predicts 272 possessions of mean 14.8 s and 37% dead ball.
Typical professional matches have 40–45% dead time, and frames predicted dead look like truly dead
frames in the public matches (players near-stationary, ~1 m/s vs ~2 m/s in play)
(`possession/diag_dead.py`).

## Repository

```
possession/        the method (entry point: make_submission.py)
scripts/           public-data download
games.py, evaluation/, starter/load_smart_tagging.py
                   the organisers' MIT-licensed code, unchanged (see third_party/)
data/              competition data goes here; never committed
```

Exploratory scripts kept for transparency: `describe.py` and `inspect_schema.py` (aggregate data
summaries), `profile_tags.py` (tracking around Smart Tagging set pieces), `crossval.py` and
`tag_crossval.py` (the earlier validation stages), and `diag_dead.py`. All of them print aggregates
only.

## Data and attribution

- **Competition data** © its owners, licensed for More than a Hack 2027 only. It is not included and not redistributed.
- **Metrica Sports sample data**: [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data).
- **IDSSE**: Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). *An integrated dataset of spatiotemporal and event data in elite soccer.* Scientific Data. CC BY 4.0. Mirror: huggingface.co/datasets/pysport/idsse-data.
- **Organisers' code** (scorer, frame alignment, Smart Tagging loader): MIT, © Metrica Sports. See `third_party/`.
