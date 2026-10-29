# Ball possession from automatic tracking

Entry for [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027): label every
frame of FC Barcelona vs Athletic Club `H`, `A` or `D` from automatic tracking.

A classifier trained on ten **other** public matches labels each frame; a decoder joins the frames
into whole possessions. The competition match is only ever predicted. Its public 5-minute ground
truth was used to check alignment, never to train or tune.

## Reproduce

```bash
python3 -m venv .venv                                    # Python 3.9
.venv/bin/pip install -r requirements.txt
PATH=.venv/bin:$PATH scripts/download_public_data.sh     # ~2.8 GB of public matches
# put the competition match in data/ (see data/README.md)
.venv/bin/python -m possession.make_submission           # -> evaluation/results/possession_v5.csv
```

Deterministic: it regenerates the submitted file byte for byte (about 10 minutes).

## Method

1. **Training data.** 3 [Metrica sample](https://github.com/metrica-sports/sample-data) matches
   (labels from events) and 7 [IDSSE](https://huggingface.co/datasets/pysport/idsse-data) DFL
   matches (labels from the tracking's own dead/possession flags).
2. **Realistic tracking.** Training copies are roughened to match unlabelled statistics of the
   competition tracking: long ball losses, a ball sometimes tracked through stoppages, missing
   players, smooth position error (`model.degrade`).
3. **Frame classifier.** Gradient boosting on 74 identity-free features: ball visibility and speed,
   distance to the nearest player of each side, player speeds, and touches. Two same-team touches
   in a row count as control, following the challenge's definition (`features.py`, `touches.py`).
4. **Smart Tagging.** The 20 s before each tagged restart are nudged towards `D` (`tags.py`).
5. **Decoder.** A Viterbi pass charges a cost for each change of team (10) or of alive/dead state
   (40), turning frame guesses into whole possessions (`model.viterbi`).

## Validation

5-fold cross-validation over the ten public matches, scored with the organisers' metric code
(`possession/crossval_all.py`).

| | spellF1 | spellF1, degraded tracking |
|---|---|---|
| Repository baseline | 0.32 | 0.24 |
| This model | **0.663** | **0.605** |

On the competition match: 292 possessions of mean 13.7 s, 37% of frames dead.

## Data and licences

- Competition data: licensed for the competition only; not included.
- Metrica sample data: [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data).
- IDSSE: Bassek, Rein, Weber & Memmert (2025), *Scientific Data*, CC BY 4.0.
- `games.py`, `evaluation/`, `starter/load_smart_tagging.py`: the organisers' MIT code, unchanged (see `third_party/`).
