#!/usr/bin/env bash
# Download the public matches the model is trained and validated on (~2.8 GB).
#
#   external/metrica  Metrica Sports sample data, games 1-3 (github.com/metrica-sports/sample-data)
#   external/idsse    IDSSE, 7 DFL matches, CC BY 4.0 (Bassek et al. 2025, Scientific Data;
#                     mirror huggingface.co/datasets/pysport/idsse-data)
#
# The competition match is NOT downloaded here: it is gated. Put it in data/ (see data/README.md).
set -euo pipefail
cd "$(dirname "$0")/.."

M=https://raw.githubusercontent.com/metrica-sports/sample-data/master/data
for p in \
  Sample_Game_1/Sample_Game_1_RawEventsData.csv \
  Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv \
  Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv \
  Sample_Game_2/Sample_Game_2_RawEventsData.csv \
  Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv \
  Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv \
  Sample_Game_3/Sample_Game_3_events.json \
  Sample_Game_3/Sample_Game_3_metadata.xml \
  Sample_Game_3/Sample_Game_3_tracking.txt; do
  mkdir -p "external/metrica/$(dirname "$p")"
  [ -s "external/metrica/$p" ] || curl -sfL -o "external/metrica/$p" "$M/$p"
done

python -m possession.idsse --download
echo "done: external/metrica, external/idsse"
