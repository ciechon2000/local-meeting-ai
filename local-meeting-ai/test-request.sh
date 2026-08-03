#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
API_KEY="${API_KEY:?Ustaw API_KEY}"
AUDIO_FILE="${1:?Podaj ścieżkę do pliku audio}"

curl --fail-with-body \
  -X POST "${BASE_URL}/transcribe-diarize" \
  -H "Authorization: Bearer ${API_KEY}" \
  -F "file=@${AUDIO_FILE}" \
  -F "language=pl" \
  -F "min_speakers=2" \
  -F "max_speakers=8" \
  -F "include_words=false"
