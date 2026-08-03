# Local Meeting AI

Lokalna usługa API do transkrypcji i prawdziwej diaryzacji nagrań spotkań. Audio jest przetwarzane przez WhisperX oraz `pyannote/speaker-diarization-community-1`.

## Endpointy

- `GET /health` — stan usługi i GPU, bez autoryzacji.
- `POST /warmup` — pobranie/załadowanie modeli, wymaga Bearer tokenu.
- `POST /transcribe-diarize` — transkrypcja, timestampy i mówcy, wymaga Bearer tokenu.
- `GET /docs` — dokumentacja Swagger.

## Wymagane działania w Hugging Face

1. Załóż konto Hugging Face.
2. Otwórz model `pyannote/speaker-diarization-community-1`.
3. Zaakceptuj warunki dostępu.
4. Utwórz token typu **Read**.
5. Przekaż token jako zmienną `HF_TOKEN`.

## Uruchomienie lokalne przez Docker

```bash
docker build -t local-meeting-ai .

docker run --rm --gpus all \
  -p 8000:8000 \
  -v local-meeting-models:/data \
  -e API_KEY='bardzo-dlugi-losowy-sekret' \
  -e HF_TOKEN='hf_xxxxxxxxxxxxxxxxx' \
  -e PYANNOTE_METRICS_ENABLED=0 \
  local-meeting-ai
```

Test stanu:

```bash
curl http://localhost:8000/health
```

Pierwsze pobranie modeli:

```bash
curl -X POST http://localhost:8000/warmup \
  -H 'Authorization: Bearer bardzo-dlugi-losowy-sekret'
```

Test nagrania:

```bash
API_KEY='bardzo-dlugi-losowy-sekret' \
BASE_URL='http://localhost:8000' \
./test-request.sh ./spotkanie.mp3
```

## Publikacja obrazu w GitHub Container Registry

Workflow `.github/workflows/docker-build.yml` buduje obraz przy każdym pushu do `main`.

Po zakończonym buildzie obraz będzie dostępny jako:

```text
ghcr.io/NAZWA-KONTA/NAZWA-REPOZYTORIUM:latest
```

W ustawieniach pakietu GitHub ustaw widoczność odpowiednią dla Trooper.AI. Najprostszy pierwszy test to pakiet publiczny. Kod nie zawiera tokenów ani dokumentów klienta.

## Zmienne środowiskowe

| Zmienna | Domyślna wartość | Znaczenie |
|---|---:|---|
| `API_KEY` | brak | Sekret Bearer wymagany przez endpointy robocze |
| `HF_TOKEN` | brak | Token Read do pobrania modelu pyannote |
| `DEVICE` | `cuda` przy dostępnym GPU | `cuda` lub `cpu` |
| `WHISPER_MODEL` | `large-v3` | Model ASR |
| `COMPUTE_TYPE` | `float16` na GPU | Typ obliczeń CTranslate2 |
| `BATCH_SIZE` | `8` | Domyślny batch WhisperX |
| `MODEL_CACHE` | `/data/models` | Trwały cache modeli |
| `TMP_DIR` | `/data/tmp` | Tymczasowe nagrania, usuwane po żądaniu |
| `MAX_FILE_MB` | `2048` | Maksymalny rozmiar pliku |
| `PRELOAD_MODELS` | `false` | Ładowanie modeli podczas startu |
| `PYANNOTE_METRICS_ENABLED` | `0` | Wyłączenie telemetrii pyannote |

## Konfiguracja Any Docker w Trooper.AI

Przykład po opublikowaniu obrazu:

```text
container_name: local_meeting_ai
docker_reprotag: ghcr.io/NAZWA-KONTA/NAZWA-REPOZYTORIUM:latest
docker_port: 8000
gpus: all
host_network: YES
keep_alive: NO
local_data_dir: /home/trooperai/local_meeting_ai_data
docker_data_dir: /data
```

Zmienne do przekazania w `start_args`:

```text
--env API_KEY=DLUGI_LOSOWY_SEKRET
--env HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
--env DEVICE=cuda
--env WHISPER_MODEL=large-v3
--env COMPUTE_TYPE=float16
--env BATCH_SIZE=8
--env PRELOAD_MODELS=false
--env PYANNOTE_METRICS_ENABLED=0
--env HF_HUB_DISABLE_TELEMETRY=1
```

Po pierwszym prawidłowym pobraniu modeli można rozważyć ustawienie `HF_HUB_OFFLINE=1`, ale dopiero po potwierdzeniu, że wszystkie modele, w tym polski model alignmentu, znajdują się w trwałym cache.

## Prywatność

- audio jest zapisywane wyłącznie w katalogu tymczasowym i usuwane po odpowiedzi;
- transkrypcja nie jest zapisywana do bazy;
- telemetria pyannote jest wyłączona;
- modele są pobierane do trwałego katalogu `/data/models`;
- endpoint roboczy jest chroniony Bearer tokenem;
- usługa uruchamia jeden worker, aby nie dublować modeli w pamięci GPU.
