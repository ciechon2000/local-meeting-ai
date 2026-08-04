# Local Meeting AI 0.2.0

Lokalny serwis FastAPI do transkrypcji i prawdziwej diaryzacji nagrań. Wersja 0.2.0 została przygotowana pod ograniczone środowiska Any Docker.

## Najważniejsza zmiana

Serwis nie wymaga bind mountu ani katalogu `/data` do pierwszego uruchomienia. Domyślnie korzysta z:

- `/cache/models` — cache modeli,
- `/tmp/local-meeting-ai` — pliki tymczasowe.

Jeśli wskazany katalog nie jest zapisywalny, entrypoint automatycznie przełącza się na katalog w `/tmp`.

## Endpointy

- `GET /health`
- `POST /warmup`
- `POST /transcribe-diarize`
- `GET /docs`

## Pierwsze wdrożenie w Trooper Any Docker

Na pierwszy test nie konfiguruj wolumenu:

- image: `TWOJ_LOGIN/local-meeting-ai:v0.2.0`
- container port: `8000`
- host network: `NO`
- volume/local_data_dir/docker_data_dir: puste
- GPU: `all`

Zmienne:

- `API_KEY`
- `HF_TOKEN`
- `DEVICE=cuda`
- `WHISPER_MODEL=large-v3`
- `COMPUTE_TYPE=float16`
- `PRELOAD_MODELS=false`

Po potwierdzeniu działania `/health` można dodać trwały mount hosta do `/cache`, ale źródłowy katalog hosta musi istnieć przed uruchomieniem kontenera.

## Przykładowe sprawdzenie

```bash
curl http://127.0.0.1:PORT_HOSTA/health
```

## Warmup

```bash
curl -X POST \
  -H "Authorization: Bearer $API_KEY" \
  http://127.0.0.1:PORT_HOSTA/warmup
```
