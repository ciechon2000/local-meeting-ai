from __future__ import annotations

import asyncio
import gc
import logging
import os
import secrets
import shutil
import tempfile
import time
from importlib import metadata
from pathlib import Path
from typing import Annotated, Any

import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.responses import ORJSONResponse

LOGGER = logging.getLogger("local-meeting-ai")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

APP_VERSION = "0.2.0"
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu").strip().lower()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3").strip()
COMPUTE_TYPE = os.getenv(
    "COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8"
).strip()
BATCH_SIZE = max(1, int(os.getenv("BATCH_SIZE", "8")))
MODEL_CACHE = Path(os.getenv("MODEL_CACHE", "/cache/models")).resolve()
TMP_DIR = Path(os.getenv("TMP_DIR", "/tmp/local-meeting-ai")).resolve()
MAX_FILE_MB = max(1, int(os.getenv("MAX_FILE_MB", "2048")))
API_KEY = os.getenv("API_KEY", "").strip()
HF_TOKEN = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or "").strip()
DIARIZATION_MODEL = os.getenv(
    "DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1"
).strip()
PRELOAD_MODELS = os.getenv("PRELOAD_MODELS", "false").lower() in {"1", "true", "yes"}

ALLOWED_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".ogg",
    ".webm",
    ".flac",
    ".aac",
}

def ensure_writable_directory(path: Path, fallback: Path) -> Path:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return path
    except Exception:
        LOGGER.warning("Katalog %s nie jest zapisywalny; używam %s", path, fallback)
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


MODEL_CACHE = ensure_writable_directory(
    MODEL_CACHE, Path("/tmp/local-meeting-ai-models")
)
TMP_DIR = ensure_writable_directory(
    TMP_DIR, Path("/tmp/local-meeting-ai-runtime")
)

# Wszystkie cache modeli trafiają do trwałego wolumenu.
os.environ.setdefault("HF_HOME", str(MODEL_CACHE / "huggingface"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(MODEL_CACHE / "huggingface" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_CACHE / "transformers"))
os.environ.setdefault("TORCH_HOME", str(MODEL_CACHE / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(MODEL_CACHE / "xdg"))
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def require_bearer_token(
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API_KEY nie został skonfigurowany na serwerze.",
        )

    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Wymagany nagłówek Authorization: Bearer <token>.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not secrets.compare_digest(token, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Nieprawidłowy token API.",
        )


class ModelManager:
    def __init__(self) -> None:
        self.whisperx: Any | None = None
        self.asr_model: Any | None = None
        self.diarization_model: Any | None = None
        self.processing_lock = asyncio.Lock()

    def _import_whisperx(self) -> Any:
        if self.whisperx is None:
            import whisperx  # import jest ciężki, dlatego wykonujemy go leniwie

            self.whisperx = whisperx
        return self.whisperx

    def ensure_asr(self) -> Any:
        if self.asr_model is not None:
            return self.asr_model

        whisperx = self._import_whisperx()
        LOGGER.info(
            "Ładowanie modelu ASR %s na urządzeniu %s (%s)",
            WHISPER_MODEL,
            DEVICE,
            COMPUTE_TYPE,
        )
        self.asr_model = whisperx.load_model(
            WHISPER_MODEL,
            DEVICE,
            compute_type=COMPUTE_TYPE,
            language=None,
            vad_method="silero",
            download_root=str(MODEL_CACHE / "whisper"),
            use_auth_token=HF_TOKEN or None,
        )
        return self.asr_model

    def ensure_diarization(self) -> Any:
        if self.diarization_model is not None:
            return self.diarization_model

        if not HF_TOKEN:
            raise RuntimeError(
                "Brak HF_TOKEN. Zaakceptuj warunki modelu pyannote i ustaw token Hugging Face."
            )

        self._import_whisperx()
        from whisperx.diarize import DiarizationPipeline

        LOGGER.info("Ładowanie modelu diaryzacji %s", DIARIZATION_MODEL)
        self.diarization_model = DiarizationPipeline(
            model_name=DIARIZATION_MODEL,
            token=HF_TOKEN,
            device=DEVICE,
            cache_dir=str(MODEL_CACHE / "huggingface"),
        )
        return self.diarization_model

    def warmup_sync(self) -> dict[str, Any]:
        started = time.perf_counter()
        self.ensure_asr()
        self.ensure_diarization()
        return {
            "success": True,
            "asr_loaded": self.asr_model is not None,
            "diarization_loaded": self.diarization_model is not None,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    def transcribe_sync(
        self,
        file_path: Path,
        language: str | None,
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
        request_batch_size: int,
        include_words: bool,
    ) -> dict[str, Any]:
        whisperx = self._import_whisperx()
        started = time.perf_counter()

        audio = whisperx.load_audio(str(file_path))
        duration = round(float(len(audio)) / 16000.0, 3)

        asr_model = self.ensure_asr()
        transcription = asr_model.transcribe(
            audio,
            batch_size=request_batch_size,
            language=language or None,
        )
        resolved_language = str(transcription.get("language") or language or "pl")

        align_model = None
        try:
            align_model, align_metadata = whisperx.load_align_model(
                language_code=resolved_language,
                device=DEVICE,
                model_dir=str(MODEL_CACHE / "alignment"),
            )
            aligned = whisperx.align(
                transcription.get("segments", []),
                align_model,
                align_metadata,
                audio,
                DEVICE,
                return_char_alignments=False,
            )
        finally:
            if align_model is not None:
                del align_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        diarizer = self.ensure_diarization()
        diarization_kwargs: dict[str, int] = {}
        if num_speakers is not None:
            diarization_kwargs["num_speakers"] = num_speakers
        else:
            if min_speakers is not None:
                diarization_kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarization_kwargs["max_speakers"] = max_speakers

        diarization = diarizer(audio, **diarization_kwargs)
        final_result = whisperx.assign_word_speakers(
            diarization,
            aligned,
            fill_nearest=True,
        )

        segments = self._normalize_segments(
            final_result.get("segments", []),
            include_words=include_words,
        )
        speakers = []
        for segment in segments:
            speaker = segment["speaker"]
            if speaker not in speakers:
                speakers.append(speaker)

        speaker_labels = {
            speaker: f"Mówca {index + 1}" for index, speaker in enumerate(speakers)
        }
        for segment in segments:
            segment["speaker_label"] = speaker_labels[segment["speaker"]]

        transcript = "\n".join(
            f"[{format_timestamp(segment['start'])}-{format_timestamp(segment['end'])}] "
            f"{segment['speaker_label']}: {segment['text']}"
            for segment in segments
        )

        return {
            "success": True,
            "language": resolved_language,
            "duration": duration,
            "speaker_count": len(speakers),
            "speaker_mapping": speaker_labels,
            "text": " ".join(segment["text"] for segment in segments).strip(),
            "transcript": transcript,
            "segments": segments,
            "metadata": {
                "whisper_model": WHISPER_MODEL,
                "diarization_model": DIARIZATION_MODEL,
                "device": DEVICE,
                "compute_type": COMPUTE_TYPE,
                "batch_size": request_batch_size,
                "processing_seconds": round(time.perf_counter() - started, 3),
                "whisperx_version": package_version("whisperx"),
                "torch_version": torch.__version__,
            },
        }

    @staticmethod
    def _normalize_segments(
        raw_segments: list[dict[str, Any]],
        include_words: bool,
    ) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        for raw in raw_segments:
            text = str(raw.get("text") or "").strip()
            if not text:
                continue

            segment: dict[str, Any] = {
                "start": round(float(raw.get("start") or 0.0), 3),
                "end": round(float(raw.get("end") or 0.0), 3),
                "speaker": str(raw.get("speaker") or "UNKNOWN"),
                "text": text,
            }
            if include_words:
                segment["words"] = [
                    {
                        key: value
                        for key, value in word.items()
                        if key in {"word", "start", "end", "score", "speaker"}
                    }
                    for word in raw.get("words", [])
                ]
            segments.append(segment)

        segments.sort(key=lambda item: (item["start"], item["end"]))
        return segments


manager = ModelManager()
app = FastAPI(
    title="Local Meeting AI",
    version=APP_VERSION,
    default_response_class=ORJSONResponse,
    docs_url="/docs",
    redoc_url=None,
)


@app.on_event("startup")
async def startup_event() -> None:
    if PRELOAD_MODELS:
        LOGGER.info("PRELOAD_MODELS=true — rozpoczynam ładowanie modeli")
        try:
            await asyncio.to_thread(manager.warmup_sync)
        except Exception:
            LOGGER.exception("Nie udało się załadować modeli podczas startu")


@app.get("/health")
def health() -> dict[str, Any]:
    gpu_name = None
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "cuda"

    return {
        "status": "ok",
        "service": "local-meeting-ai",
        "version": APP_VERSION,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
        "configured_device": DEVICE,
        "api_key_configured": bool(API_KEY),
        "hf_token_configured": bool(HF_TOKEN),
        "asr_loaded": manager.asr_model is not None,
        "diarization_loaded": manager.diarization_model is not None,
        "runtime": {
            "model_cache": str(MODEL_CACHE),
            "tmp_dir": str(TMP_DIR),
            "model_cache_writable": os.access(MODEL_CACHE, os.W_OK),
            "tmp_dir_writable": os.access(TMP_DIR, os.W_OK),
        },
        "versions": {
            "python": package_version("pip"),
            "torch": torch.__version__,
            "whisperx": package_version("whisperx"),
            "pyannote_audio": package_version("pyannote.audio"),
        },
    }


@app.post("/warmup", dependencies=[Depends(require_bearer_token)])
async def warmup() -> dict[str, Any]:
    async with manager.processing_lock:
        try:
            return await asyncio.to_thread(manager.warmup_sync)
        except Exception as exc:
            LOGGER.exception("Błąd ładowania modeli")
            raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/transcribe-diarize", dependencies=[Depends(require_bearer_token)])
async def transcribe_diarize(
    file: Annotated[UploadFile, File(description="Plik audio lub wideo")],
    language: Annotated[str, Form()] = "pl",
    num_speakers: Annotated[int | None, Form()] = None,
    min_speakers: Annotated[int | None, Form()] = 2,
    max_speakers: Annotated[int | None, Form()] = 8,
    batch_size: Annotated[int, Form()] = BATCH_SIZE,
    include_words: Annotated[bool, Form()] = False,
) -> dict[str, Any]:
    validate_speaker_limits(num_speakers, min_speakers, max_speakers)
    batch_size = max(1, min(int(batch_size), 32))
    language = language.strip().lower() or "pl"

    suffix = Path(file.filename or "audio").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Nieobsługiwany format {suffix or '(brak rozszerzenia)'}. Dozwolone: {sorted(ALLOWED_EXTENSIONS)}",
        )

    request_dir = Path(tempfile.mkdtemp(prefix="meeting_", dir=TMP_DIR))
    target = request_dir / f"input{suffix}"

    try:
        size = await save_upload_with_limit(file, target, MAX_FILE_MB)
        LOGGER.info(
            "Rozpoczynam analizę pliku %s (%.2f MB)",
            file.filename,
            size / (1024 * 1024),
        )

        async with manager.processing_lock:
            result = await asyncio.to_thread(
                manager.transcribe_sync,
                target,
                language,
                num_speakers,
                min_speakers,
                max_speakers,
                batch_size,
                include_words,
            )

        result["metadata"]["original_filename"] = file.filename
        result["metadata"]["file_size_bytes"] = size
        return result
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Błąd transkrypcji i diaryzacji")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Nie udało się przetworzyć nagrania: {exc}",
        ) from exc
    finally:
        await file.close()
        shutil.rmtree(request_dir, ignore_errors=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def validate_speaker_limits(
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> None:
    values = [value for value in [num_speakers, min_speakers, max_speakers] if value is not None]
    if any(value < 1 or value > 50 for value in values):
        raise HTTPException(
            status_code=422,
            detail="Liczba mówców musi mieścić się w przedziale 1–50.",
        )
    if num_speakers is None and min_speakers and max_speakers and min_speakers > max_speakers:
        raise HTTPException(
            status_code=422,
            detail="min_speakers nie może być większe niż max_speakers.",
        )


async def save_upload_with_limit(upload: UploadFile, destination: Path, limit_mb: int) -> int:
    limit_bytes = limit_mb * 1024 * 1024
    total = 0
    with destination.open("wb") as output:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > limit_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Plik przekracza limit {limit_mb} MB.",
                )
            output.write(chunk)
    if total == 0:
        raise HTTPException(status_code=422, detail="Przesłany plik jest pusty.")
    return total


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
