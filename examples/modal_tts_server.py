"""
OpenAI-compatible TTS endpoint for Modal.com — fixed single voice.

Deploy faster-qwen3-tts as a serverless endpoint on Modal.com with:
  • OpenAI POST /v1/audio/speech API (wav/pcm streaming, mp3)
  • Single fixed voice reference — full ICL mode (reference WAV in context)
  • Language inferred automatically — works with Thai and other languages not in
    the model's native training set because the reference audio carries the
    acoustic/phonetic context the model needs
  • Modal Volume for cached model weights — no HuggingFace re-download on cold start
  • CUDA graphs via faster-qwen3-tts for real-time inference
  • Scales to zero when idle; new containers warm up in seconds

Prerequisites
-------------
    pip install modal
    modal token new          # authenticate with Modal (free at modal.com)

Quick deploy
------------
    # 1. Download model weights to the persistent volume (once per model)
    modal run examples/modal_tts_server.py::download_model

    # 2. Upload your voice reference WAV (once, ~5–30 s of the target voice)
    modal run examples/modal_tts_server.py --ref-audio /path/to/voice.wav

    # 3. Deploy the endpoint
    modal deploy examples/modal_tts_server.py

    # 4. Call the API (replace URL with yours from "modal deploy" output)
    curl https://<workspace>--faster-qwen3-tts-tts-service-web.modal.run/v1/audio/speech \\
      -H "Content-Type: application/json" \\
      -d '{"model":"tts-1","input":"Hello from Modal!","voice":"alloy","response_format":"wav"}' \\
      --output speech.wav

Configuration
-------------
Change MODEL_NAME or GPU_TYPE below to tune quality / cost.
The 0.6B model is fastest; the 1.7B model has higher quality.
"""

import asyncio
import io
import queue
import struct
import threading
from typing import AsyncGenerator

import modal
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration — edit these to match your needs
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"   # or "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
GPU_TYPE = "A10G"       # A10G (24 GB) is cost-effective; L4 is a cheaper alternative; A100 for maximum throughput
SCALEDOWN_WINDOW = 300  # seconds to keep container warm after the last request

# Paths inside the Modal Volume
VOLUME_DIR = "/data"
MODEL_PATH = f"{VOLUME_DIR}/model"
REF_AUDIO_PATH = f"{VOLUME_DIR}/reference.wav"

# ---------------------------------------------------------------------------
# Modal primitives
# ---------------------------------------------------------------------------

volume = modal.Volume.from_name("faster-qwen3-tts-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")            # needed by pydub for MP3 encoding
    .pip_install(
        "faster-qwen3-tts>=0.2.4",
        "fastapi[standard]>=0.100.0",
        "uvicorn[standard]>=0.24.0",
        "huggingface-hub>=0.20",
        "soundfile",
        "pydub",
        # torch / numpy / transformers pulled in transitively
    )
)

app = modal.App("faster-qwen3-tts")

# ---------------------------------------------------------------------------
# Step 1 — Download model weights (run once)
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    volumes={VOLUME_DIR: volume},
    timeout=3600,
)
def download_model(model_name: str = MODEL_NAME):
    """Download Qwen3-TTS weights from HuggingFace Hub into the Modal Volume.

    Run once before deploying the endpoint:
        modal run examples/modal_tts_server.py::download_model
    """
    from huggingface_hub import snapshot_download

    print(f"Downloading {model_name} → {MODEL_PATH}")
    snapshot_download(
        repo_id=model_name,
        local_dir=MODEL_PATH,
        # skip large binaries that are unused by this library
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
    volume.commit()
    print("Model download complete.")


# ---------------------------------------------------------------------------
# Step 2 — Upload voice reference (run once)
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    volumes={VOLUME_DIR: volume},
    timeout=120,
)
def _store_reference_audio(audio_bytes: bytes) -> None:
    """Save raw WAV bytes to the Modal Volume as the fixed voice reference."""
    import os

    os.makedirs(VOLUME_DIR, exist_ok=True)
    with open(REF_AUDIO_PATH, "wb") as fh:
        fh.write(audio_bytes)
    volume.commit()
    print(f"Reference audio saved to {REF_AUDIO_PATH} ({len(audio_bytes):,} bytes).")


@app.local_entrypoint()
def main(ref_audio: str = "ref_audio.wav"):
    """Upload a local reference WAV to the Modal Volume.

    Usage:
        modal run examples/modal_tts_server.py --ref-audio /path/to/voice.wav
    """
    from pathlib import Path

    path = Path(ref_audio)
    if not path.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")
    audio_bytes = path.read_bytes()
    print(f"Uploading {path} ({len(audio_bytes):,} bytes) …")
    _store_reference_audio.remote(audio_bytes)
    print("Done. You can now deploy with: modal deploy examples/modal_tts_server.py")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "alloy"           # any voice name is accepted; only one voice is configured
    response_format: str = "wav"   # wav | pcm | mp3
    speed: float = 1.0             # accepted for API compatibility; speed adjustment is not yet implemented


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def _to_pcm16(pcm: np.ndarray) -> bytes:
    """Convert float32 numpy array → raw signed 16-bit little-endian PCM bytes."""
    return np.clip(pcm * 32768, -32768, 32767).astype(np.int16).tobytes()


def _wav_header(sample_rate: int, data_len: int = 0xFFFFFFFF) -> bytes:
    """Build a WAV header.  Pass data_len=0xFFFFFFFF for streaming (unknown size)."""
    n_ch = 1
    bits = 16
    byte_rate = sample_rate * n_ch * bits // 8
    blk_align = n_ch * bits // 8
    riff_size = 0xFFFFFFFF if data_len == 0xFFFFFFFF else 36 + data_len
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", riff_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_ch, sample_rate, byte_rate, blk_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    return buf.getvalue()


def _to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    raw = _to_pcm16(pcm)
    return _wav_header(sample_rate, len(raw)) + raw


def _to_mp3_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    try:
        from pydub import AudioSegment
    except ImportError:
        raise HTTPException(
            status_code=400,
            detail="response_format='mp3' requires pydub and ffmpeg in the image.",
        )
    segment = AudioSegment(_to_pcm16(pcm), frame_rate=sample_rate, sample_width=2, channels=1)
    buf = io.BytesIO()
    segment.export(buf, format="mp3")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# TTS Service
# ---------------------------------------------------------------------------


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    volumes={VOLUME_DIR: volume},
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=120,
)
class TTSService:
    """Serverless TTS class — one container = one GPU.

    @modal.enter runs once per container startup:
      1. Loads the Qwen3-TTS model from the Volume (no internet needed)
      2. Warms up CUDA graphs and caches the reference audio ICL prompt
         so the very first real request is as fast as every subsequent one.

    ICL mode (xvec_only=False) is used so the full reference audio codec
    tokens are included in the model context.  This is essential for languages
    that are not in the model's native training set (e.g. Thai): the acoustic
    tokens carry the phonetic and prosodic fingerprint of the voice, allowing
    the model to generalise beyond its training languages.

    Language is set to "Auto" — the model infers it from the input text, which
    avoids any mismatch when the reference speaker and the target text use
    different languages.

    For even faster cold starts, enable Modal memory snapshots:
      https://modal.com/docs/guide/memory-snapshot
    """

    @modal.enter()
    def load(self) -> None:
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        print(f"Loading {MODEL_NAME} from {MODEL_PATH} …")
        self.tts = FasterQwen3TTS.from_pretrained(
            MODEL_PATH,
            device="cuda",
            dtype=torch.bfloat16,
        )
        self.sample_rate = self.tts.sample_rate

        # Warm up: tokenises the reference audio ICL prompt + captures CUDA graphs.
        # xvec_only=False — use the full reference audio codec tokens as the
        # acoustic prompt (ICL mode).  This is required for languages outside the
        # model's native training set (e.g. Thai) because the audio tokens carry
        # the phonetic/prosodic context that the speaker embedding alone cannot.
        # language="Auto" — let the model infer the language from the input text.
        # 24 tokens ≈ 2 s of audio — short enough to keep startup fast while
        # still being long enough to trigger CUDA graph capture for the decode loop.
        print("Warming up CUDA graphs …")
        self.tts.generate_voice_clone(
            text="Warming up.",
            language="Auto",
            ref_audio=REF_AUDIO_PATH,
            ref_text="",
            xvec_only=False,   # ICL mode: reference audio tokens in context
            max_new_tokens=24,   # ~2 s of audio — just enough to capture CUDA graphs
        )

        # Serialise GPU inference — one request at a time per container.
        self._lock = threading.Lock()
        print("Ready.")

    # ------------------------------------------------------------------
    # ASGI endpoint — returns the FastAPI app to Modal's web layer
    # ------------------------------------------------------------------

    @modal.asgi_app()
    def web(self) -> FastAPI:
        web_app = FastAPI(title="faster-qwen3-tts — OpenAI-compatible API")

        tts = self.tts
        lock = self._lock
        sample_rate = self.sample_rate

        # Voice config — fixed single voice, language inferred automatically
        voice_cfg = {
            "ref_audio": REF_AUDIO_PATH,
            "ref_text": "",
            "language": "Auto",   # inferred from input text; works for Thai etc.
        }

        _CONTENT_TYPES = {
            "wav": "audio/wav",
            "pcm": "audio/pcm",
            "mp3": "audio/mpeg",
        }

        # ---- helpers ----

        async def _stream_chunks(text: str) -> AsyncGenerator[bytes, None]:
            """Run generate_voice_clone_streaming in a background thread, yield PCM chunks."""
            q: queue.Queue = queue.Queue()
            _DONE = object()

            def producer() -> None:
                try:
                    with lock:
                        for chunk, _sr, _timing in tts.generate_voice_clone_streaming(
                            text=text,
                            language=voice_cfg["language"],
                            ref_audio=voice_cfg["ref_audio"],
                            ref_text=voice_cfg["ref_text"],
                            xvec_only=False,   # ICL mode: reference audio tokens in context
                            # chunk_size=12 → one chunk per codec step at 12 Hz ≈ 1 s of audio.
                            # Lower values reduce first-chunk latency; higher values reduce overhead.
                            chunk_size=12,
                            non_streaming_mode=False,
                        ):
                            q.put(chunk)
                except Exception as exc:
                    q.put(exc)
                finally:
                    q.put(_DONE)

            thread = threading.Thread(target=producer, daemon=True)
            thread.start()

            loop = asyncio.get_event_loop()
            while True:
                item = await loop.run_in_executor(None, q.get)
                if item is _DONE:
                    break
                if isinstance(item, Exception):
                    raise item
                yield _to_pcm16(item)

        # ---- routes ----

        @web_app.get("/health")
        async def health():
            return {"status": "ok", "model": MODEL_NAME, "language": "Auto"}

        @web_app.post("/v1/audio/speech")
        async def create_speech(req: SpeechRequest):
            if not req.input.strip():
                raise HTTPException(status_code=400, detail="'input' text is empty.")
            if req.speed != 1.0:
                # Speed adjustment is not yet implemented; log and continue.
                import logging
                logging.getLogger(__name__).warning(
                    "speed=%.2f requested but speed adjustment is not yet implemented; "
                    "generating at normal speed.",
                    req.speed,
                )

            fmt = req.response_format.lower()
            if fmt not in _CONTENT_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported response_format {fmt!r}. Use: wav, pcm, mp3",
                )
            content_type = _CONTENT_TYPES[fmt]

            # MP3 — generate all audio then encode
            if fmt == "mp3":
                loop = asyncio.get_event_loop()

                def _generate():
                    with lock:
                        return tts.generate_voice_clone(
                            text=req.input,
                            language=voice_cfg["language"],
                            ref_audio=voice_cfg["ref_audio"],
                            ref_text=voice_cfg["ref_text"],
                            xvec_only=False,   # ICL mode: reference audio tokens in context
                        )

                audio_arrays, sr = await loop.run_in_executor(None, _generate)
                if not audio_arrays:
                    raise HTTPException(status_code=500, detail="TTS generation returned no audio.")
                return Response(content=_to_mp3_bytes(audio_arrays[0], sr), media_type=content_type)

            # WAV / PCM — stream chunks as they arrive
            async def audio_stream():
                if fmt == "wav":
                    yield _wav_header(sample_rate)
                async for raw_chunk in _stream_chunks(req.input):
                    yield raw_chunk

            return StreamingResponse(audio_stream(), media_type=content_type)

        return web_app
