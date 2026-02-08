import os
import sys
import time

def main() -> int:
    model_name = os.environ.get("WHISPER_MODEL", "large-v3")
    lang = os.environ.get("WHISPER_LANG", "es")
    compute_type = os.environ.get("WHISPER_COMPUTE", "float16")
    models_dir = os.environ.get("MODELS_DIR", "/models")

    # Warmup: descarga/cacha el modelo real que vas a usar.
    # Esto se hace UNA vez al arrancar la instancia Vast, antes de procesar lotes.
    t0 = time.time()
    try:
        from faster_whisper import WhisperModel
        m = WhisperModel(model_name, device="cuda", compute_type=compute_type, download_root=models_dir)

        # Transcripción mínima “en vacío” no es posible sin audio,
        # pero con crear el modelo ya fuerzas la descarga y carga del backend.
        # El lang se deja preparado como info (lo usarás en el worker al transcribir).
        print(f"[WARMUP] Model ready: {model_name} lang={lang} compute={compute_type} cache={models_dir}")
    except Exception as e:
        print(f"[WARMUP] FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    dt = time.time() - t0
    print(f"[WARMUP] OK in {dt:.2f}s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
