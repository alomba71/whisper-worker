import os
import sys
import time

def main() -> int:
    # Healthcheck rápido: valida GPU + CUDA stack con un modelo pequeño.
    # No descargamos large-v3 aquí para no depender de red ni perder minutos.
    model_name = os.environ.get("HC_MODEL", "tiny")  # tiny = rápido
    compute_type = os.environ.get("WHISPER_COMPUTE", "float16")
    device = os.environ.get("HC_DEVICE", "cuda")  # queremos GPU sí o sí

    t0 = time.time()
    try:
        from faster_whisper import WhisperModel
        m = WhisperModel(model_name, device=device, compute_type=compute_type, download_root=os.environ.get("MODELS_DIR", "/models"))
        # Fuerza una operación ligera (cargar backend). No transcribe nada.
        _ = m.model  # acceso interno para forzar init
    except Exception as e:
        print(f"[HC] FAIL: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    dt = time.time() - t0
    print(f"[HC] OK: device={device} compute={compute_type} model={model_name} t={dt:.2f}s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
