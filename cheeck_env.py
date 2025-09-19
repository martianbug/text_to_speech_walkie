# check_env.py
import sys, traceback
print("Python:", sys.version)
print("--- Intentando importar módulos ---")
mods = ["sounddevice", "webrtcvad", "numpy", "scipy", "soundfile", "faster_whisper", "tkinter"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"{m}: OK")
    except Exception as e:
        print(f"{m}: ERROR -> {e.__class__.__name__}: {e}")
print("\n--- sounddevice: listar dispositivos ---")
try:
    import sounddevice as sd
    try:
        print("Default samplerate:", sd.default.samplerate)
    except Exception as e:
        print("No default samplerate:", e)
    try:
        print(sd.query_devices())
    except Exception as e:
        print("query_devices error:", e)
except Exception as e:
    print("sounddevice import failed:", e)
print("\n--- Probar carga rápida de faster-whisper (small, en hilo para timeout) ---")
try:
    import threading, time
    from faster_whisper import WhisperModel
    ok = {"loaded": False, "error": None}
    def try_load():
        try:
            WhisperModel("tiny", device="cpu")
            ok["loaded"] = True
        except Exception as e:
            ok["error"] = str(e)

    t = threading.Thread(target=try_load, daemon=True)
    t.start()
    t.join(timeout=30)
    if t.is_alive():
        print("Carga del modelo: TIMEOUT (30s). Puede estar descargando o colgado.")
    else:
        if ok["loaded"]:
            print("Carga del modelo: OK (tiny)")
        else:
            print("Carga del modelo: ERROR ->", ok["error"])
except Exception as e:
    print("Error al intentar cargar modelo de prueba:", e)
print("\n--- FIN check_env ---")
