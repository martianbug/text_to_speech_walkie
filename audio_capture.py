import time
from vad_segmenter import VADAudio
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu")

def main():
    # Ajusta sample rate de entrada si tu tarjeta usa 44100 o 48000
    input_sr = 48000
    vad = VADAudio(
        aggressiveness=3,
        input_sample_rate=input_sr,
        vad_sample_rate=16000,
        frame_duration_ms=30,
        padding_duration_ms=300,
        device=None, #select device
    )

    import numpy as np
    print("Iniciando captura. Habla para generar segmentos (Ctrl+C para parar).")
    try:
        vad.start_stream(samplerate=input_sr, channels=1)
        # Iterar sobre segmentos de voz detectados
        for i, segment in enumerate(vad.vad_collector()):
            # segment es numpy float32 mono @ 16kHz con valores en [-1,1]
            audio_np = np.asarray(segment)
            if audio_np.dtype == np.int16:
                audio_np = audio_np.astype(np.float32) / 32768.0
            else:
                audio_np = audio_np.astype(np.float32)
            if audio_np.ndim > 1:
                audio_np = audio_np.reshape(-1)
            duration_s = audio_np.shape[0] / 16000.0
            print(f"Transcribiendo segmento {i}: dur={duration_s:.2f}s, shape={audio_np.shape}, dtype={audio_np.dtype}")
            segments, info = model.transcribe(audio_np, language="en", beam_size=5)
            text = ""
            for r in segments:
                # cada r puede contener .text u otros campos dependiendo de versión
                if hasattr(r, "text"):
                    text += r.text
                elif isinstance(r, dict) and "text" in r:
                    text += r["text"]
            print(f"Transcripción: {text}")
            
    except KeyboardInterrupt:
        print("Interrumpido por usuario")
    finally:
        vad.stop_stream()
        print("Detenido.")

if __name__ == "__main__":
    main()
