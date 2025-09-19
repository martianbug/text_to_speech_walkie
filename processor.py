# Esquema: captura -> VAD -> enviar a faster-whisper

from faster_whisper import WhisperModel
import webrtcvad
import sounddevice as sd
import numpy as np
import queue, threading

model = WhisperModel("small", device="cpu")

# Captura audio y usa VAD para segmentar. Cuando un segmento termina, llamas:
segments, info = model.transcribe(audio_segment_array, language="es", beam_size=5)
# segments contiene texto y timestamps -> lo muestras en UI.
