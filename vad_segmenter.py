# vad_segmenter.py
"""
Captura audio del micrófono, aplica WebRTC VAD y genera segmentos de voz
como arrays numpy (mono, 16 kHz, float32) listos para pasar a un modelo local.
Versión simple (sin timestamps adicionales).
"""

import collections
import queue
import sys
import threading
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import webrtcvad
from scipy import signal  # para resample si hace falta


@dataclass
class Frame:
    bytes: bytes
    timestamp: float
    duration: float


class VADAudio:
    def __init__(
        self,
        aggressiveness: int = 3,
        input_sample_rate: int = 48000,
        vad_sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        padding_duration_ms: int = 300,
        device=None,
    ):
        assert frame_duration_ms in (10, 20, 30)
        self.aggressiveness = aggressiveness
        self.input_sample_rate = input_sample_rate
        self.vad_sample_rate = vad_sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_bytes = int(vad_sample_rate * (frame_duration_ms / 1000.0) * 2)
        self.padding_duration_ms = padding_duration_ms
        self.device = device

        self._buff = queue.Queue()
        self._running = False
        self.vad = webrtcvad.Vad(self.aggressiveness)

    def _resample_to_16k(self, data: np.ndarray, src_sr: int) -> np.ndarray:
        if src_sr == self.vad_sample_rate:
            return data
        if data.dtype == np.int16:
            data_float = data.astype(np.float32) / 32768.0
        else:
            data_float = data.astype(np.float32)
        gcd = np.gcd(src_sr, self.vad_sample_rate)
        up = self.vad_sample_rate // gcd
        down = src_sr // gcd
        resampled = signal.resample_poly(data_float, up, down)
        resampled_int16 = np.clip(resampled * 32768, -32768, 32767).astype(np.int16)
        return resampled_int16

    def _bytes_to_float32(self, b: bytes) -> np.ndarray:
        int16 = np.frombuffer(b, dtype=np.int16)
        return (int16.astype(np.float32) / 32768.0).copy()

    def audio_callback(self, indata, frames, time_info, status):
        try:
            if status:
                print("Status:", status, file=sys.stderr)
            if indata.ndim > 1:
                mono = np.mean(indata, axis=1)
            else:
                mono = indata
            int16 = np.clip(mono * 32768, -32768, 32767).astype(np.int16)
            if int(sd.default.samplerate) != self.vad_sample_rate:
                int16 = self._resample_to_16k(int16, src_sr=int(sd.default.samplerate))
            self._buff.put(int16.tobytes())
        except Exception as e:
            print("ERROR audio_callback:", e, file=sys.stderr)

    def start_stream(self, samplerate=48000, channels=1, blocksize=0):
        sd.default.samplerate = samplerate
        sd.default.channels = channels
        self._running = True
        self._stream = sd.InputStream(
            callback=self.audio_callback,
            channels=channels,
            samplerate=samplerate,
            dtype="float32",
            device=self.device,
            blocksize=blocksize,
        )
        self._stream.start()

    def stop_stream(self):
        self._running = False
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
        while not self._buff.empty():
            try:
                self._buff.get_nowait()
            except queue.Empty:
                break

    def frames(self):
        bytes_per_frame = int(self.vad_sample_rate * (self.frame_duration_ms / 1000.0) * 2)
        buffer = b""
        timestamp = 0.0
        frame_duration = self.frame_duration_ms / 1000.0
        while self._running:
            try:
                data = self._buff.get(timeout=1.0)
            except queue.Empty:
                continue
            buffer += data
            while len(buffer) >= bytes_per_frame:
                frame_bytes = buffer[:bytes_per_frame]
                buffer = buffer[bytes_per_frame:]
                yield Frame(frame_bytes, timestamp, frame_duration)
                timestamp += frame_duration
        while len(buffer) >= bytes_per_frame:
            frame_bytes = buffer[:bytes_per_frame]
            buffer = buffer[bytes_per_frame:]
            yield Frame(frame_bytes, timestamp, frame_duration)
            timestamp += frame_duration

    # def vad_collector(self):
    #     """
    #     VSimplified collector: agrupa frames en segmentos y devuelve numpy float32 mono @16k.
    #     """
    #     num_padding_frames = int(self.padding_duration_ms / self.frame_duration_ms)
    #     ring_buffer = collections.deque(maxlen=num_padding_frames)

    #     triggered = False
    #     voiced_frames = []
        
    #     segment_start_time = None
    #     segment_end_time = None
        
    #     for frame in self.frames():
    #         is_speech = self.vad.is_speech(frame.bytes, sample_rate=self.vad_sample_rate)

    #         if not triggered:
    #             ring_buffer.append((frame, is_speech))
    #             num_voiced = len([f for f, speech in ring_buffer if speech])
    #             if num_voiced > 0.9 * ring_buffer.maxlen:
    #                 triggered = True
    #                 for f, s in ring_buffer:
    #                     voiced_frames.append(f)
    #                 ring_buffer.clear()
    #         else:
    #             voiced_frames.append(frame)
    #             ring_buffer.append((frame, is_speech))
    #             num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                
    #             # calcular duración actual del segmento (aprox)
    #             seg_frames = len(ring_buffer)
    #             seg_ms = seg_frames * self.frame_duration_ms
                
    #             if num_unvoiced > 0.9 * ring_buffer.maxlen:
    #                 segment_bytes = b"".join([f.bytes for f in voiced_frames])
    #                 audio_np = self._bytes_to_float32(segment_bytes)
    #                 yield audio_np, segment_start_time, segment_end_time
    #                 triggered = False
    #                 voiced_frames = []
    #                 ring_buffer.clear()

    #     if voiced_frames:
    #         segment_bytes = b"".join([f.bytes for f in voiced_frames])
    #         audio_np = self._bytes_to_float32(segment_bytes)
    #         yield audio_np, segment_start_time, segment_end_time
    def vad_collector(
        self,
        start_ratio: float = 0.6,
        end_ratio: float = 0.9,
        min_segment_ms: int = 300,
        max_segment_ms: int = 15_000,
        merge_short_with_next: bool = True,
        return_timestamps: bool = True,
    ):
        """
        Generador mejorado que agrupa frames en segmentos de voz con control fino.
        Devuelve tuplas (audio_np) o (audio_np, start_time, end_time) si return_timestamps=True.

        Parámetros:
        - start_ratio: fracción de frames voiced necesaria en el padding para iniciar.
        - end_ratio: fracción de frames unvoiced necesaria en el padding para terminar.
        - min_segment_ms: si el segmento final es menor, se puede ignorar o fusionar según merge_short_with_next.
        - max_segment_ms: limita la duración máxima de un segmento.
        - merge_short_with_next: si True, segmentos cortos se intentan unir con el siguiente (si existe).
        """
        num_padding_frames = int(self.padding_duration_ms / self.frame_duration_ms)
        ring_buffer = collections.deque(maxlen=num_padding_frames)

        triggered = False
        # ahora almacenamos tuples (frame, is_speech) para permitir trimming posterior
        buffered_voiced = []  # lista de (frame, is_speech)
        last_yielded = None  # para intentar mergear con el anterior si es corto

        for frame in self.frames():
            is_speech = self.vad.is_speech(frame.bytes, sample_rate=self.vad_sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                # contar voiced en la ventana
                num_voiced = sum(1 for f, s in ring_buffer if s)
                if num_voiced >= int(start_ratio * ring_buffer.maxlen):
                    # iniciar segmento: añadir todos los frames del ring_buffer (manteniendo is_speech)
                    triggered = True
                    buffered_voiced = list(ring_buffer)
                    ring_buffer.clear()
            else:
                # estamos en un segmento activo
                buffered_voiced.append((frame, is_speech))
                ring_buffer.append((frame, is_speech))
                num_unvoiced = sum(1 for f, s in ring_buffer if not s)

                # calcular duración actual del segmento (aprox)
                seg_frames = len(buffered_voiced)
                seg_ms = seg_frames * self.frame_duration_ms

                # Condición de fin: demasiados no-voz en la ventana o superamos max_segment_ms
                if (num_unvoiced >= int(end_ratio * ring_buffer.maxlen)) or (seg_ms >= max_segment_ms):
                    # Construir bytes y convertir a numpy
                    # Antes de eso, recortamos silencios al inicio/fin
                    # buffered_voiced es lista de (frame, is_speech)
                    # recortar inicio
                    start_idx = 0
                    while start_idx < len(buffered_voiced) and not buffered_voiced[start_idx][1]:
                        start_idx += 1
                    end_idx = len(buffered_voiced) - 1
                    while end_idx >= 0 and not buffered_voiced[end_idx][1]:
                        end_idx -= 1

                    if start_idx > end_idx:
                        # todo silencio -> ignorar
                        segment_bytes = b""
                        segment_start_time = None
                        segment_end_time = None
                    else:
                        selected = buffered_voiced[start_idx:end_idx + 1]
                        segment_bytes = b"".join([f.bytes for f, s in selected])
                        segment_start_time = selected[0][0].timestamp
                        last_frame = selected[-1][0]
                        segment_end_time = last_frame.timestamp + last_frame.duration

                    # convertir a numpy float32
                    if segment_bytes:
                        audio_np = self._bytes_to_float32(segment_bytes)
                    else:
                        audio_np = np.array([], dtype=np.float32)

                    # decidir si emitir o mergear
                    if audio_np.size > 0:
                        # si segmento muy corto
                        if seg_ms < min_segment_ms and merge_short_with_next:
                            # guardar en last_yielded para intentar fusionar con el siguiente
                            if last_yielded is None:
                                last_yielded = (audio_np, segment_start_time, segment_end_time)
                            else:
                                # fusionar con el anterior guardado
                                prev_np, prev_start, prev_end = last_yielded
                                merged = np.concatenate([prev_np, audio_np])
                                merged_start = prev_start
                                merged_end = segment_end_time
                                # emitir merged
                                if return_timestamps:
                                    yield merged, merged_start, merged_end
                                else:
                                    yield merged
                                last_yielded = None
                        else:
                            # si hay un last_yielded pendiente, emitirlo antes de este
                            if last_yielded is not None:
                                p_np, p_start, p_end = last_yielded
                                if return_timestamps:
                                    yield p_np, p_start, p_end
                                else:
                                    yield p_np
                                last_yielded = None
                            # emitir el segmento actual
                            if return_timestamps:
                                yield audio_np, segment_start_time, segment_end_time
                            else:
                                yield audio_np

                    # reset y continuar
                    triggered = False
                    buffered_voiced = []
                    ring_buffer.clear()

        # fin del stream: si queda buffered_voiced, emitir (similar al bloque anterior)
        if buffered_voiced:
            # recortar como arriba
            start_idx = 0
            while start_idx < len(buffered_voiced) and not buffered_voiced[start_idx][1]:
                start_idx += 1
            end_idx = len(buffered_voiced) - 1
            while end_idx >= 0 and not buffered_voiced[end_idx][1]:
                end_idx -= 1

            if start_idx <= end_idx:
                selected = buffered_voiced[start_idx:end_idx + 1]
                segment_bytes = b"".join([f.bytes for f, s in selected])
                audio_np = self._bytes_to_float32(segment_bytes)
                segment_start_time = selected[0][0].timestamp
                last_frame = selected[-1][0]
                segment_end_time = last_frame.timestamp + last_frame.duration

                if last_yielded is not None:
                    # fusionar último pendiente
                    p_np, p_start, p_end = last_yielded
                    merged = np.concatenate([p_np, audio_np])
                    merged_start = p_start
                    merged_end = segment_end_time
                    if return_timestamps:
                        yield merged, merged_start, merged_end
                    else:
                        yield merged
                    last_yielded = None
                else:
                    if return_timestamps:
                        yield audio_np, segment_start_time, segment_end_time
                    else:
                        yield audio_np
