#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline voice trigger demo:
- Listens to mic continuously
- Uses WebRTC VAD to detect speech segments
- Transcribes segments with faster-whisper
- Fuzzy-matches phrases and runs Linux commands on match

Tested on: Linux Mint (PulseAudio/ALSA)
"""

import argparse
import collections
import math
import queue
import signal
import subprocess
import sys
import time
import unicodedata
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import webrtcvad
from rapidfuzz import fuzz, process
from faster_whisper import WhisperModel

# -----------------------------
# CONFIG (edit to your needs)
# -----------------------------
TRIGGERS = {
    # phrase            -> command to run (string or list)
    "open browser": "xdg-open https://www.wikipedia.org",
    "turn off screen": "bash -lc 'xset dpms force off'",
    "say hello": "bash -lc 'notify-send \"Trigger activated\" \"Hello from voice trigger\"'"
}

LANGUAGE = "en"          # "en" or leave None for auto
MODEL_SIZE = "base.en"   # good options: "tiny", "tiny.en", "base", "base.en"
COMPUTE_TYPE = "int8"    # "int8" (CPU-friendly), "int8_float16", "float32"

# VAD & audio
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_MS = 30            # must be 10, 20 or 30 for webrtcvad
VAD_AGGRESSIVENESS = 3   # 0-3 (3 = most aggressive)
MAX_SEGMENT_SEC = 5   # safety cap per utterance
MIN_SPEECH_SEC = 0.4     # minimum speech before transcribing
SPEECH_PAD_MS = 150      # padding around speech

# Matching
FUZZY_THRESHOLD = 85     # 0-100 (higher = stricter)
COOLDOWN_SEC = 5.0       # avoid re-triggering the same phrase immediately

# Logging
DEBUG = True

# -----------------------------
# Helpers
# -----------------------------
def debug(*args):
    if DEBUG:
        print(*args, file=sys.stderr)

def normalize_text(s: str) -> str:
    # lower, remove accents, trim punctuation
    s = s.lower().strip()
    s = "".join(
        c for c in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(c)
    )
    # very light punctuation strip
    for ch in ",.;:!?\"'()[]{}":
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s

def pcm16_to_float32(pcm16: np.ndarray) -> np.ndarray:
    return (pcm16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)

def run_command(cmd):
    try:
        if isinstance(cmd, str):
            # shell for convenience in demo; for production prefer list without shell
            subprocess.Popen(cmd, shell=True)
        else:
            subprocess.Popen(cmd)
    except Exception as e:
        debug("Command failed:", e)

# -----------------------------
# Audio/VAD streaming
# -----------------------------
class SpeechSegmenter:
    def __init__(self, vad_level=VAD_AGGRESSIVENESS, frame_ms=FRAME_MS, sample_rate=SAMPLE_RATE):
        self.vad = webrtcvad.Vad(vad_level)
        self.sample_rate = sample_rate
        self.frame_bytes = int(sample_rate * (frame_ms/1000.0)) * 2  # 16-bit mono
        self.frame_len = int(sample_rate * (frame_ms/1000.0))        # samples
        self.pad_frames = int(SPEECH_PAD_MS / frame_ms)
        self.reset()

    def reset(self):
        self.speech_frames = []
        self.trailing_non_speech = 0
        self.in_speech = False
        self.start_time = None

    def process(self, chunk_bytes):
        """
        Feed exact FRAME_MS-sized 16-bit PCM mono frames.
        Returns (finished_segment_bytes or None, speech_active: bool)
        """
        assert len(chunk_bytes) == self.frame_bytes
        is_speech = self.vad.is_speech(chunk_bytes, self.sample_rate)

        if is_speech:
            if not self.in_speech:
                # start of speech: include left padding (handled by caller via ring buffer)
                self.in_speech = True
                self.start_time = time.time()
            self.speech_frames.append(chunk_bytes)
            self.trailing_non_speech = 0
        else:
            if self.in_speech:
                self.trailing_non_speech += 1
                self.speech_frames.append(chunk_bytes)
                # end speech if enough trailing non-speech collected
                if self.trailing_non_speech >= self.pad_frames:
                    seg = b"".join(self.speech_frames)
                    self.reset()
                    return seg, False
            # else idle

        # safety cap
        if self.in_speech and self.start_time and (time.time() - self.start_time) > MAX_SEGMENT_SEC:
            seg = b"".join(self.speech_frames)
            self.reset()
            return seg, False

        return None, self.in_speech

# -----------------------------
# Matcher
# -----------------------------
from rapidfuzz import fuzz, process

class PhraseMatcher:
    def __init__(self, triggers: dict, threshold=85, require_all_tokens=True,
                 min_chars=3):
        """
        - threshold: fuzzy prag (0-100)
        - require_all_tokens: za fraze sa >1 reči traži da SVE reči budu prisutne u transkriptu
        - min_chars: ignoriše prekratke transkripte (npr. "ok", "no", "a")
        """
        self.threshold = threshold
        self.require_all_tokens = require_all_tokens
        self.min_chars = min_chars

        # Normalizovane fraze + tokeni
        self.triggers = {}
        self.tokens = {}
        for phrase, cmd in triggers.items():
            p = normalize_text(phrase)
            self.triggers[p] = cmd
            self.tokens[p] = set(p.split())

        self._cooldowns = {}  # phrase -> last_ts

    def match(self, text: str):
        txt = normalize_text(text)
        if not txt or len(txt) < self.min_chars:
            return None, None, 0

        txt_tokens = set(txt.split())

        # Filtriraj kandidate: ako fraza ima >1 reč, zahtevaj prisustvo svih reči
        candidates = []
        for phrase in self.triggers.keys():
            phrase_tokens = self.tokens[phrase]
            if self.require_all_tokens and len(phrase_tokens) > 1:
                if not phrase_tokens.issubset(txt_tokens):
                    continue
            candidates.append(phrase)

        if not candidates:
            return None, None, 0

        # Stroži skor bez „subset 100“ efekta
        match, score, _ = process.extractOne(
            txt, candidates, scorer=fuzz.ratio
        )

        if score >= self.threshold:
            last = self._cooldowns.get(match, 0)
            now = time.time()
            if (now - last) >= COOLDOWN_SEC:
                self._cooldowns[match] = now
                return match, self.triggers[match], score

        return None, None, score

# -----------------------------
# Transcriber
# -----------------------------
class Transcriber:
    def __init__(self, model_size=MODEL_SIZE, compute_type=COMPUTE_TYPE, language=LANGUAGE):
        # You can pass local model path instead of model_size if you want full offline without auto-download
        self.model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
        self.language = language

    def transcribe_pcm16(self, pcm16: np.ndarray):
        # convert to float32 [-1,1]
        audio = pcm16_to_float32(pcm16)
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            vad_filter=False,      # we already perform VAD
            beam_size=1,
            best_of=1
        )
        text = "".join([seg.text for seg in segments]).strip()
        return text

# -----------------------------
# Main loop
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Offline voice trigger (VAD + faster-whisper)")
    ap.add_argument("--device", type=str, default=None, help="sounddevice input (e.g. 'default')")
    ap.add_argument("--model", type=str, default=MODEL_SIZE, help="faster-whisper model size or path")
    ap.add_argument("--compute", type=str, default=COMPUTE_TYPE, help="compute type (int8, int8_float16, float32)")
    ap.add_argument("--lang", type=str, default=LANGUAGE, help="language hint (e.g., en, hr, sr; or leave None)")
    ap.add_argument("--threshold", type=int, default=FUZZY_THRESHOLD, help="fuzzy match threshold 0-100")
    args = ap.parse_args()

    matcher = PhraseMatcher(TRIGGERS, threshold=args.threshold)
    transcriber = Transcriber(model_size=args.model, compute_type=args.compute, language=args.lang)

    vad_segmenter = SpeechSegmenter()

    # ring buffer to provide left padding before speech
    pre_speech_frames = collections.deque(maxlen=vad_segmenter.pad_frames)

    # buffer logger
    bytes_per_sec = SAMPLE_RATE * 2  # 16-bit mono
    min_frames_before_transcribe = int((MIN_SPEECH_SEC * 1000) / FRAME_MS)

    print("🎙️  Listening... Press Ctrl+C to stop.", flush=True)

    # prepare SD stream
    frame_samples = vad_segmenter.frame_len
    audio_q = queue.Queue()

    def audio_cb(indata, frames, time_info, status):
        if status:
            debug("Audio status:", status)
        # ensure mono 16-bit PCM
        pcm16 = (indata[:, 0] * 32768.0).astype(np.int16).tobytes()
        audio_q.put(pcm16)

    stop_flag = threading.Event()

    def sigint_handler(sig, frame):
        stop_flag.set()
    signal.signal(signal.SIGINT, sigint_handler)

    with sd.InputStream(
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype="float32",
        blocksize=frame_samples,
        device=args.device,
        callback=audio_cb,
        latency="low"
    ):
        while not stop_flag.is_set():
            try:
                frame = audio_q.get(timeout=0.2)
            except queue.Empty:
                continue

            # store in pre-speech buffer
            pre_speech_frames.append(frame)

            # Feed VAD
            seg_bytes, in_speech = vad_segmenter.process(frame)

            if seg_bytes is None:
                continue

            # We ended a segment OR hit safety cap
            # Ensure it is long enough
            total_frames = len(seg_bytes) // (2 * vad_segmenter.frame_len)  # 2 bytes per sample
            if total_frames < min_frames_before_transcribe:
                debug("Segment too short, skipping.")
                continue

            # Attach left padding frames to segment
            left_pad = b"".join(list(pre_speech_frames))
            # Right padding already included by segmenter
            full_bytes = left_pad + seg_bytes

            # Convert bytes -> PCM16 np array
            pcm16 = np.frombuffer(full_bytes, dtype=np.int16)

            # Transcribe
            t0 = time.time()
            text = transcriber.transcribe_pcm16(pcm16)
            dt = time.time() - t0

            if text:
                print(f"📝  {text}  (took {dt:.2f}s)")
            else:
                debug("Empty transcription.")
                continue

            # Match and trigger
            phrase, command, score = matcher.match(text)
            if phrase and command:
                print(f"✅ Trigger matched: '{phrase}' (score {score}) -> running command")
                run_command(command)
            else:
                debug(f"No trigger matched (best score: {score}).")

    print("👋 Bye.")

if __name__ == "__main__":
    main()
