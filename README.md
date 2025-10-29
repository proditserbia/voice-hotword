# 🗣️ Voice Trigger v1  
**Offline hotword detection + command execution (Faster-Whisper backend)**  
Low-latency, fully local speech command system for Raspberry Pi / RK3566 / x86.

---

## 🚀 Overview
This script continuously listens for short speech segments, transcribes them using [faster-whisper](https://github.com/guillaumekln/faster-whisper), and performs fuzzy matching against predefined hotwords to execute system commands — all offline.

Optimized for:
- **ARM boards** (Raspberry Pi 4/5, RK3566/3588)
- **x86 CPUs**
- Low latency (~0.5–0.6s on x86, 3–6s on ARM)

---

## 💾 Installation

### 1️⃣ System dependencies
```bash
sudo apt update
sudo apt install -y python3 python3-pip portaudio19-dev ffmpeg git
```

### 2️⃣ Clone repository
```bash
git clone https://github.com/proditserbia/voice-hotword.git
cd voice-hotword
```

If you only use `voice_trigger_v1.py`, copy it to your preferred folder (e.g. `/home/pi/voice/`).

### 3️⃣ Python dependencies
```bash
pip3 install --upgrade pip wheel setuptools
pip3 install sounddevice webrtcvad numpy rapidfuzz faster-whisper
```

---

## 🧠 Model setup

Download a small **faster-whisper** model (`tiny.en` or `tiny`):

```bash
mkdir -p ~/.cache/whisper
```

The first run will automatically download the model.  
Manual download:  
👉 [https://huggingface.co/guillaumekln/faster-whisper-tiny.en](https://huggingface.co/guillaumekln/faster-whisper-tiny.en)

---

## ▶️ Running

Example (German language):

```bash
python3 voice_trigger_v1.py --lang de
```

Default settings:
- Model: `tiny.en`
- Compute type: `int8`
- Device: `cpu`
- 1–3 CPU threads (optimized for ARM)
- Low-latency VAD (WebRTC)

Console output:
```
🎙️  Listening... Press Ctrl+C to stop.
📝  say hello  (took 0.56s)
✅ Trigger matched: 'say hello' (score 100.0) -> running command
```

---

## ⚙️ Configuration

Edit the top of `voice_trigger_v1.py`:

```python
TRIGGERS = {
    "open browser": "xdg-open https://www.wikipedia.org",
    "turn off screen": "bash -lc 'xset dpms force off'",
    "say hello": "bash -lc 'notify-send \"Trigger activated\"'"
}
```

- Add or edit phrases and their shell commands.  
- Change language with `--lang` (`en`, `de`, `sr`, ...).  
- Adjust model with `--model` (`tiny`, `base`, etc.).  

---

## ⚡ Performance

| Platform | Model | Compute | Latency |
|-----------|--------|----------|----------|
| Linux Mint (i5) | tiny.en | int8 | ~0.5–0.6s |
| Raspberry Pi 4 / RK3566 | tiny | int8 | ~3–6s |

Use `int8` for fastest inference on CPU.

---

## 🔄 Optional: Autostart on boot

```bash
crontab -e
```

Add:
```
@reboot /usr/bin/python3 /home/pi/voice/voice_trigger_v1.py --lang de >> /home/pi/voice/log.txt 2>&1
```

---

## 🧩 Credits
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [WebRTC VAD](https://webrtc.org/)
- [rapidfuzz](https://github.com/maxbachmann/RapidFuzz)
- Developed by **prodit.rs**

---

## 📜 License
MIT License © 2025 prodit.rs
