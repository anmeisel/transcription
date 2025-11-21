# Audio Transcription

Transcribe audio files using **OpenAI's Whisper**. Output is saved to `output.txt` with timestamps.

---

## Installation

```bash
pip install openai-whisper
pip install torch torchaudio
brew install ffmpeg
```

## Usage

1. Convert audio to WAV (16kHz, mono):
   `ffmpeg -i audio/[audio-name].m4a -ar 16000 -ac 1 audio/[audio-name].wav`
2. In `index.py`, replace [your-audio-name] with your desired audio filename:

- line 5: `AUDIO_FILE = "audio/[your-audio-name].wav"`
- line 52: `with open("text/[your-audio-name].txt"`

3. Run transcription: `python3 index.py`

- Console: Shows progress and completion timestamps only
- Output: text/[audio-name].txt
