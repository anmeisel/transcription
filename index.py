import whisper
import torch
from datetime import datetime

AUDIO_FILE = "audio/batool.wav"

# Check for GPU availability - but force Whisper to use CPU due to MPS compatibility issues
if torch.cuda.is_available():
    whisper_device = "cuda"
    print(f"Using device: CUDA")
elif torch.backends.mps.is_available():
    whisper_device = "cpu"  # Force Whisper to use CPU due to MPS compatibility issues
    print(f"Using device: CPU for Whisper (MPS incompatible)")
else:
    whisper_device = "cpu"
    print(f"Using device: CPU")

print("\nLoading Whisper model...")
model = whisper.load_model("base", device=whisper_device)

print("Transcribing with Whisper (this may take 5-15 minutes)...")
transcription_start = datetime.now()
print(f"Transcription started at: {transcription_start.strftime('%Y-%m-%d %H:%M:%S')}")

# Transcribe without verbose output to avoid printing segments to console
whisper_result = model.transcribe(AUDIO_FILE, word_timestamps=True, verbose=False)

transcription_end = datetime.now()
transcription_duration = transcription_end - transcription_start
print(f"Transcription complete at: {transcription_end.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Transcription took: {transcription_duration}")

print("\nSaving transcript to file...")
dialog = []

for seg in whisper_result["segments"]:
    start, end, text = seg["start"], seg["end"], seg["text"].strip()
    
    if not text:  # Skip empty segments
        continue
        
    # Format timestamp
    start_min = int(start // 60)
    start_sec = int(start % 60)
    end_min = int(end // 60)
    end_sec = int(end % 60)
    timestamp = f"[{start_min:02d}:{start_sec:02d} --> {end_min:02d}:{end_sec:02d}]"
    
    dialog.append((timestamp, text))

# Save output
with open("text/batool.txt", "w", encoding="utf-8") as f:
    f.write("Transcript\n")
    f.write(f"Audio file: {AUDIO_FILE}\n")
    f.write(f"Detected language: {whisper_result.get('language', 'Unknown')}\n")
    f.write(f"Transcription completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    for timestamp, text in dialog:
        f.write(f"{timestamp} {text}\n")

save_time = datetime.now()
print(f"Transcript saved to output.txt at: {save_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Process completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total segments transcribed: {len(dialog)}")