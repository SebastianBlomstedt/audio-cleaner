import os
from tqdm import tqdm
import whisperx
import torch
from dotenv import load_dotenv

# === SYSTEM SETUP ===
# Lägg till ffmpeg i PATH om det inte är globalt installerat
os.environ["PATH"] += os.pathsep + r"C:\Users\SebastianBlomstedt\ffmpeg\bin"

# Ladda token från .env
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

if not HUGGINGFACE_TOKEN:
    print("❌ Missing HF_TOKEN in .env file.")
    exit(1)

# === INPUT ===
audio_file = input("Enter path to your cleaned WAV file: ").strip()

if not os.path.isfile(audio_file):
    print("❌ File not found. Please check the path.")
    exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float32" if device == "cpu" else "float16"
print(f"✅ Using device: {device}")

# === TRANSCRIBE ===
print("🔁 Transcribing...")
model = whisperx.load_model("large-v3", device=device, compute_type=compute_type)
transcription = model.transcribe(audio_file)

# === ALIGN ===
print("🔁 Aligning words (svenska)...")
model_a, metadata = whisperx.load_align_model(
    align_model="KBLab/wav2vec2-large-voxrex-swedish", device=device
)
aligned_result = whisperx.align(transcription["segments"], model_a, metadata, audio_file, device)

# === DIARIZE ===
print("🔁 Performing speaker diarization...")
diarize_pipeline = whisperx.DiarizationPipeline(use_auth_token=HUGGINGFACE_TOKEN, device=device)
diarize_segments = diarize_pipeline(audio_file)

# === ASSIGN SPEAKERS WITH PROGRESS ===
print("🔁 Assigning speakers...")
final = {"segments": []}
word_segments = aligned_result["word_segments"]

for segment in tqdm(diarize_segments, desc="Assigning speakers"):
    result = whisperx.assign_word_speakers([segment], word_segments)
    final["segments"].extend(result["segments"])

# === OUTPUT ===
output_file = os.path.splitext(audio_file)[0] + "_transcribed.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for segment in final["segments"]:
        f.write(f"{segment['speaker']}: {segment['text']}\n")

print(f"✅ Transcription saved to: {output_file}")
