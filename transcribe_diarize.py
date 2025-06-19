
import torch
import os
os.environ["PATH"] += os.pathsep + r"C:\Users\SebastianBlomstedt\ffmpeg\bin"
import whisperx
from tqdm import tqdm

# === STEP 1: Ask user for input ===
audio_file = input("Enter path to your cleaned WAV file: ").strip()

if not os.path.isfile(audio_file):
    print("❌ File not found. Please check the path.")
    exit(1)

# === STEP 2: Setup Hugging Face Token ===
HUGGINGFACE_TOKEN = "your-huggingface-token-here"  # Replace this with your actual token

# === STEP 3: Determine device (GPU if available) ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# === STEP 4: Transcription (force float32 on CPU to avoid errors) ===
print("🔁 Transcribing...")
model = whisperx.load_model("large-v3", device=device, compute_type="float32" if device == "cpu" else "float16")
transcription = model.transcribe(audio_file)

# === STEP 5: Alignment ===
print("🔁 Aligning words...")
model_a, metadata = whisperx.load_align_model(language_code=transcription["language"], device=device)
aligned_result = whisperx.align(transcription["segments"], model_a, metadata, audio_file, device)

# === STEP 6: Diarization ===
print("🔁 Performing speaker diarization...")
diarize_model = whisperx.DiarizationPipeline(use_auth_token=HUGGINGFACE_TOKEN, device=device)
diarize_segments = diarize_model(audio_file)

# === STEP 7: Assign speaker labels with progress bar ===
print("🔁 Assigning speakers with progress...")

final = {"segments": []}
word_segments = aligned_result["word_segments"]

for segment in tqdm(diarize_segments, desc="Assigning speakers"):
    segment_words = whisperx.assign_word_speakers([segment], word_segments)
    final["segments"].extend(segment_words["segments"])

# === STEP 8: Save to file ===
output_file = os.path.splitext(audio_file)[0] + "_transcribed.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for segment in final["segments"]:
        f.write(f"{segment['speaker']}: {segment['text']}\n")

print(f"✅ Transcription with speakers saved to: {output_file}")
