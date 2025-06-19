import os
import torch
from tqdm import tqdm
import whisperx
from dotenv import load_dotenv
from multiprocessing import Pool, cpu_count, freeze_support

# === CONFIG ===
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
FFMPEG_PATH = r"C:\Users\SebastianBlomstedt\ffmpeg\bin"

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "int8" if device == "cpu" else "float16"
NUM_WORKERS = min(4, cpu_count())  # Adjust up to 4 or number of cores

def process_file(audio_path):
    os.environ["PATH"] += os.pathsep + FFMPEG_PATH
    torch.set_num_threads(2)  # Prevent CPU overload per process

    base = os.path.splitext(audio_path)[0]

    try:
        # Load models (in each process)
        asr_model = whisperx.load_model("large-v3", device=device, compute_type=compute_type)
        align_model, metadata = whisperx.load_align_model(
            align_model="KBLab/wav2vec2-large-voxrex-swedish", device=device
        )
        diarization_pipeline = whisperx.DiarizationPipeline(
            use_auth_token=HUGGINGFACE_TOKEN, device=device
        )

        # Transcribe
        transcription = asr_model.transcribe(audio_path)

        # Align
        aligned = whisperx.align(
            transcription["segments"], align_model, metadata, audio_path, device
        )

        # Diarize
        diarize_segments = diarization_pipeline(audio_path)

        # Assign speakers
        final = {"segments": []}
        for segment in diarize_segments:
            result = whisperx.assign_word_speakers([segment], aligned["word_segments"])
            final["segments"].extend(result["segments"])

        # Save output
        output_file = base + "_transcribed.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for segment in final["segments"]:
                f.write(f"{segment['speaker']}: {segment['text']}\n")

        return f"✅ Done: {os.path.basename(output_file)}"

    except Exception as e:
        return f"❌ Failed: {os.path.basename(audio_path)} — {str(e)}"

def main():
    folder = input("Enter folder path with cleaned WAV files: ").strip()
    if not os.path.isdir(folder):
        print("❌ Invalid folder.")
        return

    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".wav")
    ]
    if not files:
        print("⚠️ No .wav files found.")
        return

    print(f"🧠 Processing {len(files)} files using {NUM_WORKERS} workers...")
    with Pool(processes=NUM_WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_file, files), total=len(files)):
            print(result)

if __name__ == "__main__":
    freeze_support()  # Required on Windows
    main()
