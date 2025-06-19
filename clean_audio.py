import os
from pydub import AudioSegment
from pydub.utils import which
import noisereduce as nr
import soundfile as sf
import numpy as np

# Steg 1: Se till att ffmpeg och ffprobe är tillgängliga
os.environ["PATH"] += os.pathsep + r"C:\Users\SebastianBlomstedt\ffmpeg\bin"
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

# Steg 2: Konvertera till mono WAV, 16kHz
def convert_to_mono_wav(input_file, output_file, target_rate=16000):
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_channels(1).set_frame_rate(target_rate)
    audio.export(output_file, format="wav")
    print(f"✅ Konverterad till mono 16kHz WAV: {output_file}")

# Steg 3: Brusreducera baserat på 10–5 sekunder från slutet
def reduce_noise(wav_file, output_file, prop_decrease=0.6):
    data, rate = sf.read(wav_file)

    # Konvertera till mono om det fortfarande är stereo
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    total_samples = len(data)

    # Hämta brusprofil mellan -10s och -5s från slutet
    noise_start = max(0, total_samples - int(rate * 10))
    noise_end = max(0, total_samples - int(rate * 5))

    if noise_end > noise_start:
        noise_sample = data[noise_start:noise_end]
    else:
        print("⚠️ Ljudfilen är för kort för att extrahera brusprofil från slutet. Använder början istället.")
        noise_sample = data[0:int(rate * 1.5)]  # fallback: första 1.5 sekunder

    # Brusreducering
    reduced = nr.reduce_noise(
        y=data,
        y_noise=noise_sample,
        sr=rate,
        prop_decrease=prop_decrease
    )

    # Volymnormalisering
    reduced /= np.max(np.abs(reduced))

    sf.write(output_file, reduced, rate)
    print(f"✅ Rensad och normaliserad ljudfil sparad: {output_file}")

# Steg 4: Kör hela pipeline
def main():
    input_file = input("Enter path to your audio file: ").strip()

    if not os.path.isfile(input_file):
        print("❌ Filen hittades inte. Kontrollera sökvägen.")
        return

    file_name, file_ext = os.path.splitext(input_file)
    wav_file = file_name + "_converted.wav"
    cleaned_file = file_name + "_cleaned.wav"

    convert_to_mono_wav(input_file, wav_file)
    reduce_noise(wav_file, cleaned_file)

if __name__ == "__main__":
    main()
