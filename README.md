# Audio Cleaner & Speaker Transcription

This project allows you to clean noisy audio and transcribe it using OpenAI WhisperX with speaker diarization.

## Files

- `clean_audio.py`: Converts and cleans audio (mono, 16kHz, noise reduction)
- `transcribe_diarize.py`: Transcribes audio with diarization
- `requirements.txt`: Dependencies
- `.gitignore`: Exclusions

## Setup

```bash
python -m venv whisperx-env
.\whisperx-env\Scripts\activate
pip install -r requirements.txt
