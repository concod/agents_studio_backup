import whisper

def transcribe(audio_path: str):
    # Choose model: tiny | base | small | medium | large
    model = whisper.load_model("base")  

    result = model.transcribe(audio_path)
    print("\nTranscription:\n")
    print(result["text"])

if __name__ == "__main__":
    transcribe("sample.wav")
