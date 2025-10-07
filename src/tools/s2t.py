import os
from huggingface_hub import InferenceClient
from huggingface_hub.inference._generated.types import AutomaticSpeechRecognitionOutput
from dotenv import load_dotenv
load_dotenv()

if "HF_TOKEN" not in os.environ:
    raise ValueError("Please add your Hugging Face API key to an .env file or environment variable with the name 'HF_TOKEN'.")

client = InferenceClient(token=os.environ.get("HF_TOKEN"))

def transcribe_audio(audio_path:str,model:str)->str:
    if not os.path.exists(audio_path):
        err = f"ERROR : Audio file not found -> {audio_path}"
        print(err)
        return err
    
    try:
        result:AutomaticSpeechRecognitionOutput = client.automatic_speech_recognition(
            audio=audio_path,
            model=model
        )
        transcript = result.text.strip()
        print("Transcription is taken from API successfully")
        return transcript
    except Exception as e:
        err = f"ERROR : API request error -> {e}"
        print(err)
        return err