import os
from huggingface_hub import InferenceClient
from huggingface_hub.inference._generated.types import AutomaticSpeechRecognitionOutput
from dotenv import load_dotenv
load_dotenv()

if "HF_TOKEN" not in os.environ:
    raise ValueError("Lütfen Hugging Face API anahtarınızı 'HF_TOKEN' adıyla bir ortam değişkeni olarak ayarlayın.")

# InferenceClient'ı başlatıyoruz.
# Varsayılan sağlayıcı zaten Hugging Face olduğu için belirtmeye gerek yok.
client = InferenceClient(token=os.environ["HF_TOKEN"])

# --- Yerel Ses Dosyasını İşleme ---

def transcribe_with_api(file_path: str, model_name: str = "openai/whisper-large-v3"):
    """
    Verilen yoldaki yerel bir ses dosyasını Hugging Face Inference API ile transkript eder.
    """
    # Dosyanın var olup olmadığını kontrol et
    if not os.path.exists(file_path):
        print(f"HATA: Belirtilen dosya bulunamadı -> {file_path}")
        return

    print(f"\n'{os.path.basename(file_path)}' dosyası Hugging Face API'ye gönderiliyor...")
    print(f"Kullanılan model: {model_name}")

    try:
        # API'ye isteği gönderiyoruz.
        # client.automatic_speech_recognition metodu dosya yolunu doğrudan kabul eder.
        result: AutomaticSpeechRecognitionOutput = client.automatic_speech_recognition(
            audio=file_path,
            model=model_name,
        )
        
        # API'den gelen sonucu ekrana yazdır
        print("-" * 50)
        print("Transkripsiyon Sonucu:")
        print(result.text.strip())
        print("-" * 50)

    except Exception as e:
        print(f"API isteği sırasında bir hata oluştu: {e}")


# --- KULLANIM ---

# İlk kodunuzdaki dosya adını kullanıyoruz.
# Bu dosyanın, script'i çalıştırdığınız yerde olması gerektiğini unutmayın.
path_to_your_audio = "test_audio.mp3"

# Fonksiyonu çağırarak işlemi başlat
transcribe_with_api(path_to_your_audio)