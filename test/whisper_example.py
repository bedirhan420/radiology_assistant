import torch
from transformers import pipeline
import librosa
import os
import warnings

# --- Kurulum ve Model Yükleme (Değişiklik Yok) ---

# Performans uyarısını bastırmak için (isteğe bağlı)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# 1. Cihazı Belirle: Mümkünse GPU (cuda), değilse CPU kullan.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Kullanılan cihaz: {device}")

# 2. Model Seçimi
model_id = "openai/whisper-large-v3"

# 3. "pipeline" Oluşturma
print(f"'{model_id}' modeli belleğe yükleniyor... (Bu işlem ilk çalıştırmada uzun sürebilir)")
pipe = pipeline(
    "automatic-speech-recognition",
    model=model_id,
    device=device
)
print(f"'{model_id}' modeli yüklendi ve kullanıma hazır.")

# --- Yerel Ses Dosyasını İşleme (Yeni ve Basitleştirilmiş Kısım) ---

def transcribe_local_file(file_path):
    """
    Verilen yoldaki bir ses dosyasını yükler ve transkript eder.
    """
    if not os.path.exists(file_path):
        print(f"HATA: Belirtilen dosya bulunamadı -> {file_path}")
        return

    print(f"\n'{os.path.basename(file_path)}' dosyası işleniyor...")

    try:
        # librosa ile ses dosyasını yükle. Whisper 16kHz örnekleme hızı bekler.
        # sr=16000 parametresi, sesi otomatik olarak bu hıza dönüştürür.
        audio_array, sampling_rate = librosa.load(file_path, sr=16000)

        # Pipeline'ı kullanarak transkripsiyon yap
        print("Transkripsiyon işlemi yapılıyor... (Dosya uzunluğuna göre zaman alabilir)")
        result = pipe(
            audio_array,
            generate_kwargs={"language": "turkish", "task": "transcribe"}
        )
        
        print("-" * 50)
        print("Transkripsiyon Sonucu:")
        print(result["text"].strip())
        print("-" * 50)

    except Exception as e:
        print(f"Dosya işlenirken bir hata oluştu: {e}")


path_to_your_audio = "test_audio.mp3"

transcribe_local_file(path_to_your_audio)
