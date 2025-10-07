import json
import argparse
import os
import time # Bekleme süresi için time kütüphanesini ekliyoruz

from config import ORCHESTRATOR_OUTPUT_DIR, WHISPER_MODEL_NAME
from tools.s2t import transcribe_audio
from tools.llm_calls import get_llm
from schemas.form_schemas import TranscriptAnalysis
from graph.workflow import app

def run_orchestrator(audio_path: str):
    if not os.path.exists(audio_path):
        print(f"HATA: Belirtilen ses dosyası bulunamadı -> {audio_path}")
        return

    print("--- ORKESTRATÖR BAŞLATILDI ---")
    
    print("\n--- AŞAMA 1: Transkripsiyon ---")
    full_transcript = transcribe_audio(audio_path, WHISPER_MODEL_NAME)
    
    if full_transcript.startswith("HATA:") or full_transcript.startswith("ERROR:"):
        print(f"Transkripsiyon başarısız olduğu için süreç durduruldu: {full_transcript}")
        return
        
    print("Tam Transkript Alındı.")

    print("\n--- AŞAMA 2: Metin Analizi ve Hasta Bazında Gruplama ---")
    llm = get_llm()
    chunking_chain = llm.with_structured_output(TranscriptAnalysis)
    
    prompt = f"""
    Senin görevin, aşağıdaki karmaşık tıbbi metni analiz etmektir.
    Metindeki tüm farklı hastaları ve onlarla ilgili BÜTÜN bilgileri ayıklamalısın.
    
    Şu adımları izle:
    1. Metinde adı geçen her bir benzersiz hastayı (örn: 'Ahmet Yılmaz') bul.
    2. Her hasta için, metnin farklı yerlerinde dağınık halde bulunan, o hastaya ait TÜM cümleleri topla.
    3. Topladığın bu cümleleri, her hasta için tek bir metin bloğu olacak şekilde birleştir.
    
    Çıktın, 'TranscriptAnalysis' şemasına uygun bir JSON formatında olmalıdır. 'patient_name' ve 'related_text' alanlarının her ikisinin de dolu olduğundan emin ol.

    İşlenecek Transkript:
    ---
    {full_transcript}
    ---
    """
    
    try:
        analiz_sonucu = chunking_chain.invoke(prompt)
        print("DEBUG (Orchestrator): Gruplama sonrası analiz sonucu:", analiz_sonucu)
        print(f"Analiz tamamlandı. {len(analiz_sonucu.patient_chunks)} farklı hasta bulundu.")
    except Exception as e:
        print(f"HATA: Metin analizi ve gruplama aşamasında LLM hatası: {e}")
        return

    ORCHESTRATOR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- AŞAMA 3: Her Hasta İçin Rapor Üretme Agent'ı Devreye Giriyor ---")
    for chunk in analiz_sonucu.patient_chunks:
        print(f"\n>>> Hasta: '{chunk.patient_name}' işleniyor...")
        
        graph_input = {
            "patient_name": chunk.patient_name,
            "text_chunk": chunk.related_text
        }
        final_state = app.invoke(graph_input)

        output_filename = ORCHESTRATOR_OUTPUT_DIR / f"{chunk.patient_name.replace(' ', '_')}.json"
        output_data = final_state
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        print(f"<<< Sonuç '{output_filename.name}' dosyasına kaydedildi.")

        print("\nAPI rate limitini aşmamak için 35 saniye bekleniyor...")
        time.sleep(35)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bir ses dosyasını işleyip içindeki hasta raporlarını analiz eden Orkestratör Agent."
    )
    parser.add_argument(
        "--audio_file", 
        type=str, 
        help="İşlenecek ses dosyasının tam yolu. Örn: data/audio/orchestrator_test.mp3"
    )
    args = parser.parse_args()
    run_orchestrator(args.audio_file)