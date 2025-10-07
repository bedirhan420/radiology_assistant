# 🧠 Medikal Rapor Yapılandırma Agent'ı

Bu proje, içerisinde birden fazla hastaya ait karmaşık ve sırasız bilgiler içeren ses kayıtlarını analiz eden ve **her hasta için yapılandırılmış (JSON formatında) raporlar üreten yapay zeka tabanlı bir agent sistemidir.**

---

## 🎯 Projenin Amacı

Tıbbi ortamlarda doktorlar genellikle vizit sonrası notlarını uzun bir ses kaydına dikte ederler.  
Bu dikteler sırasında:
- Birden fazla hastadan bahsedilebilir,  
- Farklı hastaların bilgileri birbirine karışabilir,  
- Daha önce bahsedilen bir hastaya “geri dönüş” yapılabilir.  

Bu **karmaşık ve doğrusal olmayan** ses kayıtlarını manuel olarak deşifre etmek oldukça zaman alıcıdır.  
Bu proje, bu süreci **tamamen otomatik hale getirerek**, tek bir ses kaydından **hasta bazında ayrıştırılmış, sınıflandırılmış ve yapılandırılmış tıbbi raporlar** üretmeyi hedefler.

---

## 🚀 Öne Çıkan Özellikler

- **🎧 Tek Ses Dosyasından Çoklu Rapor:**  
  Birden fazla hastanın bilgisini içeren tek bir uzun ses dosyasını işleyebilir.

- **🧩 Akıllı Gruplama:**  
  Aynı hastaya ait, farklı yerlerde bahsedilen bilgileri birleştirir.

- **🤖 Agent Mimarisi (2 Aşamalı):**  
  - **Orkestratör Agent:** Hastaları tespit eder ve transkripti anlamlı bloklara ayırır.  
  - **Rapor Üretme Agent’ı (LangGraph):** Her metin bloğunu detaylı işleyip yapılandırılmış veriye dönüştürür.

- **🩸 Otomatik Sınıflandırma:**  
  Her hasta raporunun türünü (örneğin *MR Raporu*, *Kan Tahlili*) içerik analizine göre belirler.

- **🧱 Yapısal Veri Çıktısı:**  
  Pydantic şemalarıyla tutarlı ve temiz JSON formatı üretir.

- **📁 Hasta Bazında Çıktı:**  
  Her hasta için ayrı `.json` dosyası oluşturur.

---

## 🧬 Mimari ve İş Akışı

```text
    ┌─────────────────────────┐
    │  Tek Ses Dosyası (.mp3) │
    └───────────┬─────────────┘
                │
                ▼
┌───────────────────────────────┐
│ 1. Ses-Metin Çevrimi (S2T)    │
│   (Hugging Face Whisper API)  │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ 2. Orkestratör Agent (LLM)    │
│   - Hastaları Tespit Et       │
│   - Metinleri Grupla          │
└───────────────┬───────────────┘
                │ (Hasta A Metni), (Hasta B Metni), ...
                │
                ▼ (Her hasta metni için döngü)
╔════════════════════════════════════════════════════╗
║ 3. Rapor Üretme Agent'ı (LangGraph ile kuruldu)    ║
║                                                    ║
║      ┌──────────────────┐                          ║
║      │ Router (LLM)     │--> 'mri', 'blood_test'   ║
║      └────────┬─────────┘                          ║
║               │ (Koşullu Yönlendirme)              ║
║      ┌────────┴────────┐                           ║
║      ▼                 ▼                           ║
║ ┌────────────────┐ ┌──────────────────┐            ║
║ │ MRI Veri Çıkarıcı│ │ Kan Tahlili     │            ║
║ │ (LLM + Şema)   │ │ Veri Çıkarıcı   │            ║
║ └───────┬────────┘ └────────┬─────────┘            ║
║         │                  │                       ║
║         └────────┬─────────┘                       ║
║                  ▼                                 ║
║      ┌──────────────────┐                          ║
║      │ Yapısal JSON Veri│                          ║
║      └──────────────────┘                          ║
╚════════════════════════════════════════════════════╝
                │
                ▼
┌───────────────────────────────┐
│  Hasta_A.json, Hasta_B.json   │
└───────────────────────────────┘
