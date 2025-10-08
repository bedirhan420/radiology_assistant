# 🧠 Radyoloji Ses Dosyası Asistanı

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
║ │ MRI            │ │  Kan Tahlili     │            ║
║ │  Veri Çıkarı   │ │  Veri Çıkarıcı   │            ║
║ │ (LLM + Şema)   │ │                  │            ║
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
```

## 🏗️ Teknoloji Mimarisi

| Katman | Teknoloji / Kütüphane |
|--------|------------------------|
| **Orkestrasyon & Agent Mantığı** | LangChain, LangGraph |
| **Dil Modelleri (LLM)** | Google Gemini Pro |
| **Ses-Metin Çevrimi (S2T)** | Hugging Face Whisper |
| **Veri Yapılandırma (Schema)** | Pydantic |
| **Programlama Dili** | Python 3.10+ |

---

## ⚙️ Kurulum

### 1. Projeyi Klonlayın
```bash
git clone https://github.com/bedirhan420/radiology_assistant.git
cd radiology_assistant
```

### 2. Sanal Ortam Oluşturun ve Aktif Edin

```bash
conda create -n radiology_assistant python=3.12.11
conda activate radiology_assistant
```

### 3. Gerekli Kütüphaneleri Yükleyin

```bash
pip install -r requirements.txt
```

## 4. API Anahtarlarını Ayarlayın

Ana dizinde .env dosyası oluşturun ve aşağıdaki içeriği ekleyin:

```bash
# Google AI Studio'dan alınacak: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY="BURAYA_GOOGLE_API_ANAHTARINIZI_YAPISTIRIN"

# Hugging Face'ten alınacak: https://huggingface.co/settings/tokens
HF_TOKEN="hf_BURAYA_HUGGINGFACE_TOKENINI_YAPISTIRIN"
```
---

# 🧩 Nasıl Çalıştırılır?
## 1. Ses Dosyalarını Ekleyin

İşlemek istediğiniz .mp3 ses dosyalarını şu klasöre yerleştirin:

```bash
data/audio/
```

## 2. Orkestratörü Çalıştırın

```bash
python src/orchestrator.py data/audio/sizin_ses_dosyaniz.mp3
```

## 3. Çıktıları Kontrol Edin

İşlem tamamlandığında aşağıdaki dizinde her hasta için oluşturulan .json dosyalarını bulabilirsiniz:

```bash
data/output/orchestrator
```
---

# 📂 Dosya Yapısı

```text
/radiology_assistant/
│-- .env
│-- requirements.txt
│-- README.md
│
│-- /data/
│   │-- /audio/
│   │-- /output/
│
│-- /src/
│   │-- /graph/
│   │   │-- state.py
│   │   │-- nodes.py
│   │   │-- workflow.py
│   │
│   │-- /schemas/
│   │   │-- form_schemas.py
│   │
│   │-- /tools/
│   │   │-- s2t.py
│   │   │-- llm_calls.py
│   │
│   │-- config.py
│   │-- orchestrator.py
```







