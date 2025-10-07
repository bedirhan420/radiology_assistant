from graph.state import GraphState
from tools.llm_calls import get_llm, get_routing_chain, get_extraction_chain
from schemas.form_schemas import FormBloodTest, FormMRI

llm = get_llm()

def node_router(state: GraphState) -> dict:
    ''' Textin hangi form tipine uygun olduğuna karar verir. '''
    print(f"\n--- DÜĞÜM (Hasta: {state['patient_name']}): Router ---")
    
    text_chunk = state.get("text_chunk", "")

    print("DEBUG (Router): Analiz edilecek metin bloğu:")
    print("-------------------------------------------")
    print(text_chunk)
    print("-------------------------------------------")

    if text_chunk.startswith("HATA:") or text_chunk.startswith("ERROR:"):
        return {"form_type": "undefined", "error": text_chunk}
    
    prompt = f"""
    Sen, tıbbi metinleri sınıflandırma konusunda uzman bir asistansın.
    Görevin, sana verilen metin bloğunun konusunu analiz edip, aşağıdaki üç kategoriden hangisine ait olduğunu belirlemektir.

    KATEGORİLER:
    - 'mri': Eğer metin ağırlıklı olarak MR (Manyetik Rezonans) sonuçları, bulguları, raporları veya ilgili terimler (örn: lezyon, disk hernisi, kontrast tutulumu) içeriyorsa.
    - 'blood_test': Eğer metin ağırlıklı olarak kan tahlili sonuçları, kan değerleri veya ilgili terimler (örn: hemoglobin, WBC, lökosit, CRP, platelet) içeriyorsa.
    - 'undefined': Eğer metin bu iki kategoriye de girmiyorsa veya tıbbi bir içerik değilse.

    Sana verilen metin aşağıdadır. Lütfen sadece bu üç kategoriden birini seçerek yanıt ver.

    METİN:
    ---
    {text_chunk}
    ---
    """
    
    try:
        routing_chain = get_routing_chain(llm)
        result = routing_chain.invoke(prompt)
        form_type = result.form_type
        print(f"Router kararı: {form_type}")
        return {"form_type": form_type}
    except Exception as e:
        print(f"HATA (Router): LLM çağrısı sırasında hata oluştu: {e}")
        return {"form_type": "undefined", "error": f"Router LLM hatası: {e}"}


def node_extract_mri(state: GraphState) -> dict:
    """MR formuna göre veri çıkarır."""
    print(f"--- DÜĞÜM (Hasta: {state['patient_name']}): Extract MRI ---")
    extraction_chain = get_extraction_chain(llm, FormMRI)
    result = extraction_chain.invoke(state["text_chunk"])
    return {"extracted_data": result.dict()}

def node_extract_blood_test(state: GraphState) -> dict:
    """Kan Tahlili formuna göre veri çıkarır."""
    print(f"--- DÜĞÜM (Hasta: {state['patient_name']}): Extract Blood Test ---")
    extraction_chain = get_extraction_chain(llm, FormBloodTest)
    result = extraction_chain.invoke(state["text_chunk"])
    return {"extracted_data": result.dict()}

def node_handle_error(state: GraphState) -> dict:
    """Belirsiz durumlarda hata durumu oluşturur."""
    print(f"--- DÜĞÜM (Hasta: {state['patient_name']}): Handle Error ---")
    err_msg = state.get("error", "Bu hastaya ait metin içeriği anlaşılamadı veya ilgili form bulunamadı.")
    return {"error": err_msg, "extracted_data": None}