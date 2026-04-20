"""
Zinde AI Service — Konfigürasyon
LLM, Embedding (Custom Light), Pinecone ve Prompt ayarları.
"""

import os
import time
import requests
from typing import List, Any
from dotenv import load_dotenv

from llama_index.core import Settings, PromptTemplate
from llama_index.llms.groq import Groq
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pydantic import Field
from pinecone import Pinecone

load_dotenv()

# ── Custom Light HF Embedding (No Torch!) ────────────────
class HFLightEmbedding(BaseEmbedding):
    """Hugging Face Inference API'ye doğrudan requests atan hafif sınıf."""
    model_name: str = Field(description="Hugging Face model ismi")
    token: str = Field(description="Hugging Face API Token")
    api_url: str = Field(description="İstek atılacak URL")

    def __init__(self, model_name: str, token: str, **kwargs):
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        super().__init__(model_name=model_name, token=token, api_url=api_url, **kwargs)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        if not self.token or len(self.token) < 5:
             raise ValueError("HF_TOKEN eksik veya çok kısa. Lütfen Railway ayarlarını kontrol edin.")
             
        headers = {"Authorization": f"Bearer {self.token}"}
        
        # Daha sabırlı bir bekleme mekanizması (5 deneme x 15 saniye)
        for attempt in range(5):
            try:
                response = requests.post(self.api_url, headers=headers, json={"inputs": text}, timeout=15)
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], list):
                            return result[0]
                        return result
                    raise ValueError(f"HF API geçerli bir liste döndürmedi: {result}")
                
                err_text = response.text
                if "loading" in err_text.lower() or response.status_code == 503:
                    print(f"[HF] Model uyanıyor ({self.model_name})... {attempt+1}. deneme...")
                    time.sleep(20)
                    continue
                
                # Başka bir hata varsa (401, 404, 400 vb) doğrudan fırlat
                raise Exception(f"HF API Hatası ({response.status_code}): {err_text}")
                
            except Exception as e:
                # Eğer tüm denemeler bittiyse veya kritik bir hataysa
                if attempt == 4:
                    raise e
                if "loading" not in str(e).lower():
                    raise e
                time.sleep(20)
        
        raise Exception("Hugging Face modeli uyanamadı. Lütfen birkaç dakika sonra tekrar deneyin.")

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

# ── Groq LLM ──────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

llm = Groq(model=GROQ_MODEL, api_key=GROQ_API_KEY)
Settings.llm = llm

# Tiny & Rapid Embedding (Cloud-based API, zero local footprint)
HF_TOKEN = os.getenv("HF_TOKEN")
Settings.embed_model = HFLightEmbedding(
    model_name="BAAI/bge-small-en-v1.5", 
    token=HF_TOKEN
)

# ── Pinecone ──────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index("zinde-index")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# ── Zinde Asistan Prompt'u ────────────────────────────────
ZINDE_PROMPT_STR = (
    "Sen Zinde uygulamasının akıllı spor asistanısın. Görevin, kullanıcılara spor hedefleri, antrenörler ve paketler hakkında yardımcı olmaktır. SADECE Türkçe yanıt ver.\n\n"
    "KURALLAR:\n"
    "1. Sakatlanma, hastalık, teşhis veya ilaç tavsiyesi gibi ağır tıbbi durumlarda SADECE 'Bu konu tıbbi uzmanlık gerektirir, doktora danışın' de. Bunun dışında antrenman yöntemleri, spor faydaları veya esnediğinde kasların rahatlaması gibi genel spor sohbetlerinde kesinlikle esprili, motive edici ve doğal bir dille muhabbete katıl.\n"
    "2. Cevapların çok kısa, net, samimi ve zekice olsun. Emojileri ÇOK AZ ve sadece gerçekten gerektiğinde kullan (en fazla 1 emoji). Abartılı emoji kullanmaktan KESİNLİKLE kaçın.\n"
    "3. Kullanıcı antrenör, paket veya fiyat soruyorsa ÖNCELİKLE aşağıdaki BİLGİ KAYNAĞI verilerini kullan. Ancak kullanıcı sporla (örn: damacana çalışması, evde spor) veya hayatla ilgili genel şeyler soruyorsa KENDİ BİLGİNİ VE MANTIĞINI KULLANARAK muhabbet et! Robot gibi davranma.\n"
    "4. DİKKAT: BİLGİ KAYNAĞI BOŞSA SADECE ve SADECE kullanıcı uygulamadan spesifik bir randevu, hoca hizmeti veya ürün (örn: Kreatin markası vb) incelemek/bulmak istiyorsa 'Sistemimizde buna dair bir paket/hoca hizmeti bulunmuyor' gibi DOĞAL BİLGİ CÜMLESİYLE reddet. Asla robot gibi 'Kayıt bulamadım' deme. Ama 'Kreatin zararlı mı?', 'Squat nasıl yapılır?' gibi genel bir eğitim/spor/supplement/bilgi sorusu soruyorsa, bu ürünler mağazamızda SATILMIYOR OLSA BİLE kendi bilginle cevap ver! Asla 'bilgi bulamadım' deme.\n"
    "5. Kullanıcı senden matematiksel bir soru soruyorsa (örneğin: 'En ucuz paket hangisi?', 'En deneyimli hoca kim?'): BİLGİ KAYNAĞINDAKİ tüm verileri kendi içinde oku, analiz et ve kullanıcıya SADECE çıkan sonucu tek cümleyle söyle! Düşünce sürecini, karşılaştırdığın diğer paketleri veya bulamadığın hocaları SAKIN LİSTELEME VE YAZMA. Sadece direkt cevabı ver.\n"
    "6. DİKKAT: Kullanıcı spesifik bir şey soruyorsa SADECE o konuyla en alakalı 1 (maksimum 2) paketi/hocayı öner. Çeşit olsun diye alakasız veya ilgisiz paketleri sakın listeme, lafı uzatma! Ancak kullanıcının sorusu tamamen genelse (örn: 'Tüm Paketleriniz neler?') o zaman en fazla 3 tane örnek verebilirsin.\n\n"
    "BİLGİ KAYNAĞI:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "KULLANICI SORUSU: {query_str}\n"
    "ZİNDE ASİSTAN: "
)
ZINDE_PROMPT = PromptTemplate(ZINDE_PROMPT_STR)
