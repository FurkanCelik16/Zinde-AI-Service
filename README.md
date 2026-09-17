# Zinde AI Service — Semantik Arama & Niyet Tespiti Tabanlı RAG Servisi

[![FastAPI](https://img.shields.io/badge/Framework-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Pinecone](https://img.shields.io/badge/Vector_DB-Pinecone-green.svg)](https://www.pinecone.io/)
[![Docker](https://img.shields.io/badge/Deployment-Docker-blue.svg)](https://www.docker.com/)
[![Architecture](https://img.shields.io/badge/Architecture-RAG%20%7C%20Microservice-orange.svg)]()

Zinde AI Service; kullanıcı sorgularını anlamsal olarak analiz eden, niyet sınıflandırması (intent classification) gerçekleştiren ve Pinecone Vektör Veritabanı üzerinden yüksek doğruluklu bağlam çekerek (RAG) yanıt üreten mikroservis mimarisinde geliştirilmiş bir yapay zeka arka uç servisidir.

---

## 📌 Temel Yetenekler & Mimari Özellikler

* **Niyet Tespiti & Yönlendirme (Intent Service):** Kullanıcının serbest metin halindeki girdisini analiz ederek doğrudan bilgi sorgusu, eylem talebi veya sohbet niyetlerini ayrıştırır (intent_service.py).
* **Semantik Vektör Arama (Vector Service):** Doğal dil sorgularını çok boyutlu embedding vektörlerine dönüştürür ve Pinecone üzerinde kosinüs benzerliği (cosine similarity) araması yapar (vector_service.py).
* **Vektör Senkronizasyon Pipeline'ı:** İlişkisel/doküman veritabanındaki verileri periyodik veya olay bazlı olarak Pinecone indekslerine senkronize eden batch işleme mekanizması (sync_to_pinecone.py).
* **Asenkron REST API:** Düşük gecikme süreli ve yüksek eşzamanlı istek kapasitesine sahip FastAPI çekirdeği (main.py).
* **Üretime Hazır Konteyner:** Bağımlılıkları izole eden, hafif tabanlı ve üretime dağıtıma hazır Docker altyapısı (Dockerfile).

---

## 🏗 Dizin & Modül Yapısı

* **main.py**: REST API uç noktalarının (endpoints), middleware ve yaşam döngüsü yönetiminin bulunduğu giriş noktası.
* **intent_service.py**: Gelen kullanıcı mesajlarının bağlamsal niyetlerini belirleyen analiz modülü.
* **vector_service.py**: Vektör arama sorgularını yöneten, embedding çıkarımını yapan Pinecone arayüzü.
* **sync_to_pinecone.py**: Ham veriyi vektörize edip Pinecone indekslerine yükleyen senkronizasyon script'i.
* **database.py**: Veritabanı bağlantı yönetimi, connection pooling ve oturum döngüsü.
* **models.py**: Veritabanı şemaları ve Pydantic veri transfer objeleri (DTO / Schemas).
* **config.py**: Pydantic tabanlı ortam değişkeni (env) ve gizli anahtar yönetimi.

---

## 🛠 Kullanılan Teknolojiler

* **Çalışma Ortamı:** Python 3.11+
* **Web Çatısı:** FastAPI / Uvicorn
* **Vektör Veritabanı:** Pinecone Vector Database
* **Embedding Modelleri:** Text-Embedding LLM Entegrasyonu
* **Veri Modelleme & Doğrulama:** Pydantic
* **Konteynerleştirme:** Docker

---

## 🚀 Kurulum ve Yerel Çalıştırma

### 1. Yerel Ortam (Python Venv)

1. Depoyu klonlayıp sanal ortamı başlatın:  
`git clone https://github.com/FurkanCelik16/Zinde-AI-Service.git`  
`cd Zinde-AI-Service`  
`python3 -m venv venv`  
`source venv/bin/activate`  
`pip install -r requirements.txt`

2. Ortam değişkenlerini tanımlayın (.env):  
`PINECONE_API_KEY=your_pinecone_key_here`  
`PINECONE_ENVIRONMENT=your_environment_here`  
`PINECONE_INDEX_NAME=zinde-index`  
`DATABASE_URL=your_database_connection_string`

3. Pinecone senkronizasyonunu çalıştırın ve servisi ayağa kaldırın:  
`python sync_to_pinecone.py`  
`uvicorn main:app --reload --port 8000`

---

### 2. Docker ile Çalıştırma

Servisi Docker konteyneri olarak derleyip başlatmak için:  
`docker build -t zinde-ai-service .`  
`docker run -p 8000:8000 --env-file .env zinde-ai-service`

API dokümantasyonuna tarayıcınızdan erişebilirsiniz:  
* Swagger UI: http://localhost:8000/docs  
* ReDoc: http://localhost:8000/redoc

---

## 📄 Lisans

Bu proje kurumsal kullanım ve eğitim amacıyla açık kaynak olarak sunulmuştur.
