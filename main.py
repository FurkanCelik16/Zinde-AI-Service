"""
Zinde AI RAG Service — Ana Uygulama
Sadece endpoint tanımları. İş mantığı servis modüllerinde.
"""

from __future__ import annotations
import uuid
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader

from database import engine, Base, get_db, SessionLocal
from config import vector_store, llm
import models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from intent_service import classify_intent, get_query_engine
from vector_service import upsert_coach, upsert_supplement, upsert_package

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Zinde AI RAG Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Frontend'in her adresten istek atmasına izin ver
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request Modelleri ─────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str
    
class QueryRequest(BaseModel):
    query: str
    history: list[ChatMessage] = []

class DocumentRequest(BaseModel):
    text: str
    author: str
    category: str


# ── Genel Endpoint'ler ────────────────────────────────────

@app.get("/")
def read_root():
    return {"status": "active", "service": "Zinde AI"}


@app.post("/ask-question")
async def ask_question(request: QueryRequest):
    """LLM-based intent classification ile sorguyu yönlendirir ve interactive_cards döndürür."""
    try:
        # Hızlı karşılama kontrolü (bağlamdan bağımsız 'merhaba' yazan kullanıcılar için)
        quick_greetings = ["merhaba", "selam", "günaydın", "iyi günler", "iyi akşamlar", "merhabalar", "hey", "selamlar"]
        if request.query.lower().strip() in quick_greetings:
            return {
                "status": "success",
                "answer": "Merhaba! Size nasıl yardımcı olabilirim?",
                "intent": "greeting",
                "interactive_cards": []
            }
            
        contextualized_query = request.query
        
        # Soru eğer eksikse önceki mesajlardan bağlam kur
        if request.history:
            history_lines = []
            for msg in request.history[-4:]:
                role_tr = "Kullanıcı" if msg.role == "user" else "Asistan"
                history_lines.append(f"{role_tr}: {msg.content}")
            
            if history_lines:
                history_str = "\n".join(history_lines)
                prompt = (
                    "Aşağıda bir kullanıcı ve asistan arasındaki sohbetin geçmişi verilmiştir.\n"
                    "Buna bakarak, kullanıcının en son sorduğu soruyu kendi başına anlaşılabilecek net, eksiksiz bir soruya dönüştür. "
                    "Eğer soru zaten netse, aynen bırak. Sadece dönüştürülmüş soruyu yaz, açıklama yapma.\n\n"
                    f"Geçmiş:\n{history_str}\n\nSon Soru: {request.query}\nBağımsız Soru:"
                )
                contextualized_query = llm.complete(prompt).text.strip()
                
        print(f"\n[{'='*40}]\n[USER QUERY]: {request.query}\n[CONTEXTUALIZED QUERY]: {contextualized_query}\n")
        
        intent = classify_intent(contextualized_query)
        print(f"[CLASSIFIED INTENT]: {intent}\n")
        
        if intent == "greeting":
            return {
                "status": "success",
                "answer": "Merhaba! Ben Zinde AI. Spor hedeflerine ulaşman, aradığın antrenörü bulman veya paketlerimizi incelemen için buradayım. Sana nasıl yardımcı olabilirim?",
                "intent": intent,
                "interactive_cards": []
            }
            
        if intent == "irrelevant":
            return {
                "status": "success",
                "answer": "Üzgünüm, sadece Zinde uygulamasının spor asistanıyım. Size yalnızca spor, sağlık ve uygulamamızdaki hocalar/paketler hakkında yardımcı olabilirim.",
                "intent": intent,
                "interactive_cards": []
            }
            
        query_engine = get_query_engine(intent)
        response = query_engine.query(contextualized_query)
        
        answer_text = str(response)
        print(f"[LLM RESPONSE ENGINE OUTPUT]: {answer_text}\n")
        if hasattr(response, 'metadata'):
            print(f"[LLM RESPONSE METADATA]: {response.metadata}\n")
        
        # Yapay zekanın LlamaIndex tarafından CÜZDANINA ALDIĞI (okuduğu) belgeler
        interactive_cards = []
        seen_cards = set()
        
        answer_text = str(response)
        if answer_text.strip() == "Empty Response" or not answer_text.strip():
            answer_text = "Üzgünüm, aradığınız kriterlere uygun bir bilgi veya kayıt bulamadım."
            
        answer_lower = answer_text.lower()
        
        if hasattr(response, "source_nodes"):
            for source_node in response.source_nodes:
                metadata = getattr(source_node.node, "metadata", {})
                
                card_type = metadata.get("type")
                card_id = metadata.get("id") or metadata.get("db_id")
                
                if not (card_type and card_id):
                    continue
                    
                # HEURISTIC KONTROLÜ: Yapay zeka bu belgeyi okudu ama cevaba dahil etti mi?
                is_mentioned = False
                
                if card_type == "coach":
                    coach_name = metadata.get("coach_name", "").lower()
                    if coach_name:
                        # Antrenörün ismi veya soyismi cevapta en az 1 kere geçiyor mu?
                        # (örn "ahmet yılmaz" ise "ahmet" kelimesinin cevapta geçmesi yeterli)
                        name_parts = coach_name.split()
                        if any(len(part) > 2 and part in answer_lower for part in name_parts):
                            is_mentioned = True
                elif card_type == "package":
                    package_name = metadata.get("package_name", "").lower()
                    if package_name:
                        # Punctuation'ları kaldırarak katı eşleştirme yapıyoruz ki "trainer" kelimesi her paketi tetiklemesin
                        import string
                        clean_answer = answer_lower.translate(str.maketrans('', '', string.punctuation)).replace("  ", " ")
                        clean_pkg = package_name.lower().translate(str.maketrans('', '', string.punctuation)).replace("  ", " ")
                        
                        if clean_pkg in clean_answer:
                            is_mentioned = True
                        else:
                            # Opsiyonel: Eğer AI paketin başını yazmış sonunu yazmamışsa ihtimaline karşı kelime bazlı daha katı bir kontrol
                            pkg_words = [w for w in clean_pkg.split() if w not in ["trainer", "paketi", "paket", "programi", "program", "egitimi"]]
                            if pkg_words and all(w in clean_answer for w in pkg_words):
                                is_mentioned = True
                else:
                    is_mentioned = True
                    
                
                if is_mentioned:
                    card_key = f"{card_type}_{card_id}"
                    if card_key not in seen_cards:
                        seen_cards.add(card_key)
                        
                        try:
                            clean_id = int(card_id)
                        except ValueError:
                            clean_id = card_id
                            
                        title_val = metadata.get("package_name") or metadata.get("coach_name") or f"{card_type} #{clean_id}"
                        sub_val = metadata.get("package_price") or "Detayları görmek için dokunun"
                        if card_type == "package" and metadata.get("package_price"):
                             sub_val = f"{sub_val} - Detayları görmek için dokunun"
                        
                        interactive_cards.append({
                            "type": card_type,
                            "id": clean_id,
                            "title": title_val,
                            "subtitle": sub_val,
                            "user_id": metadata.get("user_id")
                        })
                        
        # Her türlü senaryoda en fazla 3 kart dönsün (kullanıcı isteği)
        interactive_cards = interactive_cards[:3]

        return {
            "status": "success", 
            "answer": answer_text, 
            "intent": intent,
            "interactive_cards": interactive_cards
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Senkronizasyon Endpoint'i (Pinecone 768px Re-indexing) ──

@app.get("/sync-to-pinecone")
async def sync_to_pinecone(db: Session = Depends(get_db)):
    """Veritabanındaki her şeyi Pinecone'a tekrar basar (Gemini 768 boyutlu index için)."""
    try:
        results = {"coaches": 0, "packages": 0, "supplements": 0}
        
        # 1. Coaches
        coaches = db.query(models.Coach).all()
        for coach in coaches:
            full_name = f"{coach.user.first_name} {coach.user.last_name}" if coach.user else "Bilinmeyen Hoca"
            upsert_coach(
                coach_id=coach.id,
                coach_name=full_name,
                specializations=coach.specializations or "Genel Fitness",
                city=coach.city or "Türkiye",
                years_of_experience=coach.years_of_experience or 0
            )
            results["coaches"] += 1

        # 2. Packages
        packages = db.query(models.TrainerPackage).all()
        for pkg in packages:
            coach = db.query(models.Coach).filter(models.Coach.user_id == pkg.trainer_id).first()
            coach_name = f"{coach.user.first_name} {coach.user.last_name}" if coach and coach.user else "Zinde Hocası"
            coach_city = coach.city if coach else "Türkiye"
            user_id = coach.user_id if coach else None
            
            upsert_package(
                pkg_id=pkg.id,
                name=pkg.name,
                description=pkg.description,
                total_lessons=pkg.total_lessons,
                price=pkg.price,
                coach_name=coach_name,
                coach_city=coach_city,
                user_id=user_id
            )
            results["packages"] += 1

        # 3. Supplements
        supps = db.query(models.Supplement).all()
        for supp in supps:
            upsert_supplement(
                supp_id=supp.id,
                brand=supp.brand,
                product_name=supp.product_name,
                description=supp.description
            )
            results["supplements"] += 1

        return {"status": "success", "synced": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Supplement Endpoint'leri ──────────────────────────────

@app.post("/add-supplement")
async def add_supplement(brand: str, name: str, desc: str, db: Session = Depends(get_db)):
    try:
        # 1. PostgreSQL'e kaydet
        new_supp = models.Supplement(brand=brand, product_name=name, description=desc)
        db.add(new_supp)
        db.commit()
        db.refresh(new_supp)

        # 2. Pinecone'a upsert (tek fonksiyon)
        upsert_supplement(new_supp.id, brand, name, desc)

        return {"status": "success", "db_id": new_supp.id, "message": "Veri hem DB'ye hem AI'ya işlendi."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/supplements")
def get_supplements(db: Session = Depends(get_db)):
    try:
        supplements = db.query(models.Supplement).all()
        return {"status": "success", "count": len(supplements), "data": supplements}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Antrenör Endpoint'leri ────────────────────────────────

@app.get("/coaches")
def get_coaches(db: Session = Depends(get_db)):
    """Veritabanındaki tüm hocaları user bilgisiyle birlikte döner."""
    coaches = db.query(models.Coach).all()
    result = []
    for coach in coaches:
        coach_data = {
            "id": coach.id,
            "user_id": str(coach.user_id),
            "first_name": coach.user.first_name if coach.user else None,
            "last_name": coach.user.last_name if coach.user else None,
            "full_name": f"{coach.user.first_name} {coach.user.last_name}" if coach.user else None,
            "specializations": coach.specializations,
            "city": coach.city,
            "years_of_experience": coach.years_of_experience,
            "hero_image_key": coach.hero_image_key,
            "created_at": str(coach.created_at) if coach.created_at else None,
            "updated_at": str(coach.updated_at) if coach.updated_at else None,
        }
        result.append(coach_data)
    return {"status": "success", "count": len(result), "data": result}


@app.post("/add-coach")
async def add_coach(user_id: str, name: str, specialty: str, city: str, exp: int, db: Session = Depends(get_db)):
    try:
        new_coach = models.Coach(
            user_id=uuid.UUID(user_id),
            specializations=specialty,
            city=city,
            years_of_experience=exp
        )
        db.add(new_coach)
        db.commit()
        db.refresh(new_coach)

        # Pinecone'a upsert (tek fonksiyon, sync ile aynı format)
        upsert_coach(new_coach.id, name, specialty, city, exp)

        return {"status": "success", "db_id": new_coach.id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"DB Hatası: {str(e)}")


def run_database_sync(db: Session):
    """Veritabanından verileri çekip Pinecone'a pushlayan çekirdek fonksiyon."""
    # 1. Antrenörleri upsert et
    coaches = db.query(models.Coach).all()
    for coach in coaches:
        coach_name = (
            f"{coach.user.first_name} {coach.user.last_name}"
            if coach.user else f"Antrenör #{coach.id}"
        )
        upsert_coach(coach.id, coach_name, coach.specializations, coach.city, coach.years_of_experience)

    # 2. Paketleri upsert et
    packages = db.query(models.TrainerPackage).filter(models.TrainerPackage.active == True).all()
    for pkg in packages:
        # fetch coach for details
        coach = db.query(models.Coach).filter(models.Coach.user_id == pkg.trainer_id).first()
        coach_city = coach.city if coach else "Belirtilmemiş"
        coach_name = f"{coach.user.first_name} {coach.user.last_name}" if (coach and coach.user) else "Antrenör"
        upsert_package(pkg.id, pkg.name, pkg.description, pkg.total_lessons, pkg.price, coach_name, coach_city, pkg.trainer_id)

    return len(coaches), len(packages)


@app.post("/sync-database-to-ai")
async def sync_database(db: Session = Depends(get_db)):
    """
    DB'deki tüm coach ve package verilerini Pinecone'a upsert eder.
    doc_id bazlı çalışır → delete+reinsert yok, race condition yok.
    """
    try:
        coaches_count, packages_count = run_database_sync(db)
        return {
            "status": "success",
            "message": f"{coaches_count} hoca ve {packages_count} paket AI hafızasına senkronize edildi."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))