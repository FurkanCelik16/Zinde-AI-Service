"""
Zinde AI Service — Vector Store (Pinecone) Servisi
Tek noktadan Pinecone yönetimi: upsert, build_text fonksiyonları.
Hem /add-coach hem /sync-database-to-ai aynı fonksiyonları kullanır → tutarsızlık imkansız.
"""

from llama_index.core import VectorStoreIndex, Document
from config import vector_store, pinecone_index


def _get_index():
    """Her seferinde güncel VectorStoreIndex döner."""
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)


# ── Metin Oluşturma (Tek Kaynak) ─────────────────────────

def build_coach_text(coach_name: str, specializations: str, city: str, years_of_experience: int) -> str:
    """Antrenör için Pinecone'a gönderilecek metni oluşturur. Tek fonksiyon, tek format."""
    return (
        f"Zinde Antrenörü: Koç {coach_name}. "
        f"Uzmanlık Alanı: {specializations}. "
        f"Şehir: {city}. "
        f"Deneyim: {years_of_experience} yıl."
    )


def build_supplement_text(brand: str, product_name: str, description: str) -> str:
    """Supplement için Pinecone'a gönderilecek metni oluşturur."""
    return f"Supplement Bilgisi: {brand} markasının {product_name} ürünü. Açıklama: {description}"


def build_package_text(name: str, description: str, total_lessons: int, price, coach_name: str, coach_city: str) -> str:
    """Paket için Pinecone'a gönderilecek metni oluşturur."""
    return f"SPOR PAKETİ: {name}. Bu paket {coach_city} şehrindeki Antrenör {coach_name} tarafından verilmektedir. İçerik: {description}. Ders Sayısı: {total_lessons}. Fiyat: {price} TL."


# ── Upsert Fonksiyonları (doc_id bazlı — race condition yok) ─

def upsert_coach(coach_id: int, coach_name: str, specializations: str, city: str, years_of_experience: int):
    """
    Antrenörü Pinecone'a ekler veya günceller.
    doc_id = "coach_{id}" → aynı ID varsa üzerine yazar, delete gerekmez.
    """
    doc_id = f"coach_{coach_id}"
    text = build_coach_text(coach_name, specializations, city, years_of_experience)
    
    doc = Document(
        text=text,
        doc_id=doc_id,
        metadata={
            "source": "database",
            "type": "coach",
            "id": str(coach_id),
            "coach_name": coach_name,
        },
    )
    
    index = _get_index()
    # delete eski varsa, sonra insert (LlamaIndex doc_id bazlı)
    try:
        index.delete_ref_doc(doc_id)
    except Exception:
        pass  # İlk kez ekleniyorsa hata vermesini görmezden gel
    index.insert(doc)


def upsert_supplement(supp_id: int, brand: str, product_name: str, description: str):
    """Supplement'i Pinecone'a ekler veya günceller."""
    doc_id = f"supplement_{supp_id}"
    text = build_supplement_text(brand, product_name, description)
    
    doc = Document(
        text=text,
        doc_id=doc_id,
        metadata={
            "source": "database",
            "type": "supplement",
            "db_id": supp_id,
        },
    )
    
    index = _get_index()
    try:
        index.delete_ref_doc(doc_id)
    except Exception:
        pass
    index.insert(doc)


def upsert_package(pkg_id: int, name: str, description: str, total_lessons: int, price, coach_name: str, coach_city: str, user_id: str = None):
    """Paketi Pinecone'a ekler veya günceller."""
    doc_id = f"package_{pkg_id}"
    text = build_package_text(name, description, total_lessons, price, coach_name, coach_city)
    
    doc = Document(
        text=text,
        doc_id=doc_id,
        metadata={
            "source": "database",
            "type": "package",
            "id": str(pkg_id),
            "coach_name": coach_name,
            "package_name": name,
            "package_price": f"{price} TL",
            "user_id": str(user_id) if user_id else None
        },
    )
    
    index = _get_index()
    try:
        index.delete_ref_doc(doc_id)
    except Exception:
        pass
    index.insert(doc)


def delete_coach(coach_id: int):
    """Antrenörü Pinecone'dan siler."""
    doc_id = f"coach_{coach_id}"
    index = _get_index()
    try:
        index.delete_ref_doc(doc_id)
        print(f"Deleted Coach from Pinecone: {doc_id}")
    except Exception as e:
        print(f"Error deleting Coach {doc_id}: {e}")


def delete_package(pkg_id: int):
    """Paketi Pinecone'an siler."""
    doc_id = f"package_{pkg_id}"
    index = _get_index()
    try:
        index.delete_ref_doc(doc_id)
        print(f"Deleted Package from Pinecone: {doc_id}")
    except Exception as e:
        print(f"Error deleting Package {doc_id}: {e}")
