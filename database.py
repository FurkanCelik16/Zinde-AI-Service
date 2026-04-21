import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

# .env dosyana eklediğin DATABASE_URL'i çeker
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

if SQLALCHEMY_DATABASE_URL and SQLALCHEMY_DATABASE_URL.startswith("postgres://"):
    SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Veritabanı motorunu oluştur
if not SQLALCHEMY_DATABASE_URL:
    print("WARNING: DATABASE_URL not set. Database features will fail.")
    # Railway'de patlamasın diye dummy bir sqlite engine oluşturalım (opsiyonel ama güvenli)
    engine = create_engine("sqlite:///./fallback.db", connect_args={"check_same_thread": False})
else:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Veritabanı ile konuşacak oturum (Session) ayarı
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Tablolarımızı bu sınıftan türeteceğiz
Base = declarative_base()

# Veritabanı oturumunu yöneten yardımcı fonksiyon
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()