from sqlalchemy import Column, Integer, String, Boolean, Numeric, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True)
    first_name = Column(String)
    last_name = Column(String)
    username = Column(String)
    email = Column(String)
    role = Column(String)
    avatar_key = Column(String)

class Coach(Base):
    __tablename__ = "trainer_profiles"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"))
    years_of_experience = Column(Integer)
    city = Column(String)
    specializations = Column(String) # Branş bilgisi burada
    hero_image_key = Column(String)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # users tablosuyla ilişki - isim bilgisi için
    user = relationship("User", foreign_keys=[user_id])

class TrainerPackage(Base):
    __tablename__ = "trainer_packages"
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(String)
    price = Column(Numeric(10, 2))
    total_lessons = Column(Integer)
    duration_days = Column(Integer)
    active = Column(Boolean)
    trainer_id = Column(String) # Hangi hocaya ait olduğu

class Supplement(Base):
    __tablename__ = "supplements"
    id = Column(Integer, primary_key=True)
    brand = Column(String)
    product_name = Column(String)
    description = Column(Text)