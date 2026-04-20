import os
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
from vector_service import upsert_coach, upsert_package, upsert_supplement

def sync_data():
    db = SessionLocal()
    print("Connecting to DB and Pinecone...")
    
    try:
        # 1. Sync Coaches
        coaches = db.query(models.Coach).all()
        print(f"Found {len(coaches)} coaches. Syncing...")
        for coach in coaches:
            full_name = f"{coach.user.first_name} {coach.user.last_name}" if coach.user else "Bilinmeyen Hoca"
            upsert_coach(
                coach_id=coach.id,
                coach_name=full_name,
                specializations=coach.specializations or "Genel Fitness",
                city=coach.city or "Türkiye",
                years_of_experience=coach.years_of_experience or 0
            )
            print(f"Synced Coach: {full_name}")

        # 2. Sync Packages
        packages = db.query(models.TrainerPackage).all()
        print(f"Found {len(packages)} packages. Syncing...")
        for pkg in packages:
            # Paket hangi hocaya ait bulalım (user_id üzerinden)
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
            print(f"Synced Package: {pkg.name} (Coach: {coach_name})")

        # 3. Sync Supplements
        supps = db.query(models.Supplement).all()
        print(f"Found {len(supps)} supplements. Syncing...")
        for supp in supps:
            upsert_supplement(
                supp_id=supp.id,
                brand=supp.brand,
                product_name=supp.product_name,
                description=supp.description
            )
            print(f"Synced Supplement: {supp.product_name}")

        print("\nSUCCESS: All data synced to Pinecone (768 dimensions).")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    sync_data()
