from app import app, db, HR, Candidate
from mongo_models import HR as MongoHR, Candidate as MongoCandidate
from werkzeug.security import generate_password_hash

def migrate_credentials():
    print("Starting migration of credentials to MongoDB...")
    
    # Migrate HR Users
    print("Migrating HR users...")
    for hr in HR.query.all():
        # Check if HR user already exists in MongoDB
        existing_hr = MongoHR.find_by_username(hr.username)
        if not existing_hr:
            MongoHR.create({
                "username": hr.username,
                "password_hash": hr.password_hash,
                "created_at": datetime.utcnow()
            })
            print(f"Migrated HR user: {hr.username}")
    
    # Migrate Candidates
    print("\nMigrating Candidates...")
    for candidate in Candidate.query.all():
        # Check if candidate already exists in MongoDB
        existing_candidate = MongoCandidate.find_by_username(candidate.username)
        if not existing_candidate:
            MongoCandidate.create({
                "username": candidate.username,
                "password_hash": candidate.password_hash,
                "name": candidate.name,
                "email": candidate.email,
                "phone": candidate.phone,
                "created_at": datetime.utcnow()
            })
            print(f"Migrated candidate: {candidate.username}")
    
    print("\nCredential migration completed successfully!")

if __name__ == '__main__':
    with app.app_context():
        from datetime import datetime
        migrate_credentials()
