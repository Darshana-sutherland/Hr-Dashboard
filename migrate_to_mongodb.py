from app import app, db, HR, Candidate, Job, Application
from mongo_models import HR as MongoHR, Candidate as MongoCandidate, Job as MongoJob, Application as MongoApplication, db as mongo_db
from werkzeug.security import generate_password_hash

def migrate_data():
    print("Starting migration from SQLite to MongoDB...")
    
    # Create indexes
    print("Creating indexes...")
    mongo_db.hr_users.create_index("username", unique=True)
    mongo_db.candidates.create_index("username", unique=True)
    mongo_db.applications.create_index([("candidate_id", 1), ("job_id", 1)])
    
    # Migrate HR Users
    print("Migrating HR users...")
    for hr in HR.query.all():
        MongoHR.create({
            "username": hr.username,
            "password_hash": hr.password_hash
        })
    
    # Migrate Candidates
    print("Migrating Candidates...")
    candidate_map = {}
    for candidate in Candidate.query.all():
        mongo_id = MongoCandidate.create({
            "username": candidate.username,
            "password_hash": candidate.password_hash,
            "name": candidate.name,
            "email": candidate.email,
            "phone": candidate.phone
        })
        candidate_map[candidate.id] = mongo_id
    
    # Migrate Jobs
    print("Migrating Jobs...")
    job_map = {}
    for job in Job.query.all():
        mongo_id = MongoJob.create({
            "title": job.title,
            "company": job.company,
            "tags": job.tags.split(',') if job.tags else []
        })
        job_map[job.id] = mongo_id
    
    # Migrate Applications
    print("Migrating Applications...")
    for app in Application.query.all():
        MongoApplication.create({
            "candidate_id": candidate_map[app.candidate_id],
            "job_id": job_map[app.job_id],
            "status": app.status,
            "created_at": app.created_at,
            "resume_filename": app.resume_filename
        })
    
    print("Migration completed successfully!")

if __name__ == '__main__':
    with app.app_context():
        migrate_data()
