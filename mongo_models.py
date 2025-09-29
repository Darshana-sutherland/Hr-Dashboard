from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId
from config import Config

# Initialize MongoDB client
def get_mongodb_connection():
    client = MongoClient(Config.MONGODB_URI)
    # Get the database name from the URI or use a default
    db_name = Config.MONGODB_URI.split('/')[-1].split('?')[0]
    return client[db_name] if db_name else client.get_database()

# Global database connection
db = get_mongodb_connection()

class MongoModel:
    collection = None
    
    @classmethod
    def find_by_id(cls, id):
        return db[cls.collection].find_one({"_id": ObjectId(id)})
    
    @classmethod
    def find_all(cls, query=None):
        if query is None:
            query = {}
        return list(db[cls.collection].find(query))
    
    @classmethod
    def create(cls, data):
        if 'created_at' not in data:
            data['created_at'] = datetime.utcnow()
        result = db[cls.collection].insert_one(data)
        return str(result.inserted_id)
    
    @classmethod
    def update(cls, id, data):
        data['updated_at'] = datetime.utcnow()
        return db[cls.collection].update_one(
            {"_id": ObjectId(id)},
            {"$set": data}
        )
    
    @classmethod
    def delete(cls, id):
        return db[cls.collection].delete_one({"_id": ObjectId(id)})


class HR(MongoModel):
    collection = 'hr_users'
    
    @classmethod
    def find_by_username(cls, username):
        return db[cls.collection].find_one({"username": username})
        
    @classmethod
    def create(cls, data):
        if 'created_at' not in data:
            data['created_at'] = datetime.utcnow()
        return super().create(data)


class Candidate(MongoModel):
    collection = 'candidates'
    
    @classmethod
    def find_by_username(cls, username):
        return db[cls.collection].find_one({"username": username})
    
    @classmethod
    def get_applications(cls, candidate_id):
        return list(db['applications'].find({"candidate_id": candidate_id}))
        
    @classmethod
    def create(cls, data):
        if 'created_at' not in data:
            data['created_at'] = datetime.utcnow()
        return super().create(data)


class Job(MongoModel):
    collection = 'jobs'


class Application(MongoModel):
    collection = 'applications'
    
    @classmethod
    def find_by_candidate_and_job(cls, candidate_id, job_id):
        return db[cls.collection].find_one({
            "candidate_id": candidate_id,
            "job_id": job_id
        })
    
    @classmethod
    def get_with_details(cls, query=None):
        if query is None:
            query = {}
        
        pipeline = [
            {"$match": query},
            {"$lookup": {
                "from": "candidates",
                "localField": "candidate_id",
                "foreignField": "_id",
                "as": "candidate"
            }},
            {"$unwind": "$candidate"},
            {"$lookup": {
                "from": "jobs",
                "localField": "job_id",
                "foreignField": "_id",
                "as": "job"
            }},
            {"$unwind": "$job"},
            {"$sort": {"created_at": -1}}
        ]
        
        return list(db[cls.collection].aggregate(pipeline))
    
    @classmethod
    def get_resume_file(cls, application_id):
        """
        Get the resume file for an application
        :param application_id: Application ID
        :return: File object or None if not found
        """
        application = cls.find_by_id(application_id)
        if not application or 'resume_file_id' not in application:
            return None
            
        return storage.get_file(application['resume_file_id'])
    
    @classmethod
    def get_resume_info(cls, application_id):
        """
        Get resume file info without loading the actual file
        :param application_id: Application ID
        :return: File info dictionary or None if not found
        """
        application = cls.find_by_id(application_id)
        if not application or 'resume_file_id' not in application:
            return None
            
        return storage.get_file_info(application['resume_file_id'])
