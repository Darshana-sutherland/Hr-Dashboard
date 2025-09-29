import os
from gridfs import GridFS
from bson import ObjectId
from mongo_models import db
from config import Config

class GridFSStorage:
    def __init__(self):
        self.fs = GridFS(db, collection=Config.FS_FILES_COLLECTION)
    
    def save_file(self, file, filename=None, content_type=None, metadata=None):
        """
        Save a file to GridFS
        :param file: FileStorage object or file-like object
        :param filename: Optional filename
        :param content_type: Optional content type
        :param metadata: Optional metadata dictionary
        :return: File ID as string
        """
        if not filename and hasattr(file, 'filename'):
            filename = file.filename
        
        if not content_type and hasattr(file, 'content_type'):
            content_type = file.content_type
        
        file_id = self.fs.put(
            file,
            filename=filename,
            content_type=content_type,
            metadata=metadata or {}
        )
        
        return str(file_id)
    
    def get_file(self, file_id):
        """
        Get a file from GridFS by ID
        :param file_id: File ID as string or ObjectId
        :return: GridOut object or None if not found
        """
        try:
            if not isinstance(file_id, ObjectId):
                file_id = ObjectId(file_id)
            return self.fs.get(file_id)
        except:
            return None
    
    def delete_file(self, file_id):
        """
        Delete a file from GridFS by ID
        :param file_id: File ID as string or ObjectId
        :return: True if deleted, False otherwise
        """
        try:
            if not isinstance(file_id, ObjectId):
                file_id = ObjectId(file_id)
            self.fs.delete(file_id)
            return True
        except:
            return False
    
    def get_file_info(self, file_id):
        """
        Get file metadata without loading the actual file
        :param file_id: File ID as string or ObjectId
        :return: File metadata dictionary or None if not found
        """
        try:
            if not isinstance(file_id, ObjectId):
                file_id = ObjectId(file_id)
            return db[Config.FS_FILES_COLLECTION].find_one({'_id': file_id})
        except:
            return None

# Global instance
storage = GridFSStorage()
